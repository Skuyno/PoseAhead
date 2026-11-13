"""Pinned, current-frame MeTRAbs PyTorch inference backend."""

from __future__ import annotations

import math
import subprocess
import sys
import time
import types
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from poseahead.artifacts import sha256_file
from poseahead.perception.camera_geometry import CameraIntrinsics
from poseahead.perception.metrabs_mapping import (
    METRABS_JOINT_NAMES,
    METRABS_SKELETON,
    metrabs_to_pose_frame,
)
from poseahead.perception.precision import CROP_PRECISIONS, set_convolution_storage

METRABS_COMMIT = "8b2b116dd27372e7dbd8207809f868df4e3f852e"
METRABS_MODEL_URL = (
    "https://omnomnom.vision.rwth-aachen.de/data/metrabs/"
    "metrabs_eff2s_256px_800k_28ds_pytorch.tar.gz"
)
METRABS_ARCHIVE_SHA256 = "d6861899d0c05a93a4bae078dca5382186f04d568255eb688ace1c7dea7a7634"
METRABS_MODEL_DIR = "metrabs_eff2s_256px_800k_28ds_pytorch"
YOLO_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
YOLO_MODEL_SHA256 = "31e20dde3def09e2cf938c7be6fe23d9150bbbe503982af13345706515f2ef95"
MODEL_FILE_HASHES = {
    "ckpt.pt": "82833e7b0d9bf282271e17aeae66e69cc2b1070a21d94540ad864313c477962a",
    "config.yaml": "147e259ef04457c0cfadd24a44a961057f78399b2da5aa5625627fb75aa95c94",
    "joint_info.npz": "de211b8679238955ada9a0f0b06bfccc2ea8fc5914d6feb9b3fd2d0b2b1cd97b",
    "joint_transform_matrix.npy": (
        "a1dd2b3c4807c1abd447300f142fe97673c40b510fac7d2cd2d6aa132a6a27cc"
    ),
}
H36M_INDICES = [48, 24, 25, 26, 29, 30, 31, 34, 35, 36, 37, 38, 39, 40, 43, 44, 45]
H36M_EDGES = [
    (1, 2),
    (1, 0),
    (2, 3),
    (4, 5),
    (4, 0),
    (5, 6),
    (7, 8),
    (7, 0),
    (8, 9),
    (8, 11),
    (8, 14),
    (9, 10),
    (11, 12),
    (12, 13),
    (14, 15),
    (15, 16),
]


@dataclass(frozen=True, slots=True)
class InferenceTiming:
    total_ms: float


class _NoPersonDetected(Exception):
    """The detector returned zero boxes for this single-frame request."""


class MetrabsStartupCancelled(Exception):
    """Cooperative cancellation at a safe model-initialization boundary."""


def _verify_file(path: Path, expected: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"required MeTRAbs file is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch for {path.name}: {actual} != {expected}")


class _JointInfo:
    """Inference-only subset of posepile.JointInfo used by the pinned source."""

    def __init__(self, names, edges=()):
        self.names = [str(name) for name in names]
        self.stick_figure_edges = [tuple(map(int, edge)) for edge in edges]
        self.n_joints = len(self.names)
        ids = {name: index for index, name in enumerate(self.names)}
        self.mirror_mapping = [ids[self._other_side(name)] for name in self.names]

    @staticmethod
    def _other_side(name: str) -> str:
        if name.startswith("l"):
            return "r" + name[1:]
        if name.startswith("r"):
            return "l" + name[1:]
        return name


def _install_inference_shims() -> None:
    """Avoid importing PosePile's dataset/TensorFlow stack for inference."""
    posepile = types.ModuleType("posepile")
    joint_info = types.ModuleType("posepile.joint_info")
    joint_info.JointInfo = _JointInfo
    datasets3d = types.ModuleType("posepile.datasets3d")
    paths = types.ModuleType("posepile.paths")
    paths.DATA_ROOT = ""
    posepile.joint_info = joint_info
    posepile.datasets3d = datasets3d
    posepile.paths = paths
    sys.modules.update(
        {
            "posepile": posepile,
            "posepile.joint_info": joint_info,
            "posepile.datasets3d": datasets3d,
            "posepile.paths": paths,
        }
    )
    sys.modules.setdefault("simplepyutils", types.ModuleType("simplepyutils"))

    # The pinned util module imports Hydra, but PoseAhead supplies the immutable
    # model config directly and never invokes Hydra composition.
    hydra = types.ModuleType("hydra")
    hydra.__path__ = []
    hydra_core = types.ModuleType("hydra.core")
    hydra_core.__path__ = []
    hydra_global = types.ModuleType("hydra.core.global_hydra")
    hydra.core = hydra_core
    hydra_core.global_hydra = hydra_global
    sys.modules.update(
        {"hydra": hydra, "hydra.core": hydra_core, "hydra.core.global_hydra": hydra_global}
    )


def _model_config() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        proc_side=256,
        stride_train=32,
        stride_test=32,
        centered_stride=False,
        legacy_centered_stride_bug=True,
        efficientnet_size="s",
        backbone="efficientnetv2-s",
        depth=8,
        box_size_mm=2200,
        weak_perspective=False,
        mix_3d_inside_fov=0.5,
        affine_weights=None,
        transform_coords=False,
        predict_all_and_latents=False,
        regularize_to_manifold=False,
    )


class MetrabsBackend:
    """Estimate one person's absolute 3D pose from the current RGB frame."""

    def __init__(
        self,
        *,
        device: str = "cuda:0",
        confidence_threshold: float = 0.3,
        num_aug: int = 1,
        internal_batch_size: int = 8,
        detector_interval: int = 10,
        max_input_side: int | None = 640,
        default_fov_degrees: float = 55.0,
        camera_intrinsics: CameraIntrinsics | None = None,
        crop_precision: str = "autocast",
        repository_root: Path | None = None,
        model=None,
        on_startup: Callable[[str], None] | None = None,
        stop_requested: Callable[[], bool] | None = None,
    ) -> None:
        if not isinstance(device, str) or not device.lower().startswith("cuda"):
            raise ValueError("the pinned MeTRAbs PyTorch backend currently requires CUDA")
        if not isinstance(crop_precision, str) or crop_precision not in CROP_PRECISIONS:
            raise ValueError("crop_precision must be autocast or fp16_conv")
        if (
            type(confidence_threshold) not in (int, float)
            or not math.isfinite(confidence_threshold)
            or not 0 < confidence_threshold <= 1
        ):
            raise ValueError("confidence_threshold must be in (0, 1]")
        if type(num_aug) is not int or num_aug < 1:
            raise ValueError("num_aug must be a positive integer")
        if type(internal_batch_size) is not int or internal_batch_size < num_aug:
            raise ValueError("internal_batch_size must be an integer not smaller than num_aug")
        if type(detector_interval) is not int or detector_interval < 1:
            raise ValueError("detector_interval must be a positive integer")
        if max_input_side is not None and (
            type(max_input_side) is not int or max_input_side < 64
        ):
            raise ValueError("max_input_side must be null or an integer of at least 64")
        if (
            type(default_fov_degrees) not in (int, float)
            or not math.isfinite(default_fov_degrees)
            or not 1 < default_fov_degrees < 179
        ):
            raise ValueError("default_fov_degrees must be finite and in (1, 179)")
        self.repository_root = repository_root or Path(__file__).resolve().parents[3]
        self._device = device.lower()
        self._confidence_threshold = float(confidence_threshold)
        self._num_aug = num_aug
        self._internal_batch_size = internal_batch_size
        self._detector_interval = detector_interval
        self._max_input_side = max_input_side
        self._default_fov_degrees = float(default_fov_degrees)
        self._camera_intrinsics = camera_intrinsics
        self._crop_precision = crop_precision
        self._last_timing: InferenceTiming | None = None
        self._last_box_xywh: np.ndarray | None = None
        self._last_detector_score: float | None = None
        self._bbox_age_processed_frames = 0
        self._last_frame_id: int | None = None
        self._last_inference_size: tuple[int, int] | None = None
        self._on_startup = on_startup
        self._stop_requested = stop_requested
        if model is None:
            self._torch, self._model = self._load_model()
        else:
            self._torch, self._model = None, model

    @property
    def last_timing(self) -> InferenceTiming | None:
        return self._last_timing

    def _startup(self, message: str) -> None:
        if self._stop_requested is not None and self._stop_requested():
            raise MetrabsStartupCancelled()
        if self._on_startup is not None:
            self._on_startup(message)
        if self._stop_requested is not None and self._stop_requested():
            raise MetrabsStartupCancelled()

    def infer(self, image, *, frame_id: int, source_timestamp_ns: int):
        import cv2

        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be a readable three-channel BGR image")
        source_height, source_width = image.shape[:2]
        if self._camera_intrinsics is not None and (
            self._camera_intrinsics.width,
            self._camera_intrinsics.height,
        ) != (source_width, source_height):
            raise ValueError("camera intrinsics image size does not match the input image")
        scale = 1.0
        if (
            self._max_input_side is not None
            and max(source_width, source_height) > self._max_input_side
        ):
            scale = self._max_input_side / max(source_width, source_height)
            inference_size = (
                max(1, round(source_width * scale)),
                max(1, round(source_height * scale)),
            )
            image = cv2.resize(image, inference_size, interpolation=cv2.INTER_AREA)
        height, width = image.shape[:2]
        rgb = np.ascontiguousarray(image[..., ::-1].transpose(2, 0, 1))
        intrinsic = self._intrinsic_matrix(
            width, height, source_width=source_width, source_height=source_height
        )
        stream_restarted = self._last_frame_id is not None and frame_id <= self._last_frame_id
        geometry_changed = self._last_inference_size not in (None, (width, height))
        if stream_restarted or geometry_changed:
            self._last_box_xywh = None
            self._last_detector_score = None
            self._bbox_age_processed_frames = 0
        detect_person = (
            self._last_box_xywh is None
            or self._bbox_age_processed_frames >= self._detector_interval - 1
        )

        started = time.perf_counter_ns()
        try:
            if self._torch is None:
                # Injectable test/dummy models use the public detection path on every frame.
                detect_person = True
                prediction = self._model.detect_poses(
                    rgb,
                    intrinsic_matrix=intrinsic,
                    detector_threshold=self._confidence_threshold,
                    max_detections=1,
                    num_aug=self._num_aug,
                    internal_batch_size=self._internal_batch_size,
                    skeleton=METRABS_SKELETON,
                )
            else:
                tensor = self._torch.from_numpy(rgb).to(self._device)
                with self._torch.inference_mode(), self._torch.device(self._device):
                    if detect_person:
                        prediction = self._model.detect_poses(
                            tensor,
                            intrinsic_matrix=intrinsic,
                            detector_threshold=self._confidence_threshold,
                            max_detections=1,
                            num_aug=self._num_aug,
                            internal_batch_size=self._internal_batch_size,
                            skeleton=METRABS_SKELETON,
                            suppress_implausible_poses=False,
                        )
                    else:
                        tracked_box = self._torch.as_tensor(
                            self._last_box_xywh[np.newaxis],
                            dtype=self._torch.float32,
                            device=self._device,
                        )
                        tracked_boxes = [
                            self._torch.cat(
                                [tracked_box, self._torch.ones_like(tracked_box[:, :1])], dim=-1
                            )
                        ]
                        # The pinned upstream public estimate_poses wrappers mishandle their
                        # box container. Calling the pinned internal method preserves exactly
                        # the same crop/pose path while supplying the expected list-of-tensors.
                        batched_prediction = self._model._estimate_poses_batched(
                            tensor[np.newaxis],
                            tracked_boxes,
                            self._torch.as_tensor(
                                intrinsic[np.newaxis],
                                dtype=self._torch.float32,
                                device=self._device,
                            ),
                            self._torch.zeros(
                                (1, 5), dtype=self._torch.float32, device=self._device
                            ),
                            self._torch.eye(4, dtype=self._torch.float32, device=self._device)[
                                np.newaxis
                            ],
                            self._torch.tensor(
                                (0, -1, 0), dtype=self._torch.float32, device=self._device
                            ),
                            self._default_fov_degrees,
                            self._internal_batch_size,
                            1,
                            self._num_aug,
                            True,
                            METRABS_SKELETON,
                            False,
                        )
                        prediction = {
                            key: value[0] for key, value in batched_prediction.items()
                        }
                self._torch.cuda.synchronize(self._device)
        except _NoPersonDetected:
            if self._torch is not None:
                self._torch.cuda.synchronize(self._device)
            produced = time.perf_counter_ns()
            self._last_timing = InferenceTiming((produced - started) / 1e6)
            self._last_box_xywh = None
            self._last_detector_score = None
            self._bbox_age_processed_frames = 0
            self._last_frame_id = frame_id
            self._last_inference_size = (width, height)
            return None
        produced = time.perf_counter_ns()
        self._last_timing = InferenceTiming((produced - started) / 1e6)
        if detect_person:
            boxes = self._numpy(prediction["boxes"])
        else:
            boxes = np.concatenate(
                [
                    self._last_box_xywh[np.newaxis],
                    np.array([[self._last_detector_score]], dtype=np.float32),
                ],
                axis=1,
            )
        poses = self._numpy(prediction["poses3d"])
        poses2d = self._numpy(prediction["poses2d"])
        if (
            boxes.ndim != 2
            or boxes.shape[1] < 5
            or poses.ndim != 3
            or poses.shape[1:] != (17, 3)
            or boxes.shape[0] != poses.shape[0]
            or poses2d.ndim != 3
            or poses2d.shape[0] != poses.shape[0]
            or poses2d.shape[1] != poses.shape[1]
            or poses2d.shape[2] < 2
        ):
            raise RuntimeError("unexpected MeTRAbs result shapes")
        usable = np.flatnonzero(
            np.isfinite(boxes[:, :5]).all(axis=1)
            & (boxes[:, 2] > 0)
            & (boxes[:, 3] > 0)
            & (boxes[:, 4] >= self._confidence_threshold)
            & np.isfinite(poses).all(axis=(1, 2))
            & np.isfinite(poses2d).all(axis=(1, 2))
        )
        if usable.size == 0:
            # A failed tracked pose must force detection on the very next frame.
            # Otherwise the age counter never advances and the stale box persists forever.
            self._last_box_xywh = None
            self._last_detector_score = None
            self._bbox_age_processed_frames = 0
            self._last_frame_id = frame_id
            self._last_inference_size = (width, height)
            return None
        selected = int(usable[np.argmax(boxes[usable, 4])])
        tracking_box_updated = False
        if detect_person:
            self._last_box_xywh = boxes[selected, :4].astype(np.float32, copy=True)
            self._last_detector_score = float(boxes[selected, 4])
            self._bbox_age_processed_frames = 0
        else:
            self._bbox_age_processed_frames += 1
            updated_box = self._box_from_pose2d(poses2d[selected], width, height)
            if updated_box is not None:
                self._last_box_xywh = updated_box
                tracking_box_updated = True
            else:
                self._last_box_xywh = None
        self._last_frame_id = frame_id
        self._last_inference_size = (width, height)
        camera = self._intrinsic_metadata(
            width, height, source_width=source_width, source_height=source_height
        )
        return metrabs_to_pose_frame(
            poses[selected],
            detection_confidence=float(boxes[selected, 4]),
            confidence_threshold=self._confidence_threshold,
            frame_id=frame_id,
            source_timestamp_ns=source_timestamp_ns,
            produced_timestamp_ns=produced,
            metadata={
                "backend": "metrabs-eff2s-pytorch",
                "root_translation_status": "estimated_monocular",
                "scale_status": "metrabs_metric_estimate",
                "intrinsics_source": "calibrated" if self._camera_intrinsics else "assumed_fov",
                "camera_intrinsics": camera,
                "source_image_size": [source_width, source_height],
                "inference_image_size": [width, height],
                "input_scale": scale,
                "default_fov_degrees": self._default_fov_degrees,
                "num_aug": self._num_aug,
                "pose_crop_size": 256,
                "crop_precision": self._crop_precision,
                "detector_interval": self._detector_interval,
                "detector_mode": "detected_current_frame" if detect_person else "reused_past_box",
                "bbox_age_processed_frames": self._bbox_age_processed_frames,
                "tracking_box_updated": tracking_box_updated,
                "detector_score": float(boxes[selected, 4]),
                "bbox_xywh": boxes[selected, :4].astype(float).tolist(),
                "inference_ms": self._last_timing.total_ms,
            },
        )

    @staticmethod
    def _numpy(value) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    @staticmethod
    def _box_from_pose2d(
        pose2d: np.ndarray, width: int, height: int
    ) -> np.ndarray | None:
        points = np.asarray(pose2d, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 2:
            return None
        points = points[np.isfinite(points[:, :2]).all(axis=1), :2]
        if len(points) < 4:
            return None
        low = points.min(axis=0)
        high = points.max(axis=0)
        extent = high - low
        if np.any(extent < 8):
            return None
        center = (low + high) / 2
        expanded = extent * 1.5
        x1 = max(0.0, float(center[0] - expanded[0] / 2))
        y1 = max(0.0, float(center[1] - expanded[1] / 2))
        x2 = min(float(width), float(center[0] + expanded[0] / 2))
        y2 = min(float(height), float(center[1] + expanded[1] / 2))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)

    def _intrinsic_matrix(
        self,
        width: int,
        height: int,
        *,
        source_width: int | None = None,
        source_height: int | None = None,
    ) -> np.ndarray:
        values = self._intrinsic_metadata(
            width, height, source_width=source_width, source_height=source_height
        )
        return np.array(
            [[values["fx"], 0, values["cx"]], [0, values["fy"], values["cy"]], [0, 0, 1]],
            dtype=np.float32,
        )

    def _intrinsic_metadata(
        self,
        width: int,
        height: int,
        *,
        source_width: int | None = None,
        source_height: int | None = None,
    ) -> dict[str, float | int]:
        if self._camera_intrinsics is not None:
            camera = self._camera_intrinsics
            source_width = source_width or width
            source_height = source_height or height
            scale_x = width / source_width
            scale_y = height / source_height
            return {
                "fx": camera.fx * scale_x,
                "fy": camera.fy * scale_y,
                "cx": camera.cx * scale_x,
                "cy": camera.cy * scale_y,
                "width": width,
                "height": height,
            }
        focal = max(width, height) / (2 * math.tan(math.radians(self._default_fov_degrees) / 2))
        return {
            "fx": focal,
            "fy": focal,
            "cx": width / 2,
            "cy": height / 2,
            "width": width,
            "height": height,
        }

    def _load_model(self):
        self._startup("Проверка исходников MeTRAbs")
        source = self.repository_root / ".vendor" / "metrabs-source"
        model_dir = self.repository_root / ".vendor" / METRABS_MODEL_DIR
        yolo_path = self.repository_root / ".vendor" / "yolov8n.pt"
        if not (source / ".git").is_dir():
            raise RuntimeError("MeTRAbs source is missing; run scripts/setup_metrabs.ps1")
        try:
            revision = subprocess.run(
                ["git", "-C", str(source), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            ).stdout.strip()
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                "Проверка исходников MeTRAbs: Git не ответил за 10 секунд"
            ) from error
        if revision != METRABS_COMMIT:
            raise RuntimeError(f"unexpected MeTRAbs revision: {revision} != {METRABS_COMMIT}")
        self._startup("Проверка локальных весов MeTRAbs")
        for filename, expected in MODEL_FILE_HASHES.items():
            _verify_file(model_dir / filename, expected)
        _verify_file(yolo_path, YOLO_MODEL_SHA256)
        self._startup("Загрузка PyTorch и детектора YOLO")
        try:
            import torch
            import ultralytics
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError(
                "MeTRAbs dependencies are missing; run scripts/setup_metrabs.ps1"
            ) from error
        self._startup("Проверка CUDA и загрузка модулей MeTRAbs")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required by the pinned MeTRAbs PyTorch implementation")
        torch.backends.cudnn.benchmark = True

        _install_inference_shims()
        source_text = str(source)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        import torchvision
        from metrabs_pytorch import util
        from metrabs_pytorch.backbones import efficientnet
        from metrabs_pytorch.models import metrabs as metrabs_model
        from metrabs_pytorch.multiperson import multiperson_model, person_detector

        self._startup("Подготовка модели MeTRAbs на GPU")
        util._cfg = _model_config()
        joint_data = np.load(model_dir / "joint_info.npz", allow_pickle=False)
        joint_info = _JointInfo(joint_data["joint_names"], joint_data["joint_edges"])
        with torch.device(self._device):
            backbone_raw = efficientnet.efficientnet_v2_s()
            backbone = torch.nn.Sequential(efficientnet.PreprocLayer(), backbone_raw.features)
            crop_model = metrabs_model.Metrabs(backbone, joint_info).eval()
            crop_model(
                (
                    torch.zeros((1, 3, 256, 256), dtype=torch.float32),
                    torch.eye(3, dtype=torch.float32)[np.newaxis],
                )
            )
        self._startup("Загрузка весов MeTRAbs на GPU")
        state = torch.load(model_dir / "ckpt.pt", map_location=self._device, weights_only=True)
        crop_model.load_state_dict(state)
        crop_model = crop_model.to(
            self._device, memory_format=torch.channels_last
        ).eval()
        if self._crop_precision == "fp16_conv":
            self._startup("Подготовка FP16 весов свёрток; нормализация остаётся FP32")
            set_convolution_storage(crop_model, self._crop_precision, torch)

        self._startup("Подготовка детектора людей YOLO")
        original_detector = person_detector.PersonDetector

        class PinnedPersonDetector(original_detector):
            def __init__(detector_self):
                torch.nn.Module.__init__(detector_self)
                detector_self.input_size = 320
                detector_self.yolo = ultralytics.YOLO(str(yolo_path))

            def forward(detector_self, images, threshold, nms_iou_threshold, max_detections):
                height, width = images.shape[2:4]
                factor = detector_self.input_size / max(height, width)
                target_height = int(factor * height)
                target_width = int(factor * width)
                resized = (images.float() / 255) ** 2.2
                resized = torchvision.transforms.functional.resize(
                    resized, (target_height, target_width), antialias=factor < 1
                )
                resized **= 1 / 2.2
                resized.clamp_(0, 1)
                pad_height = -target_height % 32
                pad_width = -target_width % 32
                half_pad_height = pad_height // 2
                half_pad_width = pad_width // 2
                resized = torch.nn.functional.pad(
                    resized,
                    (
                        half_pad_width,
                        pad_width - half_pad_width,
                        half_pad_height,
                        pad_height - half_pad_height,
                    ),
                    value=0.5,
                )
                with torch.autocast(dtype=torch.float16, device_type="cuda"):
                    results = detector_self.yolo.predict(
                        source=resized,
                        conf=threshold,
                        iou=nms_iou_threshold,
                        max_det=max_detections,
                        classes=[0],
                        verbose=False,
                    )
                y_factor = height / target_height
                x_factor = width / target_width
                if len(results) == 1 and len(results[0].boxes) == 0:
                    raise _NoPersonDetected()
                return [
                    person_detector.scale_boxes(
                        result.boxes,
                        float(half_pad_width),
                        float(half_pad_height),
                        x_factor,
                        y_factor,
                    )
                    for result in results
                ]

        person_detector.PersonDetector = PinnedPersonDetector
        try:
            model = multiperson_model.Pose3dEstimator(
                crop_model,
                {
                    METRABS_SKELETON: {
                        "indices": H36M_INDICES,
                        "names": list(METRABS_JOINT_NAMES),
                        "edges": H36M_EDGES,
                    }
                },
                np.load(model_dir / "joint_transform_matrix.npy", allow_pickle=False),
            )
            # Upstream stores this as a plain Tensor rather than a registered buffer.
            model.joint_transform_matrix = model.joint_transform_matrix.to(self._device)
        finally:
            person_detector.PersonDetector = original_detector
        return torch, model.to(self._device).eval()
