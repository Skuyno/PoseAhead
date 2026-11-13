import subprocess
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import poseahead.perception.metrabs as module
from poseahead.perception.camera_geometry import CameraIntrinsics


class FakeModel:
    def __init__(self):
        self.boxes = np.array([[1, 2, 30, 40, 0.9]], dtype=np.float32)
        self.poses = np.zeros((1, 17, 3), dtype=np.float32)
        self.poses[..., 2] = 3000
        self.call = None

    def detect_poses(self, image, **kwargs):
        self.call = SimpleNamespace(image=image, kwargs=kwargs)
        poses2d = np.zeros((*self.poses.shape[:2], 2), dtype=np.float32)
        return {"boxes": self.boxes, "poses3d": self.poses, "poses2d": poses2d}


def test_backend_converts_bgr_selects_person_and_emits_absolute_root():
    model = FakeModel()
    model.boxes = np.array([[0, 0, 1, 1, 0.8], [1, 2, 3, 4, 0.9]], dtype=np.float32)
    model.poses = np.zeros((2, 17, 3), dtype=np.float32)
    model.poses[:, :, 2] = 3000
    model.poses[1, 0] = [100, 200, 4000]
    backend = module.MetrabsBackend(model=model)
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[0, 0] = [1, 2, 3]
    frame = backend.infer(image, frame_id=4, source_timestamp_ns=0)
    np.testing.assert_array_equal(model.call.image[:, 0, 0], [3, 2, 1])
    assert model.call.image.shape == (3, 4, 6)
    assert model.call.kwargs["skeleton"] == "h36m_17"
    assert model.call.kwargs["max_detections"] == 1
    np.testing.assert_allclose(frame.root_position, [0.1, 0.2, 4])
    assert frame.metadata["root_translation_status"] == "estimated_monocular"
    assert frame.metadata["scale_status"] == "metrabs_metric_estimate"
    assert frame.metadata["pose_crop_size"] == 256
    assert backend.last_timing.total_ms >= 0


def test_calibrated_intrinsics_are_passed_exactly_and_size_checked():
    model = FakeModel()
    camera = CameraIntrinsics(500, 510, 3, 2, 6, 4)
    backend = module.MetrabsBackend(model=model, camera_intrinsics=camera)
    backend.infer(np.zeros((4, 6, 3), dtype=np.uint8), frame_id=0, source_timestamp_ns=0)
    np.testing.assert_allclose(
        model.call.kwargs["intrinsic_matrix"], [[500, 0, 3], [0, 510, 2], [0, 0, 1]]
    )
    with pytest.raises(ValueError, match="image size"):
        backend.infer(np.zeros((5, 6, 3), dtype=np.uint8), frame_id=1, source_timestamp_ns=0)


def test_assumed_fov_intrinsics_match_image_geometry():
    model = FakeModel()
    backend = module.MetrabsBackend(model=model, default_fov_degrees=90)
    frame = backend.infer(np.zeros((4, 6, 3), dtype=np.uint8), frame_id=0, source_timestamp_ns=0)
    np.testing.assert_allclose(
        model.call.kwargs["intrinsic_matrix"], [[3, 0, 3], [0, 3, 2], [0, 0, 1]]
    )
    assert frame.metadata["intrinsics_source"] == "assumed_fov"


def test_large_input_and_calibrated_intrinsics_are_scaled_together():
    model = FakeModel()
    camera = CameraIntrinsics(1000, 900, 100, 50, 200, 100)
    backend = module.MetrabsBackend(
        model=model, camera_intrinsics=camera, max_input_side=100
    )
    frame = backend.infer(
        np.zeros((100, 200, 3), dtype=np.uint8), frame_id=0, source_timestamp_ns=0
    )
    assert model.call.image.shape == (3, 50, 100)
    np.testing.assert_allclose(
        model.call.kwargs["intrinsic_matrix"],
        [[500, 0, 50], [0, 450, 25], [0, 0, 1]],
    )
    assert frame.metadata["source_image_size"] == (200, 100)
    assert frame.metadata["inference_image_size"] == (100, 50)
    assert frame.metadata["input_scale"] == 0.5


def test_missing_or_malformed_detection_returns_none_or_fails_explicitly():
    model = FakeModel()
    backend = module.MetrabsBackend(model=model)
    model.boxes = np.empty((0, 5), dtype=np.float32)
    model.poses = np.empty((0, 17, 3), dtype=np.float32)
    assert backend.infer(np.zeros((4, 6, 3)), frame_id=0, source_timestamp_ns=0) is None
    model.boxes = np.zeros((1, 4))
    with pytest.raises(RuntimeError, match="result shapes"):
        backend.infer(np.zeros((4, 6, 3)), frame_id=1, source_timestamp_ns=0)


@pytest.mark.parametrize(
    "values",
    [
        {"device": "cpu"},
        {"confidence_threshold": 0},
        {"confidence_threshold": float("nan")},
        {"num_aug": 0},
        {"num_aug": True},
        {"num_aug": 2, "internal_batch_size": 1},
        {"detector_interval": 0},
        {"detector_interval": True},
        {"max_input_side": 32},
        {"max_input_side": True},
        {"default_fov_degrees": 180},
        {"crop_precision": "fp16"},
        {"crop_precision": True},
    ],
)
def test_invalid_options_are_rejected_before_model_use(values):
    with pytest.raises(ValueError):
        module.MetrabsBackend(model=FakeModel(), **values)


def test_file_verification_rejects_corruption(tmp_path: Path):
    path = tmp_path / "weights.pt"
    path.write_bytes(b"expected")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        module._verify_file(path, "0" * 64)


def test_startup_cancel_is_checked_before_files_or_gpu(tmp_path):
    with pytest.raises(module.MetrabsStartupCancelled):
        module.MetrabsBackend(repository_root=tmp_path, stop_requested=lambda: True)


def test_startup_callback_can_cancel_before_first_operation(tmp_path):
    messages = []
    with pytest.raises(module.MetrabsStartupCancelled):
        module.MetrabsBackend(
            repository_root=tmp_path, on_startup=messages.append,
            stop_requested=lambda: bool(messages),
        )
    assert messages == ["Проверка исходников MeTRAbs"]


def test_git_verification_has_isolated_stdin_and_a_timeout(tmp_path, monkeypatch):
    (tmp_path / ".vendor/metrabs-source/.git").mkdir(parents=True)
    calls = []

    def revision(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout=module.METRABS_COMMIT + "\n")

    monkeypatch.setattr(module.subprocess, "run", revision)
    with pytest.raises(RuntimeError, match="required MeTRAbs file is missing"):
        module.MetrabsBackend(repository_root=tmp_path)
    assert len(calls) == 1
    assert calls[0][0][-2:] == ["rev-parse", "HEAD"]
    assert calls[0][1]["stdin"] == subprocess.DEVNULL
    assert calls[0][1]["timeout"] == 10


def test_git_verification_timeout_is_reported_clearly(tmp_path, monkeypatch):
    (tmp_path / ".vendor/metrabs-source/.git").mkdir(parents=True)

    def timeout(args, **kwargs):
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    monkeypatch.setattr(module.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="Git не ответил за 10 секунд"):
        module.MetrabsBackend(repository_root=tmp_path)


def test_pose_box_tracker_expands_and_clips_joint_bounds():
    pose2d = np.array([[10, 20], [30, 60], [20, 40], [25, 50]], dtype=np.float32)
    box = module.MetrabsBackend._box_from_pose2d(pose2d, width=35, height=70)
    np.testing.assert_allclose(box, [5, 10, 30, 60])


def test_pose_box_tracker_rejects_invalid_or_degenerate_points():
    assert module.MetrabsBackend._box_from_pose2d(np.zeros((3, 2)), 100, 100) is None
    assert module.MetrabsBackend._box_from_pose2d(np.zeros((17, 2)), 100, 100) is None


class TrackingModel(FakeModel):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.fail_tracking = False

    def detect_poses(self, image, **kwargs):
        self.calls.append("detect")
        return super().detect_poses(image, **kwargs)

    def _estimate_poses_batched(self, *args):
        self.calls.append("track")
        poses = self.poses.copy()
        if self.fail_tracking:
            poses[:] = np.nan
        points = np.tile([[10, 20], [30, 60]], (9, 1))[:17]
        return {"poses3d": [poses], "poses2d": [points[np.newaxis]]}


def tracked_backend(model, interval=3):
    """Exercise the production tracking branch without CUDA or model downloads."""
    backend = module.MetrabsBackend(model=model, detector_interval=interval)
    backend._torch = SimpleNamespace(
        from_numpy=lambda array: SimpleNamespace(to=lambda device: array),
        inference_mode=nullcontext,
        device=lambda device: nullcontext(),
        float32=np.float32,
        as_tensor=lambda data, **kwargs: np.asarray(data, dtype=kwargs.get("dtype")),
        tensor=lambda data, **kwargs: np.asarray(data, dtype=kwargs.get("dtype")),
        zeros=lambda shape, **kwargs: np.zeros(shape, dtype=kwargs.get("dtype")),
        eye=lambda size, **kwargs: np.eye(size, dtype=kwargs.get("dtype")),
        cat=lambda arrays, dim: np.concatenate(arrays, axis=dim),
        ones_like=np.ones_like,
        cuda=SimpleNamespace(synchronize=lambda device: None),
    )
    return backend


def test_failed_tracked_pose_forces_redetection_and_recovers():
    model = TrackingModel()
    backend = tracked_backend(model)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    assert backend.infer(image, frame_id=0, source_timestamp_ns=0) is not None
    model.fail_tracking = True
    assert backend.infer(image, frame_id=1, source_timestamp_ns=0) is None
    assert backend.infer(image, frame_id=2, source_timestamp_ns=0) is not None
    assert model.calls == ["detect", "track", "detect"]


def test_tracking_interval_and_restart_force_detection():
    model = TrackingModel()
    backend = tracked_backend(model)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    for frame_id in (0, 1, 2, 3, 4, 0):
        assert backend.infer(image, frame_id=frame_id, source_timestamp_ns=0) is not None
    assert model.calls == ["detect", "track", "track", "detect", "track", "detect"]


def test_empty_detector_is_handled_but_unrelated_runtime_failure_is_not_hidden():
    model = FakeModel()
    backend = module.MetrabsBackend(model=model)
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    def no_person(*args, **kwargs):
        raise module._NoPersonDetected()

    model.detect_poses = no_person
    assert backend.infer(image, frame_id=0, source_timestamp_ns=0) is None
    assert backend.last_timing is not None

    def broken_model(*args, **kwargs):
        raise RuntimeError("torch.cat(): expected a non-empty list of Tensors")

    model.detect_poses = broken_model
    with pytest.raises(RuntimeError, match="non-empty"):
        backend.infer(image, frame_id=1, source_timestamp_ns=0)
