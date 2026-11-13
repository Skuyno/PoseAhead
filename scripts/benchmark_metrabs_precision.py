"""Paired recorded-frame speed/numerical probe, not live or ground-truth acceptance.

Compare the existing autocast path with FP16 convolution storage while retaining
FP32 BatchNorm storage and the original geometric path. Original weights are restored from
a CPU clone before every block. ABBA order, fixed input prefix and frame-ID restart
make tracking schedules comparable. No camera, sockets, downloads or saved-weight
mutation. The candidate is experimental and is not automatically promoted.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from poseahead.artifacts import sha256_file
from poseahead.config import LiveConfig, load_live_config
from poseahead.perception.metrabs import MetrabsBackend
from poseahead.perception.precision import set_convolution_storage
from poseahead.runtime.benchmark import BenchmarkStats, latency_summary


def compare_poses(reference, candidate):
    if len(reference) != len(candidate):
        raise ValueError("paired frame counts differ")
    body, root, absolute = [], [], []
    exact, status_match, schedule_match, paired = True, True, True, 0
    for a, b in zip(reference, candidate, strict=True):
        if (a is None) != (b is None):
            exact = status_match = False
        if a is None or b is None:
            continue
        status_match &= np.array_equal(a.joint_validity, b.joint_validity)
        schedule_match &= a.metadata["detector_mode"] == b.metadata["detector_mode"]
        has_roots = a.root_position is not None and b.root_position is not None
        status_match &= (a.root_position is None) == (b.root_position is None)
        exact &= np.array_equal(a.joint_positions, b.joint_positions)
        exact &= np.array_equal(a.joint_confidence, b.joint_confidence)
        exact &= (a.root_position is None and b.root_position is None
                  or has_roots and np.array_equal(a.root_position, b.root_position))
        valid = a.joint_validity & b.joint_validity
        body.extend(np.linalg.norm(a.joint_positions[valid] - b.joint_positions[valid], axis=-1)
                    .astype(float).tolist())
        if has_roots:
            root.append(float(np.linalg.norm(a.root_position - b.root_position)))
            absolute.extend(np.linalg.norm(
                (a.joint_positions[valid] + a.root_position)
                - (b.joint_positions[valid] + b.root_position), axis=-1,
            ).astype(float).tolist())
        paired += 1
    return {
        "paired_frames": paired, "validity_and_presence_match": bool(status_match),
        "detection_schedule_match": bool(schedule_match), "arrays_exact": bool(exact),
        "body_delta_mm": latency_summary([x * 1000 for x in body]),
        "root_delta_mm": latency_summary([x * 1000 for x in root]),
        "absolute_joint_delta_mm": latency_summary([x * 1000 for x in absolute]),
        "scope": "numerical differences from existing inference, not ground-truth pose error",
    }


def run_probe(input_path, output, *, config, frames, warmup, backend_factory=MetrabsBackend):
    import cv2
    import torch

    input_path, output = Path(input_path), Path(output)
    if output.exists():
        raise FileExistsError("choose a new output directory")
    if not input_path.is_file():
        raise ValueError("recorded video file is required; camera indexes/URLs are not accepted")
    if type(frames) is not int or not 1 <= frames <= 600:
        raise ValueError("frames must be between 1 and 600")
    if type(warmup) is not int or not 1 <= warmup <= 600:
        raise ValueError("warmup must be between 1 and 600")
    capture = cv2.VideoCapture(str(input_path))
    images = []
    try:
        if not capture.isOpened():
            raise ValueError("could not open recorded video")
        for _ in range(max(frames, warmup)):
            ok, image = capture.read()
            if not ok:
                break
            images.append(image)
    finally:
        capture.release()
    if len(images) < max(frames, warmup):
        raise ValueError("video is shorter than requested prefix")
    source_hash = sha256_file(input_path)
    backend = backend_factory(
        confidence_threshold=config.confidence_threshold, num_aug=config.num_aug,
        internal_batch_size=config.internal_batch_size, detector_interval=config.detector_interval,
        max_input_side=config.max_input_side, default_fov_degrees=config.default_fov_degrees,
        camera_intrinsics=config.camera_intrinsics,
        crop_precision="autocast",  # the reference clone must retain original FP32 weights
    )
    crop = backend._model.crop_model
    original = {key: value.detach().cpu().clone() for key, value in crop.state_dict().items()}
    runs, poses = [], []
    try:
        for index, precision in enumerate(("autocast", "fp16_conv", "fp16_conv", "autocast")):
            set_convolution_storage(crop, "autocast", torch)
            crop.load_state_dict(original)
            layers = set_convolution_storage(crop, precision, torch)
            for frame_id, image in enumerate(images[:warmup]):
                backend.infer(image, frame_id=frame_id, source_timestamp_ns=time.perf_counter_ns())
            # Restart IDs to trigger the backend's normal stream reset after warmup.
            stats, output_poses = BenchmarkStats(), []
            torch.cuda.synchronize()
            started = time.perf_counter()
            for frame_id, image in enumerate(images[:frames]):
                begin = time.perf_counter_ns()
                pose = backend.infer(image, frame_id=frame_id, source_timestamp_ns=begin)
                stats.observe(pose, (time.perf_counter_ns() - begin) / 1e6)
                output_poses.append(pose)
            report = stats.report(time.perf_counter() - started)
            runs.append({"block": index, "precision": precision, "converted_layers": layers,
                         "metrics": report})
            poses.append(output_poses)
            print(f"Block {index} {precision}: {report['valid_pose_fps']:.2f} valid FPS, "
                  f"p95 {report['all_results_latency_ms']['p95']:.2f} ms", flush=True)
    finally:
        set_convolution_storage(crop, "autocast", torch)
        crop.load_state_dict(original)
    comparisons = {str(index): compare_poses(poses[0], poses[index]) for index in range(1, 4)}
    report = {
        "schema_version": 1, "input": str(input_path.resolve()), "input_sha256": source_hash,
        "config": config.to_dict(), "measured_frames_per_block": frames, "warmup": warmup,
        "initial_precision_override": "autocast",
        "order": "ABBA", "runs": runs, "comparisons_to_first_baseline": comparisons,
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(),
                        "cpu_threads": torch.get_num_threads()},
        "benchmark_script_sha256": sha256_file(Path(__file__)),
        "backend_source_sha256": sha256_file(
            Path(__file__).resolve().parents[1] / "src/poseahead/perception/metrabs.py"),
        "precision_helper_sha256": sha256_file(
            Path(__file__).resolve().parents[1] / "src/poseahead/perception/precision.py"),
        "scope": "predecoded recorded prefix; backend return timing, no capture/decode/UDP/Unity",
        "acceptance_status": "not_assessed", "automatic_promotion": False,
    }
    if sha256_file(input_path) != source_hash:
        raise ValueError("input changed during probe")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    arrays = {}
    for index, frames_out in enumerate(poses):
        arrays[f"block{index}_present"] = np.array([p is not None for p in frames_out])
        arrays[f"block{index}_validity"] = np.array([
            p.joint_validity if p is not None else np.zeros(17, dtype=bool) for p in frames_out
        ])
        arrays[f"block{index}_confidence"] = np.array([
            p.joint_confidence if p is not None else np.zeros(17, dtype=np.float32)
            for p in frames_out
        ], dtype=np.float32)
        arrays[f"block{index}_detected"] = np.array([
            p is not None and p.metadata["detector_mode"] == "detected_current_frame"
            for p in frames_out
        ])
        for key, shape in (("body", (frames, 17, 3)), ("root", (frames, 3))):
            values = np.full(shape, np.nan, dtype=np.float32)
            for i, pose in enumerate(frames_out):
                value = ((pose.joint_positions if key == "body" else pose.root_position)
                         if pose else None)
                if value is not None:
                    values[i] = value
            arrays[f"block{index}_{key}"] = values
    with (output / "poses.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with (output / "completion.json").open("x", encoding="utf-8") as stream:
        json.dump({"format": "poseahead-precision-probe-v1", "sha256": {
            name: sha256_file(output / name) for name in ("report.json", "poses.npz")
        }}, stream, indent=2)
    return report


def verify_probe(directory):
    directory = Path(directory)
    completion = json.loads((directory / "completion.json").read_text(encoding="utf-8"))
    if (completion.get("format") != "poseahead-precision-probe-v1"
            or set(completion.get("sha256", {})) != {"report.json", "poses.npz"}):
        raise ValueError("invalid precision completion manifest")
    for name, digest in completion["sha256"].items():
        if sha256_file(directory / name) != digest:
            raise ValueError("precision artifact checksum mismatch: " + name)
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    frames = report["measured_frames_per_block"]
    shapes = {"present": (frames,), "validity": (frames, 17), "confidence": (frames, 17),
              "detected": (frames,), "body": (frames, 17, 3), "root": (frames, 3)}
    if (type(frames) is not int or not 1 <= frames <= 600
            or [x["precision"] for x in report["runs"]]
            != ["autocast", "fp16_conv", "fp16_conv", "autocast"]):
        raise ValueError("invalid precision block layout")
    identity = True
    with np.load(directory / "poses.npz", allow_pickle=False) as poses:
        if set(poses.files) != {f"block{i}_{key}" for i in range(4) for key in shapes}:
            raise ValueError("unexpected precision pose arrays")
        for index in range(4):
            for key, shape in shapes.items():
                array = poses[f"block{index}_{key}"]
                expected_dtype = (np.bool_ if key in ("present", "validity", "detected")
                                  else np.float32)
                if array.shape != shape or array.dtype != expected_dtype:
                    raise ValueError("invalid precision array shape/dtype")
        for index in range(1, 4):
            equal = all(np.array_equal(poses["block0_" + key], poses[f"block{index}_{key}"],
                                       equal_nan=True)
                        for key in shapes)
            comparison = report["comparisons_to_first_baseline"][str(index)]
            claimed_equal = (comparison["arrays_exact"]
                             and comparison["validity_and_presence_match"]
                             and comparison["detection_schedule_match"])
            if equal != claimed_equal:
                raise ValueError("saved numerical identity does not reproduce")
            identity &= equal
        all_valid = all(
            poses[f"block{i}_present"].all() and poses[f"block{i}_validity"].all()
            and (poses[f"block{i}_confidence"] > 0).all() for i in range(4)
        )
    speed = {}
    p95 = {}
    for precision in ("autocast", "fp16_conv"):
        runs = [r["metrics"] for r in report["runs"] if r["precision"] == precision]
        speed[precision] = sum(r["valid_pose_frames"] for r in runs) / sum(
            r["duration_s"] for r in runs)
        p95[precision] = [r["all_results_latency_ms"]["p95"] for r in runs]
    return {"status": "verified", "numeric_identity_all_blocks": bool(identity),
            "all_frames_valid": bool(all_valid), "frames_per_block": frames,
            "aggregate_valid_fps": speed, "per_block_p95_ms": p95,
            "throughput_growth_percent": ((speed["fp16_conv"] / speed["autocast"] - 1) * 100
                                          if speed["autocast"] > 0 else None),
            "acceptance_status": "not_assessed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path, help="verify a completed probe without GPU/camera")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()
    if args.verify is not None:
        if args.input is not None or args.output is not None:
            parser.error("--verify cannot be combined with input/output")
        print(json.dumps(verify_probe(args.verify), indent=2, allow_nan=False))
        return
    if args.input is None or args.output is None:
        parser.error("input and --output are required for a new probe")
    config = load_live_config(args.config) if args.config else LiveConfig()
    run_probe(args.input, args.output, config=config, frames=args.frames, warmup=args.warmup)


if __name__ == "__main__":
    main()
