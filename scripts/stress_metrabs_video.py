"""Long-running recorded-motion stability fallback when webcam hardware is unavailable."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2

from poseahead.config import LiveConfig, load_live_config
from poseahead.perception.metrabs import MetrabsBackend
from poseahead.runtime.benchmark import BenchmarkStats
from poseahead.runtime.memory import working_set_mib


def _working_set_mib() -> float:
    value = working_set_mib()
    if value is None:
        raise RuntimeError("host-memory sampling is unavailable on this host")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--duration", type=float, default=600.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--sample-period", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        not math.isfinite(args.duration)
        or args.duration <= 0
        or args.warmup < 0
        or not math.isfinite(args.sample_period)
        or args.sample_period <= 0
    ):
        parser.error("duration/sample-period must be positive and warmup non-negative")

    import torch

    config = load_live_config(args.config) if args.config else LiveConfig()
    backend = MetrabsBackend(
        confidence_threshold=config.confidence_threshold,
        num_aug=config.num_aug,
        internal_batch_size=config.internal_batch_size,
        detector_interval=config.detector_interval,
        max_input_side=config.max_input_side,
        default_fov_degrees=config.default_fov_degrees,
        camera_intrinsics=config.camera_intrinsics,
        crop_precision=config.crop_precision,
    )
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {args.input}")
    frame_id = 0
    try:
        for _ in range(args.warmup):
            success, image = capture.read()
            if not success:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, image = capture.read()
            if not success:
                raise RuntimeError("video delivered no frame during warmup")
            backend.infer(image, frame_id=frame_id, source_timestamp_ns=time.perf_counter_ns())
            frame_id += 1

        torch.cuda.reset_peak_memory_stats()
        stats = BenchmarkStats()
        samples: list[list[float]] = []
        loops = 0
        started = time.perf_counter()
        next_sample = 0.0
        while time.perf_counter() - started < args.duration:
            success, image = capture.read()
            if not success:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                loops += 1
                success, image = capture.read()
            if not success:
                raise RuntimeError("video failed after rewind")
            source_ns = time.perf_counter_ns()
            pose = backend.infer(image, frame_id=frame_id, source_timestamp_ns=source_ns)
            stats.observe(pose, (time.perf_counter_ns() - source_ns) / 1e6)
            frame_id += 1
            elapsed = time.perf_counter() - started
            if elapsed >= next_sample:
                samples.append(
                    [
                        elapsed,
                        _working_set_mib(),
                        torch.cuda.memory_allocated() / 1024**2,
                        torch.cuda.memory_reserved() / 1024**2,
                    ]
                )
                next_sample += args.sample_period
        elapsed = time.perf_counter() - started
    finally:
        capture.release()

    rss_growth = samples[-1][1] - samples[0][1]
    gpu_reserved_growth = samples[-1][3] - samples[0][3]
    long_enough = elapsed >= 590
    stable = rss_growth <= 128 and gpu_reserved_growth <= 32
    report = stats.report(elapsed)
    report.update(
        {
            "input": str(args.input),
            "input_kind": "looped_recorded_motion_not_webcam",
            "resolved_config": config.to_dict(),
            "warmup_frames": args.warmup,
            "completed_video_rewinds": loops,
            "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "memory_samples_seconds_rss_allocated_reserved_mib": samples,
            "rss_growth_mib": rss_growth,
            "gpu_reserved_growth_mib": gpu_reserved_growth,
            "stability_thresholds": {
                "minimum_duration_s": 590,
                "maximum_rss_growth_mib": 128,
                "maximum_gpu_reserved_growth_mib": 32,
            },
            "stability_status": (
                "pass_recorded_fallback"
                if long_enough and stable
                else "fail_thresholds" if long_enough else "not_assessed_short_run"
            ),
        }
    )
    rendered = json.dumps(report, indent=2, allow_nan=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
