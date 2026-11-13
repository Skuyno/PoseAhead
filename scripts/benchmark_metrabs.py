"""Repeated-still MeTRAbs throughput probe, not a motion acceptance test."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from poseahead.config import LiveConfig, load_live_config
from poseahead.perception.metrabs import MetrabsBackend
from poseahead.runtime.benchmark import BenchmarkStats, latency_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        parser.error("warmup must be non-negative and iterations must be positive")
    config = load_live_config(args.config) if args.config else LiveConfig()

    import torch

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
    stats = BenchmarkStats()
    inference_times: list[float] = []
    started = 0.0
    last_result = None
    for frame_id in range(args.warmup + args.iterations):
        if frame_id == args.warmup:
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
        source_ns = time.perf_counter_ns()
        result = backend.infer(args.input, frame_id=frame_id, source_timestamp_ns=source_ns)
        completed_ns = time.perf_counter_ns()
        if frame_id < args.warmup:
            continue
        stats.observe(result, (completed_ns - source_ns) / 1e6)
        if backend.last_timing is not None:
            inference_times.append(backend.last_timing.total_ms)
        last_result = result

    report = stats.report(time.perf_counter() - started)
    report.update(
        {
            "input": str(args.input),
            "input_kind": "repeated_still_image",
            "warmup_frames": args.warmup,
            "resolved_config": config.to_dict(),
            "inference_ms": latency_summary(inference_times),
            "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "latency_scope": "file_load_through_canonical_frame_return_no_udp",
        }
    )
    if last_result is not None:
        report["sample"] = {
            "root_position_m": last_result.root_position.tolist(),
            "valid_joint_count": int(last_result.joint_validity.sum()),
            "scale_status": last_result.metadata["scale_status"],
        }
    rendered = json.dumps(report, indent=2, allow_nan=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
