"""Recorded-video throughput probe. Event-time replay is a separate milestone."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from poseahead.config import LiveConfig, load_live_config
from poseahead.core.serialization import encode_pose_frame
from poseahead.perception.metrabs import MetrabsBackend
from poseahead.runtime.benchmark import BenchmarkStats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--input-kind", choices=("unverified", "motion", "synthetic-still"), default="unverified"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sample-dir", type=Path)
    parser.add_argument("--sample-frames", type=int, default=60)
    args = parser.parse_args()
    if (
        args.warmup < 0
        or args.sample_frames < 1
        or (args.max_frames is not None and args.max_frames <= args.warmup)
    ):
        parser.error("max-frames must exceed non-negative warmup")
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
    stats = BenchmarkStats()
    frame_id = 0
    started = None
    sample_count = 0
    if args.sample_dir is not None:
        args.sample_dir.mkdir(parents=True, exist_ok=True)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"could not open video: {args.input}")
        while args.max_frames is None or frame_id < args.max_frames:
            if frame_id == args.warmup:
                started = time.perf_counter()
            success, image = capture.read()
            if not success:
                break
            source_ns = time.perf_counter_ns()
            result = backend.infer(image, frame_id=frame_id, source_timestamp_ns=source_ns)
            completed_ns = time.perf_counter_ns()
            if frame_id >= args.warmup:
                stats.observe(result, (completed_ns - source_ns) / 1e6)
                if result is not None and args.sample_dir is not None:
                    if sample_count < args.sample_frames:
                        sample_path = args.sample_dir / f"{sample_count:04d}.pafr"
                        sample_path.write_bytes(encode_pose_frame(result))
                        sample_count += 1
            frame_id += 1
    finally:
        capture.release()
    if started is None or not stats.latency_ms:
        raise RuntimeError("video contained no measured frames after warmup")
    report = stats.report(time.perf_counter() - started)
    report.update(
        {
            "input": str(args.input),
            "input_kind": args.input_kind,
            "resolved_config": config.to_dict(),
            "warmup_frames": args.warmup,
            "latency_scope": "decoded_frame_available_to_canonical_pose_no_udp",
            "clock_semantics": "processing_monotonic_clock_not_recorded_event_time",
            "saved_pose_sample_frames": sample_count,
        }
    )
    if args.sample_dir is not None:
        manifest = {
            "format": "one PoseAhead PAFR v1 file per frame",
            "source": str(args.input),
            "input_kind": args.input_kind,
            "frames": sample_count,
        }
        (args.sample_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    rendered = json.dumps(report, indent=2, allow_nan=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
