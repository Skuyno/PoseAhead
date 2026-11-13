"""Profile the detector and crop model inside the pinned MeTRAbs backend."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from poseahead.artifacts import sha256_file
from poseahead.config import LiveConfig, load_live_config
from poseahead.perception.metrabs import MetrabsBackend


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"samples": 0, "median": None, "p95": None}
    return {
        "samples": len(values),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
    }


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
    image = cv2.imread(str(args.input))
    if image is None:
        parser.error(f"could not read image: {args.input}")

    config = load_live_config(args.config) if args.config else LiveConfig()
    backend = MetrabsBackend(
        confidence_threshold=config.confidence_threshold, num_aug=config.num_aug,
        internal_batch_size=config.internal_batch_size, detector_interval=config.detector_interval,
        max_input_side=config.max_input_side, default_fov_degrees=config.default_fov_degrees,
        camera_intrinsics=config.camera_intrinsics, crop_precision=config.crop_precision,
    )
    samples: dict[str, list[float]] = defaultdict(list)
    starts: dict[str, float] = {}
    current_stage_ms: dict[str, float] = defaultdict(float)
    measured = False

    def pre(name: str):
        def hook(_module, _inputs):
            torch.cuda.synchronize()
            starts[name] = time.perf_counter()

        return hook

    def post(name: str):
        def hook(_module, _inputs, _output):
            torch.cuda.synchronize()
            if measured:
                elapsed_ms = (time.perf_counter() - starts[name]) * 1000
                samples[name].append(elapsed_ms)
                current_stage_ms[name] += elapsed_ms

        return hook

    handles = [
        backend._model.detector.register_forward_pre_hook(pre("detector")),
        backend._model.detector.register_forward_hook(post("detector")),
        backend._model.crop_model.register_forward_pre_hook(pre("crop_model")),
        backend._model.crop_model.register_forward_hook(post("crop_model")),
    ]
    total: list[float] = []
    overhead: list[float] = []
    try:
        for frame_id in range(args.warmup + args.iterations):
            measured = frame_id >= args.warmup
            current_stage_ms.clear()
            started = time.perf_counter()
            result = backend.infer(
                image,
                frame_id=frame_id,
                source_timestamp_ns=time.perf_counter_ns(),
            )
            elapsed = (time.perf_counter() - started) * 1000
            if result is None:
                raise RuntimeError("person was not detected in the profiling image")
            if measured:
                total.append(elapsed)
                overhead.append(max(0.0, elapsed - sum(current_stage_ms.values())))
    finally:
        for handle in handles:
            handle.remove()

    report = {
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "resolved_config": config.to_dict(),
        "scope": "repeated still with synchronizing stage hooks; "
                 "profiling overhead included, not live timing acceptance",
        "warmup": args.warmup,
        "iterations": args.iterations,
        "total_ms": _summary(total),
        "detector_ms": _summary(samples["detector"]),
        "crop_model_ms": _summary(samples["crop_model"]),
        "other_work_ms": _summary(overhead),
    }
    rendered = json.dumps(report, indent=2, allow_nan=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
