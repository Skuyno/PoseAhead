"""Analytic smoke-test fixtures, not a biomechanically realistic motion dataset."""

import json
from pathlib import Path

import numpy as np

from poseahead.core import PoseFrame, PoseSequence
from poseahead.recording import save_recording


def create_synthetic_experiment(directory):
    """Create new known-truth inputs and five configs; never reuse existing paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    frames = []
    for i in range(400):
        t = i / 100
        value = (np.exp(-((t - 1.2) / .13) ** 2)
                 - .7 * np.exp(-((t - 2.1) / .21) ** 2)
                 + .4 * np.exp(-((t - 3.2) / .1) ** 2))
        positions = np.zeros((17, 3))
        positions[1:, 0] = value
        frames.append(PoseFrame(
            i, i * 10_000_000, i * 10_000_000, positions, np.ones(17), np.ones(17, bool),
            root_position=[value, 0, 3], metadata={"synthetic": "analytic_pulses_v1",
                                                "root_translation_status": "ground_truth"},
        ))
    save_recording(directory / "truth.npz", PoseSequence(
        "synthetic-analytic-pulses-v1", "synthetic:analytic-pulses-v1", tuple(frames),
    ))
    methods = ("raw", "ema", "one_euro", "kalman", "kalman_cv")
    for method in methods:
        config = {
            "version": 1, "run_id": f"synthetic-{method}-delay100",
            "observations": "truth.npz", "reference": "truth.npz",
            "processor": {"method": method, "parameters": {}}, "seed": 42,
            "delay_ms": 100, "start_ns": 200_000_000, "end_ns": 3_990_000_000,
            "step_ns": 10_000_000, "window_frames": 380,
        }
        with (directory / f"{method}.json").open("x", encoding="utf-8") as stream:
            json.dump(config, stream, indent=2, allow_nan=False)
    with (directory / "batch.json").open("x", encoding="utf-8") as stream:
        json.dump({"version": 1, "batch_id": "synthetic-baselines-delay100",
                   "split": "unlabelled", "experiments": [f"{m}.json" for m in methods]},
                  stream, indent=2, allow_nan=False)
    return {"directory": str(directory.resolve()), "frames": 400, "configs": 5,
            "purpose": "synthetic_metric_sanity_not_real_motion_accuracy"}
