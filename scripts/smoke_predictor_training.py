"""Retained synthetic end-to-end training artifact; not a Human3.6M experiment."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from poseahead.core import PoseSequence
from poseahead.datasets.human36m import camera_poses_to_sequence
from poseahead.datasets.index import freeze_dataset
from poseahead.datasets.normalization import fit_normalizer
from poseahead.datasets.window_store import WindowStore
from poseahead.datasets.windows import export_training_windows
from poseahead.prediction.checkpoint import FixedHorizonPredictor
from poseahead.prediction.training import train_predictor
from poseahead.recording import save_recording


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--horizon-ms", type=int, choices=(33, 66, 100, 150), default=33)
    args = parser.parse_args()
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = []
    for subject, split in (("S1", "train"), ("S8", "validation"), ("S9", "test")):
        points = np.zeros((35, 32, 3))
        points[:, 1:, 0] = np.arange(35)[:, None] * 20
        sequence = camera_poses_to_sequence(points, sequence_id=f"synthetic/{subject}",
                                            camera_id="synthetic_camera")
        frames = tuple(replace(f, metadata={**f.metadata, "subject": subject, "split": split,
                                             "synthetic_fixture": "linear_training_smoke_not_h36m"})
                       for f in sequence.frames)
        path = output / f"synthetic_{subject}.npz"
        save_recording(path, PoseSequence(sequence.sequence_id, sequence.clock_domain, frames))
        paths.append(path)
    index = output / "index.json"
    freeze_dataset(paths, index)
    for split in ("train", "validation", "test"):
        export_training_windows(index, output / split, split=split, horizon_ms=args.horizon_ms,
                                history_frames=3, shard_size=16)
        WindowStore(output / split).verify()
    fit_normalizer(output / "train", output / "normalizer.json")
    config = {"version": 1, "train_windows": "train", "validation_windows": "validation",
              "normalizer": "normalizer.json", "seed": 42, "epochs": 8,
              "batch_size": 16, "hidden_size": 8, "learning_rate": .01}
    config_path = output / "training.json"
    with config_path.open("x", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)
    report = train_predictor(config_path, output / "model")
    predictor = FixedHorizonPredictor(output / "model")
    first = next(WindowStore(output / "validation").iter_shards())
    prediction = predictor.predict(first["history_positions_m"][0],
                                   tuple(int(t) for t in first["history_timestamps_ns"][0]))
    report.update(purpose="synthetic_software_smoke_not_research_result",
                  prediction_finite=bool(np.isfinite(prediction).all()))
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
