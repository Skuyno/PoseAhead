"""One fixed 100-ms training-noise experiment, clean validation selection."""

import argparse
import json
from pathlib import Path

from poseahead.datasets.normalization import fit_normalizer
from poseahead.datasets.windows import export_training_windows
from poseahead.prediction.training import train_predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="expanded CMU run")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = {"horizon_ms": 100, "training_noise_std_m": .01,
                "augmentation_seed": 2026, "model_seed": 42, "epochs": 8,
                "selection": "unchanged clean validation only",
                "targets": "clean source future poses", "test_used": False}
    with (output / "protocol.json").open("x", encoding="utf-8") as stream:
        json.dump(protocol, stream, indent=2)
    windows = export_training_windows(
        source / "index.json", output / "train", split="train", horizon_ms=100,
        history_sampling="linear_past", training_noise_std_m=.01, training_noise_seed=2026)
    print(json.dumps(windows), flush=True)
    fit_normalizer(output / "train", output / "normalizer.json")
    config = {"version": 1, "train_windows": "train",
              "validation_windows": str(source / "h100" / "validation"),
              "normalizer": "normalizer.json", "seed": 42, "epochs": 8,
              "batch_size": 64, "hidden_size": 64, "learning_rate": .001,
              "input_features": "relative_last", "batch_order": "global_windows"}
    path = output / "training.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)
    import torch

    torch.set_num_threads(2)
    report = train_predictor(path, output / "model")
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
