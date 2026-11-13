"""Fixed 2x2 validation-only experiment: input representation versus batch order."""

import argparse
import json
from pathlib import Path

import numpy as np

from poseahead.artifacts import sha256_file
from poseahead.datasets.window_store import WindowStore
from poseahead.prediction.checkpoint import FixedHorizonPredictor
from poseahead.prediction.training import train_predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if WindowStore(source / "validation").manifest["horizon_ms"] != 100:
        raise ValueError("ablation expects 100-ms windows")
    output.mkdir(parents=True, exist_ok=False)
    variants = [(features, order) for features in ("positions", "relative_last")
                for order in ("within_shards", "global_windows")]
    protocol = {"horizon_ms": 100, "epochs": 8, "seed": 42, "variants": variants,
                "source": str(source), "test_used": False,
                "train_sha256": sha256_file(source / "train" / "manifest.json"),
                "validation_sha256": sha256_file(source / "validation" / "manifest.json"),
                "normalizer_sha256": sha256_file(source / "normalizer.json")}
    with (output / "protocol.json").open("x", encoding="utf-8") as stream:
        json.dump(protocol, stream, indent=2)
    import torch

    torch.set_num_threads(2)
    reports = []
    for features, order in variants:
        name = f"{features}_{order}"
        config = {"version": 1, "train_windows": str(source / "train"),
                  "validation_windows": str(source / "validation"),
                  "normalizer": str(source / "normalizer.json"),
                  "seed": 42, "epochs": 8, "batch_size": 64, "hidden_size": 64,
                  "learning_rate": .001, "input_features": features, "batch_order": order}
        path = output / f"{name}.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(config, stream, indent=2)
        report = train_predictor(path, output / name)
        predictor = FixedHorizonPredictor(output / name)
        validation = WindowStore(source / "validation")
        if validation.manifest["horizon_ms"] != 100:
            raise ValueError("ablation expects 100-ms windows")
        errors = {"hold": 0., "constant_velocity": 0., "model": 0.}
        count = 0
        for arrays in validation.iter_shards():
            x, y = arrays["history_positions_m"], arrays["target_positions_m"]
            predictions = {
                "hold": x[:, -1],
                "constant_velocity": x[:, -1] + (x[:, -1] - x[:, -2])
                * (100_000_000 / predictor.step_ns),
                "model": np.stack([predictor.predict(p, tuple(int(t) for t in ts))
                                   for p, ts in zip(x, arrays["history_timestamps_ns"],
                                                    strict=True)]),
            }
            for method, values in predictions.items():
                errors[method] += float(np.linalg.norm(values.astype(float) - y, axis=-1).sum())
            count += len(y) * 17
        report.update(variant=name, reloaded_mpjpe_mm={k: v * 1000 / count
                                                      for k, v in errors.items()})
        reports.append(report)
        print(json.dumps(report), flush=True)
    for key, path in (("train_sha256", source / "train" / "manifest.json"),
                      ("validation_sha256", source / "validation" / "manifest.json"),
                      ("normalizer_sha256", source / "normalizer.json")):
        if sha256_file(path) != protocol[key]:
            raise ValueError("ablation source changed")
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump({"protocol": protocol, "results": reports,
                   "limitations": "single seed; validation only; no latency/jitter/noise claims"},
                  stream, indent=2)


if __name__ == "__main__":
    main()
