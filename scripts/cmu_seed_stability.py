"""Predeclared four-horizon, three-seed validation study of relative-last MLP."""

import argparse
import json
from pathlib import Path

import numpy as np

from poseahead.artifacts import sha256_file
from poseahead.datasets.window_store import WindowStore
from poseahead.prediction.checkpoint import FixedHorizonPredictor
from poseahead.prediction.stability import HORIZONS, SEEDS, summarize_stability
from poseahead.prediction.training import train_predictor


def evaluate(source, checkpoint):
    import torch

    predictor = FixedHorizonPredictor(checkpoint)
    store = WindowStore(source / "validation")
    if store.manifest["split"] != "validation":
        raise ValueError("validation only")
    errors = dict.fromkeys(("model_mm", "hold_mm", "velocity_mm"), 0.)
    count = 0
    for arrays in store.iter_shards():
        x, target = arrays["history_positions_m"], arrays["target_positions_m"]
        inputs = predictor.normalizer.transform(x).astype(np.float32)
        with torch.no_grad():
            normalized = predictor.model(torch.from_numpy(inputs)).numpy()
        prediction = predictor.normalizer.transform(normalized, inverse=True)
        hold = x[:, -1]
        velocity = hold + (x[:, -1] - x[:, -2]) * (
            predictor.horizon_ms * 1_000_000 / predictor.step_ns)
        for name, values in (("model_mm", prediction), ("hold_mm", hold),
                             ("velocity_mm", velocity)):
            errors[name] += float(np.linalg.norm(values.astype(float) - target, axis=-1).sum())
        count += len(x) * 17
    if count == 0:
        raise ValueError("empty validation")
    return {k: v * 1000 / count for k, v in errors.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    pins = {}
    for h in HORIZONS:
        for split in ("train", "validation"):
            store = WindowStore(source / f"h{h}" / split)
            if store.manifest["split"] != split or store.manifest["horizon_ms"] != h:
                raise ValueError("source protocol mismatch")
        for relative in (f"h{h}/train/manifest.json", f"h{h}/validation/manifest.json",
                         f"h{h}/normalizer.json"):
            pins[relative] = sha256_file(source / relative)
    output.mkdir(parents=True, exist_ok=False)
    protocol = {"horizons": HORIZONS, "seeds": SEEDS, "epochs": 8,
                "input_features": "relative_last", "batch_order": "global_windows",
                "source": str(source), "source_sha256": pins, "test_used": False,
                "interpretation": "seed dispersion only; not participant confidence intervals"}
    with (output / "protocol.json").open("x", encoding="utf-8") as stream:
        json.dump(protocol, stream, indent=2)
    import torch

    torch.set_num_threads(2)
    rows = []
    for h in HORIZONS:
        for seed in SEEDS:
            name = f"h{h}_seed{seed}"
            directory = source / f"h{h}"
            config = {"version": 1, "train_windows": str(directory / "train"),
                      "validation_windows": str(directory / "validation"),
                      "normalizer": str(directory / "normalizer.json"),
                      "seed": seed, "epochs": 8, "batch_size": 64, "hidden_size": 64,
                      "learning_rate": .001, "input_features": "relative_last",
                      "batch_order": "global_windows"}
            path = output / f"{name}.json"
            with path.open("x", encoding="utf-8") as stream:
                json.dump(config, stream, indent=2)
            report = train_predictor(path, output / name)
            metrics = evaluate(directory, output / name)
            if not np.isclose(metrics["model_mm"], report["validation_mpjpe_mm"], atol=1e-4):
                raise ValueError("reloaded checkpoint metric mismatch")
            row = {**report, **metrics, "seed": seed}
            rows.append(row)
            with (output / f"{name}_result.json").open("x", encoding="utf-8") as stream:
                json.dump(row, stream, indent=2, allow_nan=False)
            print(json.dumps(row), flush=True)
    if any(sha256_file(source / p) != digest for p, digest in pins.items()):
        raise ValueError("source changed during stability experiment")
    result = {"protocol": protocol, "runs": rows, "summary": summarize_stability(rows)}
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
