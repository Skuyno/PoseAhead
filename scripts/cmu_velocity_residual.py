"""Fixed clean/noisy 100-ms pilot: learn a correction to constant velocity."""

import argparse
import json
from pathlib import Path

import numpy as np
from cmu_seed_stability import evaluate

from poseahead.artifacts import sha256_file
from poseahead.datasets.window_store import WindowStore
from poseahead.prediction.training import train_predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="expanded CMU run")
    parser.add_argument("noise_source", type=Path, help="completed noisy training run")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source, noise_source = args.source.resolve(), args.noise_source.resolve()
    validation = source / "h100" / "validation"
    variants = [("clean", source / "h100" / "train", source / "h100" / "normalizer.json"),
                ("noisy", noise_source / "train", noise_source / "normalizer.json")]
    pins = {str(validation / "manifest.json"): sha256_file(validation / "manifest.json")}
    for _, train, normalizer in variants:
        store = WindowStore(train)
        if store.manifest["split"] != "train" or store.manifest["horizon_ms"] != 100:
            raise ValueError("protocol requires 100-ms training windows")
        pins[str(train / "manifest.json")] = sha256_file(train / "manifest.json")
        pins[str(normalizer)] = sha256_file(normalizer)
    if WindowStore(validation).manifest["split"] != "validation":
        raise ValueError("protocol requires validation, never test")
    clean_store, noisy_store = (WindowStore(v[1]) for v in variants)
    # Noise must change only observation histories, not the clean supervision.
    for field in ("sequences", "dataset_index_sha256", "history_frames", "step_ns",
                  "horizon_ms", "history_sampling", "max_history_gap_ns", "windows"):
        if clean_store.manifest.get(field) != noisy_store.manifest.get(field):
            raise ValueError(f"clean/noisy export contract mismatch: {field}")
    if (len(clean_store.manifest["shards"]) != len(noisy_store.manifest["shards"])
            or noisy_store.manifest.get("training_noise", {}).get("std_m") != .01):
        raise ValueError("requires matching clean and sigma-10mm augmented exports")
    for clean, noisy in zip(clean_store.iter_shards(), noisy_store.iter_shards(), strict=True):
        for key in ("history_timestamps_ns", "target_positions_m", "target_timestamp_ns",
                    "anchor_frame_id"):
            if not np.array_equal(clean[key], noisy[key]):
                raise ValueError(f"clean/noisy window mismatch: {key}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = {"horizon_ms": 100, "model_seed": 42, "epochs": 8,
                "output_baseline": "constant_velocity", "input_features": "relative_last",
                "batch_order": "global_windows", "variants": [v[0] for v in variants],
                "source_sha256": pins, "test_used": False,
                "selection": "clean validation; epoch-zero constant velocity eligible",
                "scope": "single seed; repeatedly used validation ID; no live promotion"}
    with (output / "protocol.json").open("x", encoding="utf-8") as stream:
        json.dump(protocol, stream, indent=2)
    import torch

    torch.set_num_threads(2)
    rows = []
    for name, train, normalizer in variants:
        config = {"version": 1, "train_windows": str(train),
                  "validation_windows": str(validation), "normalizer": str(normalizer),
                  "seed": 42, "epochs": 8, "batch_size": 64, "hidden_size": 64,
                  "learning_rate": .001, "input_features": "relative_last",
                  "output_baseline": "constant_velocity", "batch_order": "global_windows"}
        path = output / f"{name}.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(config, stream, indent=2)
        report = train_predictor(path, output / name)
        metrics = evaluate(source / "h100", output / name)
        if not np.isclose(metrics["model_mm"], report["validation_mpjpe_mm"], atol=1e-4):
            raise ValueError("reloaded checkpoint metric mismatch")
        rows.append({"variant": name, **report, **metrics})
        print(json.dumps(rows[-1]), flush=True)
    if any(sha256_file(Path(path)) != digest for path, digest in pins.items()):
        raise ValueError("source changed during experiment")
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump({"protocol": protocol, "results": rows}, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
