"""Matched target-error diagnostics on validation only, not latency/jitter claims."""

import argparse
import json
from pathlib import Path

import numpy as np

from poseahead.artifacts import sha256_file
from poseahead.datasets.index import verify_dataset
from poseahead.datasets.window_comparison import verify_same_validation_windows
from poseahead.datasets.window_store import WindowStore
from poseahead.prediction.checkpoint import FixedHorizonPredictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--reference-run", type=Path,
                        help="require numerically identical validation windows to this run")
    args = parser.parse_args()
    verify_dataset(args.directory / "index.json")
    reports = []
    for horizon in (33, 66, 100, 150):
        run = args.directory / f"h{horizon}"
        store = WindowStore(run / "validation")
        if store.manifest["split"] != "validation":
            raise ValueError("diagnostics require validation only")
        matched = None
        if args.reference_run:
            previous = WindowStore(args.reference_run / f"h{horizon}" / "validation")
            matched = verify_same_validation_windows(previous, store)
        predictor = FixedHorizonPredictor(run / "model")
        sums = {"hold": 0., "constant_velocity_last_two": 0., "mlp": 0.}
        count = 0
        per_sequence = {}
        for shard, arrays in zip(store.manifest["shards"], store.iter_shards(), strict=True):
            sequence_index = shard["sequence_index"]
            sequence = per_sequence.setdefault(sequence_index, {
                "sums": dict.fromkeys(sums, 0.), "count": 0})
            x, y = arrays["history_positions_m"], arrays["target_positions_m"]
            predictions = {
                "hold": x[:, -1],
                "constant_velocity_last_two": x[:, -1] + (x[:, -1] - x[:, -2])
                * (horizon * 1_000_000 / predictor.step_ns),
                "mlp": np.stack([predictor.predict(p, tuple(int(t) for t in times))
                                 for p, times in zip(x, arrays["history_timestamps_ns"],
                                                     strict=True)]),
            }
            for method, values in predictions.items():
                error = float(np.linalg.norm(values.astype(np.float64) - y,
                                              axis=-1).sum()) * 1000
                sums[method] += error
                sequence["sums"][method] += error
            sequence["count"] += len(y) * 17
            count += len(y) * 17
        if not count:
            raise ValueError("empty validation windows")
        report = {"horizon_ms": horizon, "windows": count // 17,
                  "canonical17_mpjpe_mm_including_pelvis": {k: v / count for k, v in sums.items()},
                  "validation_manifest_sha256": store.manifest_sha256,
                  "checkpoint_manifest_sha256": sha256_file(run / "model" / "manifest.json")}
        report["matched_reference"] = matched
        report["per_sequence"] = [{
            "sequence_id": store.manifest["sequences"][i]["sequence_id"],
            "windows": row["count"] // 17,
            "mpjpe_mm": {k: v / row["count"] for k, v in row["sums"].items()},
        } for i, row in per_sequence.items()]
        report["sequence_macro_mpjpe_mm"] = {
            k: float(np.mean([r["mpjpe_mm"][k] for r in report["per_sequence"]])) for k in sums}
        reports.append(report)
    result = {"purpose": "validation_target_error_diagnostic_not_final_method_selection",
              "test_used": False, "latency_and_jitter_evaluated": False,
              "reports": reports}
    with (args.directory / "validation_comparison.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
