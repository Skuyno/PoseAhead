"""Verified descriptive diagnostics; common latency windows are NOT a superiority claim."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from poseahead.artifacts import sha256_file
from poseahead.evaluation.selection import select_from_batches

_LATENCY_OPTIONS = ("sample_interval_ms", "max_lag_ms", "searched_max_lag_ms",
                    "min_velocity_variation_m_s", "min_correlation", "ambiguity_margin")


def paired_lag_diagnostic(baseline, candidate):
    """Intersect actual eligible (time-window, joint) identities; never pair different masks.

    This posthoc subset changes the estimand and may be selection-biased. It cannot
    override the frozen selector's full-mask requirement or establish end-to-end delay.
    A cell without any paired eligible window has no lag comparison, not a zero lag.
    """
    left, right = baseline["latency"]["windows"], candidate["latency"]["windows"]
    if len(left) != len(right):
        raise ValueError("latency window grids differ")
    a, b = [], []
    counts = {"common": 0, "baseline_only": 0, "candidate_only": 0, "neither": 0}
    for x, y in zip(left, right, strict=True):
        keys = ("start_ns", "end_ns", *_LATENCY_OPTIONS)
        if any(x[k] != y[k] for k in keys):
            raise ValueError("latency window identity/settings differ")
        indexed = []
        for window in (x, y):
            joints = {j["joint"]: j for j in window["joint_windows"]}
            if len(joints) != len(window["joint_windows"]):
                raise ValueError("duplicate latency joint identity")
            indexed.append(joints)
        if indexed[0].keys() != indexed[1].keys():
            raise ValueError("latency joint grids differ")
        for joint in indexed[0]:
            values = [index[joint] for index in indexed]
            ok = [v["status"] == "ok" for v in values]
            for value, usable in zip(values, ok, strict=True):
                if usable and (type(value["lag_ms"]) not in (int, float)
                               or not math.isfinite(value["lag_ms"])):
                    raise ValueError("eligible latency must have a finite numeric lag")
            if all(ok):
                counts["common"] += 1
                a.append(values[0]["lag_ms"])
                b.append(values[1]["lag_ms"])
            elif ok[0]:
                counts["baseline_only"] += 1
            elif ok[1]:
                counts["candidate_only"] += 1
            else:
                counts["neither"] += 1
    return {
        "status": "common_subset_descriptive_only" if a else "no_common_eligible_windows",
        "joint_window_counts": counts, "total_joint_windows": sum(counts.values()),
        "baseline_common_median_ms": float(np.median(a)) if a else None,
        "candidate_common_median_ms": float(np.median(b)) if b else None,
        "delta_of_common_medians_ms": float(np.median(b) - np.median(a)) if a else None,
        "median_paired_window_delta_ms": float(np.median(np.array(b) - a)) if a else None,
    }


def _mean_complete(values):
    return float(np.mean(values)) if values and all(v is not None for v in values) else None


def _difference(candidate, baseline):
    return candidate - baseline if candidate is not None and baseline is not None else None


def _improved(value):
    return value is not None and value < 0


def analyze_confirmation(directory, candidate_id="frozen_adaptive"):
    directory = Path(directory)
    result = select_from_batches(directory / "selection_protocol.json",
                                 directory / "selection_manifest.json")
    if result != json.loads((directory / "selection.json").read_text(encoding="utf-8")):
        raise ValueError("saved selection report does not reproduce from pinned artifacts")
    manifest = json.loads((directory / "selection_manifest.json").read_text(encoding="utf-8"))
    protocol = json.loads((directory / "selection_protocol.json").read_text(encoding="utf-8"))
    cells = {}
    for batch in manifest["batches"]:
        folder = directory / batch["directory"]
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        for row in summary["runs"]:
            labels = batch["cells"][row["run_id"]]
            child = folder / row["directory"]
            config = json.loads((child / "resolved_config.json").read_text(encoding="utf-8"))
            metrics = json.loads((child / "metrics.json").read_text(encoding="utf-8"))
            key = labels["candidate_id"], labels["profile_id"], config["seed"], row["sequence_id"]
            cells[key] = metrics
    choices = [c for c in result["candidates"] if c["candidate_id"] == candidate_id]
    if len(choices) != 1 or candidate_id == "kalman_cv":
        raise ValueError("diagnostic candidate must identify one non-baseline configuration")
    adaptive = choices[0]
    profiles = {}
    for profile in protocol["profile_ids"]:
        pairs = []
        for seed in protocol["seeds"]:
            for sequence in protocol["sequences"]:
                base = cells["kalman_cv", profile, seed, sequence]
                candidate = cells[candidate_id, profile, seed, sequence]
                pairs.append({
                    "seed": seed, "sequence_id": sequence,
                    "display_error_delta_mm": _difference(candidate["display_error"]["mpjpe_mm"],
                                                          base["display_error"]["mpjpe_mm"]),
                    "jitter_delta_m_s2": _difference(candidate["jitter"]["acceleration_error_m_s2"],
                                                     base["jitter"]["acceleration_error_m_s2"]),
                    "lag": paired_lag_diagnostic(base, candidate),
                    "baseline_runtime": base["runtime"], "candidate_runtime": candidate["runtime"],
                    "adaptive": candidate["adaptive_prediction"],
                    "output_counts": candidate["learned_prediction"]["observation_output_counts"],
                })
        complete_delta = _mean_complete([p["lag"]["delta_of_common_medians_ms"] for p in pairs])
        profiles[profile] = {
            **adaptive["profiles"][profile], "paired_cells": pairs,
            "improved_error_cells": sum(_improved(p["display_error_delta_mm"]) for p in pairs),
            "improved_jitter_cells": sum(_improved(p["jitter_delta_m_s2"]) for p in pairs),
            "improved_both_cells": sum(_improved(p["display_error_delta_mm"]) and
                                       _improved(p["jitter_delta_m_s2"]) for p in pairs),
            "common_subset_latency_delta_macro_ms": complete_delta,
            "cells_with_common_latency": sum(
                p["lag"]["delta_of_common_medians_ms"] is not None for p in pairs),
            "seed_macro_changes": {
                str(seed): {field: _mean_complete([p[field] for p in pairs if p["seed"] == seed])
                            for field in ("display_error_delta_mm", "jitter_delta_m_s2")}
                for seed in protocol["seeds"]
            },
            "candidate_process_p95_macro_ms": _mean_complete([
                p["candidate_runtime"]["processor_p95_ms"] for p in pairs]),
            "candidate_process_p95_max_cell_ms": max(
                p["candidate_runtime"]["processor_p95_ms"] for p in pairs),
        }
    return {
        "protocol_sha256": result["protocol_sha256"],
        "declaration_sha256": result["declaration_sha256"],
        "selection_sha256": sha256_file(directory / "selection.json"),
        "candidate_id": candidate_id,
        "frozen_candidate": result["candidate_processors"][candidate_id],
        "adaptive_eligible": adaptive["eligible"], "adaptive_reasons": adaptive["exclusions"],
        "adaptive_robust_score": adaptive["robust_score"],
        "selected_candidate_id": result["selected_candidate_id"], "profiles": profiles,
        "test_scored": False, "advantage_claim": "not_assessed",
        "aggregation": "equal_clip_and_noise_seed; seeds are not independent subjects",
        "latency_scope": "posthoc paired-common-window subset; possible selection bias",
        "latency_limitation": "does not override full-mask gates or prove 20-ms/CI superiority",
        "runtime_scope": "CPU process call; not injected simulation delay or capture-to-screen",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("output", type=Path, help="new report; never overwritten")
    parser.add_argument("--candidate", default="frozen_adaptive")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("diagnostic output already exists")
    result = analyze_confirmation(args.study, args.candidate)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"adaptive_eligible": result["adaptive_eligible"],
                      "adaptive_robust_score": result["adaptive_robust_score"],
                      "advantage_claim": result["advantage_claim"]}))


if __name__ == "__main__":
    main()
