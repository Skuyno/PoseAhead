"""Paired sequence bootstrap and frozen-comparator research decision reporting."""

import json
from pathlib import Path

import numpy as np

from poseahead.artifacts import sha256_file
from poseahead.evaluation.batch import verify_batch

_METRICS = ("latency_ms", "display_mpjpe_mm", "jitter_m_s2")
_COVERAGE = ("joint_coverage", "jitter_coverage", "latency_coverage")
_RULE = {"latency_reduction_ms": 20., "maximum_relative_growth": .10,
         "confidence": .95, "bootstrap_unit": "sequence_clock_pair"}


def _read_batch(directory):
    directory = Path(directory)
    verify_batch(directory)
    return json.loads((directory / "summary.json").read_text(encoding="utf-8"))


def _baseline(summary, run_id):
    rows = [r for r in summary["runs"] if r["run_id"] == run_id]
    if len(rows) != 1:
        raise ValueError("baseline run ID must identify exactly one row")
    return rows[0]["condition"]


def freeze_comparator(batch_directory, baseline_run_id, output):
    """Record a user-chosen comparator from validation; never choose one on test."""
    summary = _read_batch(batch_directory)
    if summary["split"] != "validation":
        raise ValueError("comparator can only be frozen from a validation batch")
    report = {"format": "poseahead-fixed-comparator", "version": 1,
              "selection_split": "validation", "selection_batch_id": summary["batch_id"],
              "selection_batch_directory": str(Path(batch_directory).resolve()),
              "selection_batch_sha256": sha256_file(
                  Path(batch_directory) / "batch_completion.json"),
              "baseline_run_id": baseline_run_id, "condition": _baseline(summary, baseline_run_id),
              "rule": dict(_RULE)}
    with Path(output).open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


def _bootstrap_options(seed, repetitions):
    if type(seed) is not int or seed < 0:
        raise ValueError("bootstrap seed must be a nonnegative integer")
    if type(repetitions) is not int or not 100 <= repetitions <= 100_000:
        raise ValueError("bootstrap repetitions must be in [100, 100000]")


def paired_bootstrap(baseline, candidate, *, seed=42, repetitions=2000):
    """Pointwise percentile CIs on paired mean candidate-minus-baseline changes.

    Each sequence has equal weight. This is not a simultaneous three-metric CI or
    a subject-cluster bootstrap. The same paired sample is used across metrics.
    """
    _bootstrap_options(seed, repetitions)
    a, b = np.asarray(baseline, dtype=np.float64), np.asarray(candidate, dtype=np.float64)
    if (a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3 or not len(a)
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("bootstrap requires finite matching (sequences, 3) metrics")
    delta = b - a
    intervals = None
    if len(a) >= 2:
        rng = np.random.default_rng(seed)
        # Chunk draws so a large repetition count does not allocate B*N at once.
        means = []
        chunk_size = max(1, min(128, 1_000_000 // len(a)))
        for start in range(0, repetitions, chunk_size):
            draws = rng.integers(0, len(a), size=(min(chunk_size, repetitions - start), len(a)))
            means.append(delta[draws].mean(axis=1))
        intervals = np.quantile(np.concatenate(means), [.025, .975], axis=0)
    result = {}
    for i, name in enumerate(_METRICS):
        result[name] = {"baseline_mean": float(a[:, i].mean()),
                        "candidate_mean": float(b[:, i].mean()),
                        "delta_mean": float(delta[:, i].mean()),
                        "delta_ci95": intervals[:, i].tolist() if intervals is not None else None}
    return result


def _rules(metrics, n):
    if n < 2:
        return {"rule1": None, "rule2": None, "reason": "fewer_than_two_eligible_sequences"}
    upper = [metrics[m]["delta_ci95"][1] for m in _METRICS]
    rule1 = all(v <= 0 for v in upper) and any(v < 0 for v in upper)
    allowed = []
    for name in _METRICS[1:]:
        baseline, candidate = metrics[name]["baseline_mean"], metrics[name]["candidate_mean"]
        # Zero comparator error has zero tolerance, rather than division by zero.
        allowed.append(candidate <= baseline * (1 + _RULE["maximum_relative_growth"]))
    rule2 = (metrics["latency_ms"]["delta_mean"] <= -_RULE["latency_reduction_ms"]
             and all(allowed))
    return {"rule1": rule1, "rule2": rule2,
            "reason": "pointwise_upper_ci_noninferiority_or_frozen_mean_tradeoff"}


def _paired_settings(directory, row):
    config = json.loads((Path(directory) / row["directory"] / "resolved_config.json")
                        .read_text(encoding="utf-8"))
    return {k: config[k] for k in ("observations_sha256", "reference_sha256", "seed",
                                   "start_ns", "end_ns", "step_ns")}


def _same_coverage(directory, left, right):
    folders = [Path(directory) / row["directory"] for row in (left, right)]
    with np.load(folders[0] / "display_samples.npz", allow_pickle=False) as a:
        with np.load(folders[1] / "display_samples.npz", allow_pickle=False) as b:
            if not np.array_equal(a["validity"], b["validity"]):
                return False
    masks = []
    for folder in folders:
        report = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        masks.append([[j["status"] == "ok" for j in w["joint_windows"]]
                      for w in report["latency"]["windows"]])
    return masks[0] == masks[1]


def pareto_front(points):
    """Indices of nondominated finite three-metric points; lower is better."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Pareto points must be finite (methods, 3) metrics")
    return [i for i, point in enumerate(points) if not any(
        np.all(other <= point) and np.any(other < point) for other in points
    )]


def compare_batch(directory, baseline_run_id, *, frozen_comparator=None, seed=42, repetitions=2000):
    _bootstrap_options(seed, repetitions)
    summary = _read_batch(directory)
    baseline_condition = _baseline(summary, baseline_run_id)
    frozen_hash = None
    if frozen_comparator is not None:
        frozen = json.loads(Path(frozen_comparator).read_text(encoding="utf-8"))
        fields = {"format", "version", "selection_split", "selection_batch_id",
                  "selection_batch_directory", "selection_batch_sha256", "baseline_run_id",
                  "condition", "rule"}
        if (not isinstance(frozen, dict) or set(frozen) != fields
                or frozen.get("format") != "poseahead-fixed-comparator"
                or type(frozen.get("version")) is not int or frozen["version"] != 1
                or frozen.get("selection_split") != "validation" or frozen.get("rule") != _RULE
                or frozen.get("condition") != baseline_condition):
            raise ValueError("frozen validation comparator/protocol does not match baseline")
        for field in ("selection_batch_directory", "selection_batch_sha256", "baseline_run_id"):
            if not isinstance(frozen[field], str) or not frozen[field].strip():
                raise ValueError("invalid frozen comparator provenance")
        selection = _read_batch(frozen["selection_batch_directory"])
        if (selection["split"] != "validation"
                or _baseline(selection, frozen["baseline_run_id"]) != baseline_condition
                or sha256_file(Path(frozen["selection_batch_directory"]) / "batch_completion.json")
                != frozen["selection_batch_sha256"]):
            raise ValueError("validation comparator provenance mismatch")
        frozen_hash = sha256_file(Path(frozen_comparator))
    if summary["split"] == "test" and frozen_hash is None:
        raise ValueError("test comparison requires a frozen validation comparator")
    groups = {}
    for row in summary["runs"]:
        key = json.dumps(row["condition"], sort_keys=True)
        groups.setdefault(key, []).append(row)
    baseline_rows = groups[json.dumps(baseline_condition, sort_keys=True)]
    base = {(r["sequence_id"], r["clock_domain"]): r for r in baseline_rows}
    operating_point = {k: v for k, v in baseline_condition.items() if k != "processor"}
    comparisons = []
    for rows in groups.values():
        condition = rows[0]["condition"]
        if condition == baseline_condition:
            continue
        if {k: v for k, v in condition.items() if k != "processor"} != operating_point:
            continue
        candidates = {(r["sequence_id"], r["clock_domain"]): r for r in rows}
        common = sorted(base.keys() & candidates.keys())
        a, b, excluded, pairs = [], [], [], []
        for identity in common:
            left, right = base[identity], candidates[identity]
            reason = None
            if _paired_settings(directory, left) != _paired_settings(directory, right):
                reason = "input_seed_or_evaluation_grid_mismatch"
            elif any(left.get(k) is None or right.get(k) is None for k in _METRICS):
                reason = "missing_metric"
            elif (any(left[k] != right[k] for k in _COVERAGE)
                  or not _same_coverage(directory, left, right)):
                reason = "different_metric_coverage"
            else:
                a.append([left[k] for k in _METRICS])
                b.append([right[k] for k in _METRICS])
                pairs.append({"sequence_id": identity[0], "clock_domain": identity[1],
                              "subject": left["subject"], "baseline_run_id": left["run_id"],
                              "candidate_run_id": right["run_id"]})
            if reason:
                excluded.append({"sequence_id": identity[0], "clock_domain": identity[1],
                                 "reason": reason})
        metrics = paired_bootstrap(a, b, seed=seed, repetitions=repetitions) if a else None
        rules = _rules(metrics, len(a))
        complete = len(a) == len(base) == len(candidates)
        comparisons.append({"candidate_condition": condition, "paired_sequences": pairs,
                            "excluded_pairs": excluded, "baseline_sequences": len(base),
                            "candidate_sequences": len(candidates), "eligible_pairs": len(a),
                            "complete_matched_panel": complete, "metrics": metrics, "rules": rules,
                            "eligible_for_frozen_rule_claim": bool(frozen_hash and complete
                                                                    and len(a) >= 2),
                            "passes_rule_on_eligible_pairs":
                            bool(rules["rule1"] or rules["rule2"])})
    panel = []
    for i, comparison in enumerate(comparisons):
        if comparison["complete_matched_panel"] and comparison["metrics"]:
            metrics = comparison["metrics"]
            if not panel:
                panel.append({"method": "baseline", "values": [metrics[m]["baseline_mean"]
                                                                for m in _METRICS]})
            panel.append({"method": f"comparison-{i}", "values": [metrics[m]["candidate_mean"]
                                                                    for m in _METRICS]})
    front = pareto_front([p["values"] for p in panel]) if panel else []
    return {"batch_id": summary["batch_id"], "split": summary["split"],
            "batch_sha256": sha256_file(Path(directory) / "batch_completion.json"),
            "baseline_condition": baseline_condition, "frozen_comparator_sha256": frozen_hash,
            "seed": seed, "bootstrap_repetitions": repetitions, "rule": dict(_RULE),
            "comparisons": comparisons,
            "pareto": {"scope": "descriptive_means_on_complete_mask_matched_panels",
                       "metric_order": _METRICS, "points": panel,
                       "nondominated": [panel[i]["method"] for i in front]},
            "limitations": ["Pointwise CIs, not simultaneous or multiplicity-adjusted.",
                            "Sequence bootstrap does not model within-subject/camera dependence.",
                            "Differing validity or latency eligibility masks are excluded.",
                            "No auto-winner or proof of validation-before-test chronology."]}
