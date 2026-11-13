import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest


@pytest.fixture
def analyzer():
    path = Path(__file__).resolve().parents[2] / "scripts/analyze_adaptive_confirmation.py"
    spec = importlib.util.spec_from_file_location("confirmation_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(lags):
    return {"latency": {"windows": [{
        "start_ns": 0, "end_ns": 975_000_000, "sample_interval_ms": 25., "max_lag_ms": 200.,
        "searched_max_lag_ms": 200., "min_velocity_variation_m_s": 1e-4,
        "min_correlation": .6, "ambiguity_margin": .02,
        "joint_windows": [{"joint": i, "lag_ms": lag, "status": "ok" if lag is not None
                           else "ambiguous_peak"} for i, lag in enumerate(lags)],
    }]}}


def test_pairs_only_the_same_joint_and_window_not_the_unmatched_medians(analyzer):
    result = analyzer.paired_lag_diagnostic(metrics([100, None, 1000, None]),
                                           metrics([80, -1000, None, None]))
    assert result["joint_window_counts"] == {
        "common": 1, "baseline_only": 1, "candidate_only": 1, "neither": 1,
    }
    assert result["baseline_common_median_ms"] == 100
    assert result["candidate_common_median_ms"] == 80
    assert result["delta_of_common_medians_ms"] == -20
    assert result["median_paired_window_delta_ms"] == -20
    assert result["total_joint_windows"] == 4
    assert result["status"] == "common_subset_descriptive_only"


def test_no_shared_eligible_window_is_missing_not_zero(analyzer):
    result = analyzer.paired_lag_diagnostic(metrics([100, None]), metrics([None, 100]))
    assert result["status"] == "no_common_eligible_windows"
    assert result["baseline_common_median_ms"] is None
    assert result["delta_of_common_medians_ms"] is None


@pytest.mark.parametrize("field", ["start_ns", "end_ns", "min_correlation", "max_lag_ms"])
def test_window_grid_and_threshold_must_match(analyzer, field):
    left, right = metrics([100]), metrics([80])
    right["latency"]["windows"][0][field] += 1
    with pytest.raises(ValueError, match="identity/settings"):
        analyzer.paired_lag_diagnostic(left, right)


@pytest.mark.parametrize("mutation", ["windows", "duplicate", "joint", "nan", "bool", "null"])
def test_invalid_pairing_and_nonfinite_eligible_values_fail(analyzer, mutation):
    left, right = metrics([100]), metrics([80])
    window = right["latency"]["windows"][0]
    if mutation == "windows":
        right["latency"]["windows"] = []
    elif mutation == "duplicate":
        window["joint_windows"] *= 2
    elif mutation == "joint":
        window["joint_windows"][0]["joint"] = 1
    else:
        window["joint_windows"][0]["lag_ms"] = {
            "nan": float("nan"), "bool": True, "null": None,
        }[mutation]
    with pytest.raises(ValueError):
        analyzer.paired_lag_diagnostic(left, right)


def test_joint_order_does_not_change_pair_identity_and_input_is_immutable(analyzer):
    left, right = metrics([100, 120]), metrics([80, 90])
    original = deepcopy(right)
    normal = analyzer.paired_lag_diagnostic(left, right)
    assert right == original
    right["latency"]["windows"][0]["joint_windows"].reverse()
    assert analyzer.paired_lag_diagnostic(left, right) == normal


def test_missing_cells_are_not_silently_removed_from_macro_mean(analyzer):
    assert analyzer._mean_complete([1, None, 3]) is None
    assert analyzer._mean_complete([1, 3]) == 2
    assert analyzer._difference(None, 2) is None
    assert analyzer._improved(None) is False


def test_saved_selection_must_reproduce_before_diagnostics(analyzer, tmp_path, monkeypatch):
    (tmp_path / "selection.json").write_text(json.dumps({"changed": True}))
    monkeypatch.setattr(analyzer, "select_from_batches", lambda *a: {"changed": False})
    with pytest.raises(ValueError, match="does not reproduce"):
        analyzer.analyze_confirmation(tmp_path)


def test_report_uses_exclusions_counts_and_null_latency_not_a_false_win(
    analyzer, tmp_path, monkeypatch,
):
    candidate = {"candidate_id": "frozen_adaptive", "eligible": False,
                 "exclusions": ["noisy:insufficient_pose_coverage"], "robust_score": None,
                 "profiles": {"noisy": {"means": {"display_mpjpe_mm": 8.}}}}
    selection = {"candidates": [candidate], "selected_candidate_id": "kalman_cv",
                 "candidate_processors": {"frozen_adaptive": {"method": "adaptive"}},
                 "protocol_sha256": "a" * 64, "declaration_sha256": "b" * 64}
    (tmp_path / "selection.json").write_text(json.dumps(selection))
    (tmp_path / "selection_protocol.json").write_text(json.dumps({
        "profile_ids": ["noisy"], "seeds": [42, 43], "sequences": ["one"],
    }))
    rows, mapping = [], {}
    folder = tmp_path / "batch"
    folder.mkdir()
    for seed in (42, 43):
        for method in ("kalman_cv", "frozen_adaptive"):
            run_id = f"{method}-{seed}"
            row = {"run_id": run_id, "directory": run_id, "sequence_id": "one"}
            rows.append(row)
            mapping[run_id] = {"candidate_id": method, "profile_id": "noisy"}
            child = folder / run_id
            child.mkdir()
            (child / "resolved_config.json").write_text(json.dumps({"seed": seed}))
            lag = 100 if method == "kalman_cv" else 80 if seed == 42 else None
            report = {**metrics([lag]), "display_error": {"mpjpe_mm": 10 if
                      method == "kalman_cv" else 8}, "jitter": {"acceleration_error_m_s2": 5},
                      "runtime": {"processor_p95_ms": 2.},
                      "adaptive_prediction": {"mean_alpha": .25},
                      "learned_prediction": {"observation_output_counts": {"predicted": 10}}}
            (child / "metrics.json").write_text(json.dumps(report))
    (folder / "summary.json").write_text(json.dumps({"runs": rows}))
    (tmp_path / "selection_manifest.json").write_text(json.dumps({
        "batches": [{"directory": "batch", "cells": mapping}],
    }))
    monkeypatch.setattr(analyzer, "select_from_batches", lambda *a: selection)
    result = analyzer.analyze_confirmation(tmp_path)
    assert result["adaptive_reasons"] == candidate["exclusions"]
    assert result["adaptive_eligible"] is False
    assert result["advantage_claim"] == "not_assessed"
    p = result["profiles"]["noisy"]
    assert p["improved_error_cells"] == 2
    assert p["improved_both_cells"] == 0
    assert p["cells_with_common_latency"] == 1
    assert p["common_subset_latency_delta_macro_ms"] is None
    assert p["seed_macro_changes"]["43"]["display_error_delta_mm"] == -2
