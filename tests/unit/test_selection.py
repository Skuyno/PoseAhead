from copy import deepcopy
from itertools import product

import pytest

from poseahead.evaluation.selection import SELECTION_RULE, select_validation_panel


@pytest.fixture
def panel():
    protocol = {
        "version": 1, "protocol_id": "synthetic-validation-selector",
        "baseline_id": "baseline", "clean_profile": "clean",
        "candidate_ids": ["candidate", "baseline"], "profile_ids": ["clean", "noisy"],
        "seeds": [42, 43], "sequences": ["a", "b"], "rule": dict(SELECTION_RULE),
    }
    rows = [{
        "candidate_id": candidate, "profile_id": profile, "seed": seed,
        "sequence_id": sequence, "split": "validation",
        "display_mpjpe_mm": 10., "jitter_m_s2": 5., "latency_ms": 100.,
        "joint_coverage": 1., "jitter_coverage": 1., "latency_coverage": .8,
        "input_fingerprint": "a" * 64, "pose_mask_sha256": "b" * 64,
        "latency_mask_sha256": "c" * 64,
    } for candidate, profile, seed, sequence in product(
        protocol["candidate_ids"], protocol["profile_ids"], protocol["seeds"],
        protocol["sequences"],
    )]
    return protocol, rows


def test_baseline_wins_exact_ties_even_if_declared_last(panel):
    protocol, rows = panel
    result = select_validation_panel(protocol, rows)
    assert result["selected_candidate_id"] == "baseline"
    assert result["advantage_claim"] == "not_assessed"
    assert result["test_scored"] is False
    assert select_validation_panel(protocol, rows[::-1]) == result


def test_robust_candidate_can_use_clean_ten_percent_budget(panel):
    protocol, rows = panel
    for row in rows:
        if row["candidate_id"] == "candidate":
            row["display_mpjpe_mm"] = 11. if row["profile_id"] == "clean" else 8.
            row["jitter_m_s2"] = 5.5 if row["profile_id"] == "clean" else 4.
    result = select_validation_panel(protocol, rows)
    assert result["selected_candidate_id"] == "candidate"
    assert result["candidates"][0]["robust_score"] == pytest.approx(.8)


def test_lower_error_alone_cannot_win_by_hiding_extra_jitter(panel):
    protocol, rows = panel
    for row in rows:
        if row["candidate_id"] == "candidate" and row["profile_id"] == "noisy":
            row.update(display_mpjpe_mm=1, jitter_m_s2=8)
    result = select_validation_panel(protocol, rows)
    assert result["selected_candidate_id"] == "baseline"
    assert result["candidates"][0]["robust_score"] == 1.6


def test_best_noise_seed_is_not_selected_instead_of_mean(panel):
    protocol, rows = panel
    for row in rows:
        if row["candidate_id"] == "candidate" and row["profile_id"] == "noisy":
            row["display_mpjpe_mm"] = 0. if row["seed"] == 42 else 30.
    result = select_validation_panel(protocol, rows)
    assert result["selected_candidate_id"] == "baseline"
    assert result["candidates"][0]["profiles"]["noisy"]["means"]["display_mpjpe_mm"] == 15.


@pytest.mark.parametrize("field,value", [
    ("display_mpjpe_mm", 11.001), ("jitter_m_s2", 5.501),
    ("joint_coverage", .949), ("jitter_coverage", .949),
    ("pose_mask_sha256", "e" * 64), ("display_mpjpe_mm", None),
])
def test_clean_budget_and_pose_coverage_are_hard_constraints(panel, field, value):
    protocol, rows = panel
    for row in rows:
        if row["candidate_id"] == "candidate" and row["profile_id"] == "clean":
            row[field] = value
    result = select_validation_panel(protocol, rows)
    assert result["candidates"][0]["eligible"] is False
    assert result["selected_candidate_id"] == "baseline"


def test_latency_on_different_masks_is_explicitly_not_comparable(panel):
    protocol, rows = panel
    for row in rows:
        if row["candidate_id"] == "candidate":
            row.update(latency_ms=-100, latency_mask_sha256="f" * 64)
    result = select_validation_panel(protocol, rows)
    assert result["selected_candidate_id"] == "baseline"
    assert result["candidates"][0]["profiles"]["noisy"]["latency_status"] == "not_comparable"
    assert result["candidates"][0]["profiles"]["noisy"]["latency_mean_ms"] is None


def test_zero_baseline_is_handled_without_infinity(panel):
    protocol, rows = panel
    for row in rows:
        row["display_mpjpe_mm"] = 0
    rows[4]["display_mpjpe_mm"] = .01
    result = select_validation_panel(protocol, rows)
    assert result["selected_candidate_id"] == "baseline"
    assert result["candidates"][0]["robust_score"] is None


def test_low_coverage_baseline_does_not_authorize_any_selection(panel):
    protocol, rows = panel
    rows[-1]["joint_coverage"] = 0
    result = select_validation_panel(protocol, rows)
    assert result["status"] == "no_eligible_comparator"
    assert result["selected_candidate_id"] is None


@pytest.mark.parametrize("mutation", [
    "missing", "duplicate", "test", "train", "input", "nan", "negative", "bool", "seed",
])
def test_incomplete_unmatched_or_invalid_panel_is_rejected(panel, mutation):
    protocol, rows = panel
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif mutation in ("test", "train"):
        rows[0]["split"] = mutation
    elif mutation == "input":
        rows[0]["input_fingerprint"] = "f" * 64
    elif mutation == "seed":
        rows[0]["seed"] = True
    else:
        rows[0]["display_mpjpe_mm"] = {
            "nan": float("nan"), "negative": -1, "bool": True,
        }[mutation]
    with pytest.raises(ValueError):
        select_validation_panel(protocol, rows)


@pytest.mark.parametrize("field,value", [
    ("version", True), ("baseline_id", "missing"), ("clean_profile", "missing"),
    ("profile_ids", ["clean"]), ("seeds", [42, 42]), ("seeds", [False]),
    ("candidate_ids", ["candidate", "candidate"]), ("rule", {}),
])
def test_invalid_protocol_is_rejected(panel, field, value):
    protocol, rows = panel
    protocol[field] = value
    with pytest.raises(ValueError):
        select_validation_panel(protocol, rows)
