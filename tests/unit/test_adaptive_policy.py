from dataclasses import replace

import pytest

from poseahead.prediction.policy import ThresholdPolicy
from poseahead.prediction.predictability import HORIZONS_MS, PredictabilityFeatures


def parameters(**changes):
    return {"version": 1, "error_limits_m": {str(h): .04 for h in HORIZONS_MS},
            "max_horizon_ms": 100, "minimum_matured": 3,
            "maximum_acceleration_m_s2": 20., "mode": "horizon_only", **changes}


def features(**changes):
    values = {"source_timestamp_ns": 500_000_000, "status": "ready", "minimum_confidence": 1.,
              "speed_m_s": 1., "acceleration_m_s2": 0., "recent_error_m": (.01,) * 4,
              "matured_counts": (3,) * 4, "pending_predictions": 4, "discarded_predictions": 0,
              **changes}
    return PredictabilityFeatures(**values)


def test_longest_eligible_horizon_with_explicit_budget_and_blending_ablation():
    f = features()
    fixed = ThresholdPolicy(**parameters()).decide(f, HORIZONS_MS)
    blend = ThresholdPolicy(**parameters(mode="horizon_blend")).decide(f, HORIZONS_MS)
    assert fixed.horizon_ms == blend.horizon_ms == 100
    assert fixed.alpha == 1 and blend.alpha == pytest.approx(.75)
    assert fixed.score == blend.score == pytest.approx(.75)
    assert "not_probability" in blend.to_dict()["score_semantics"]
    longer = ThresholdPolicy(**parameters(max_horizon_ms=150)).decide(f, HORIZONS_MS)
    assert longer.horizon_ms == 150


def test_bad_long_forecasts_fall_back_to_shorter_or_raw_not_best_unknown_error():
    p = ThresholdPolicy(**parameters())
    f = features(recent_error_m=(.01, .05, None, .01))
    assert p.decide(f, HORIZONS_MS).horizon_ms == 33
    assert p.decide(f, (66, 100)).horizon_ms == 0
    assert p.decide(replace(f, matured_counts=(2,) * 4), HORIZONS_MS).alpha == 0
    assert p.decide(features(recent_error_m=(None,) * 4), HORIZONS_MS).score is None


@pytest.mark.parametrize("status", ["warmup", "tracking_gap", "invalid_or_low_confidence"])
def test_nonready_support_always_returns_raw(status):
    d = ThresholdPolicy(**parameters()).decide(features(status=status), HORIZONS_MS)
    assert (d.horizon_ms, d.alpha, d.score, d.reason) == (0, 0, None, status)


@pytest.mark.parametrize("acceleration", [None, -1, float("nan"), float("inf"), 20.01])
def test_unknown_nonfinite_or_abrupt_acceleration_returns_raw(acceleration):
    d = ThresholdPolicy(**parameters()).decide(
        features(acceleration_m_s2=acceleration), HORIZONS_MS,
    )
    assert d.horizon_ms == 0


def test_threshold_boundary_and_missing_model_are_explicit():
    f = features(acceleration_m_s2=20., recent_error_m=(.04,) * 4)
    d = ThresholdPolicy(**parameters()).decide(f, (33, 66))
    assert d.horizon_ms == 66 and d.alpha == 1 and d.score == 0
    assert ThresholdPolicy(**parameters(mode="horizon_blend")).decide(f, HORIZONS_MS).alpha == 0


@pytest.mark.parametrize("changes", [
    {"version": True}, {"version": 2}, {"mode": "oracle"}, {"max_horizon_ms": 101},
    {"max_horizon_ms": True}, {"minimum_matured": 0}, {"minimum_matured": 257},
    {"minimum_matured": True}, {"maximum_acceleration_m_s2": 0},
    {"maximum_acceleration_m_s2": True}, {"maximum_acceleration_m_s2": float("nan")},
    {"error_limits_m": {33: .04}}, {"error_limits_m": {}},
    {"error_limits_m": {str(h): float("inf") for h in HORIZONS_MS}},
    {"error_limits_m": {str(h): True for h in HORIZONS_MS}},
])
def test_invalid_policy_is_rejected(changes):
    with pytest.raises(ValueError):
        ThresholdPolicy(**parameters(**changes))


@pytest.mark.parametrize("available", [(33, 33), (33, 101), (True,), (33.,)])
def test_invalid_available_horizons(available):
    with pytest.raises(ValueError):
        ThresholdPolicy(**parameters()).decide(features(), available)


def test_policy_does_not_alias_mutable_parameter_dictionary():
    values = parameters()
    p = ThresholdPolicy(**values)
    values["error_limits_m"]["100"] = 1e-6
    p.parameters["error_limits_m"]["100"] = 1e-6
    assert p.decide(features(), HORIZONS_MS).horizon_ms == 100
