from dataclasses import replace

import numpy as np
import pytest

from poseahead.core import PoseFrame
from poseahead.prediction.predictability import CausalPredictabilityTracker


@pytest.mark.parametrize("window", [20, 100, 200])
def test_window_fit_exact_quadratic_dynamics_with_irregular_past_samples(window):
    tracker = CausalPredictabilityTracker(dynamics_window_ms=window)
    times = sorted({0, 5, 13, window // 2, window - 1, window})
    for i, ms in enumerate(times):
        f = tracker.observe(pose(i, ms, acceleration=True))
        if ms < window:
            assert f.status == "warmup"
    assert f.status == "ready"
    assert f.acceleration_m_s2 == pytest.approx(1, abs=1e-5)
    assert f.speed_m_s == pytest.approx(window / 1000, abs=1e-5)


def test_window_dynamics_reject_noise_amplification_without_filtering_matured_error():
    raw = CausalPredictabilityTracker()
    smooth = CausalPredictabilityTracker(dynamics_window_ms=200)
    rng = np.random.default_rng(42)
    raw_acc, smooth_acc = [], []
    for i in range(240):
        ms = i * 25 // 3  # Approximately 120 Hz, with integer-ms irregular intervals.
        frame = pose(i, ms)
        positions = frame.joint_positions + rng.normal(0, .01, (17, 3))
        positions[0] = 0
        frame = replace(frame, joint_positions=positions)
        a, b = raw.observe(frame), smooth.observe(frame)
        if b.status == "ready":
            raw_acc.append(a.acceleration_m_s2)
            smooth_acc.append(b.acceleration_m_s2)
        if i == 30:
            prediction = forecast(frame, error=.1)
            raw.record_prediction(prediction)
            smooth.record_prediction(prediction)
    assert np.median(smooth_acc) < np.median(raw_acc) * .1
    assert np.median(smooth_acc) < 20
    assert a.recent_error_m == b.recent_error_m  # Raw observation error remains raw.


@pytest.mark.parametrize("loss", ["gap", "confidence", "invalid"])
def test_window_dynamics_clear_on_loss_and_never_bridge_old_support(loss):
    tracker = CausalPredictabilityTracker(dynamics_window_ms=100, minimum_confidence=.3)
    for i in range(8):
        f = tracker.observe(pose(i))
    assert f.status == "ready"
    current = pose(8)
    if loss == "gap":
        current = pose(8, 500)
    elif loss == "confidence":
        current = replace(current, joint_confidence=np.full(17, .2))
    else:
        current = replace(current, joint_validity=np.zeros(17, bool))
    assert tracker.observe(current).acceleration_m_s2 is None
    f = tracker.observe(pose(9, 520 if loss == "gap" else 180))
    assert f.status == "warmup" and f.acceleration_m_s2 is None


def test_window_capacity_and_future_suffix_causality():
    tracker = CausalPredictabilityTracker(dynamics_window_ms=200)
    first = [tracker.observe(pose(i)) for i in range(30)]
    tracker.reset()
    second = [tracker.observe(pose(i) if i < 20 else replace(
        pose(i), joint_positions=pose(i).joint_positions * 10)) for i in range(30)]
    assert first[:20] == second[:20]
    tracker.reset()
    for i in range(1000):
        f = tracker.observe(pose(i, 0 if i == 0 else i))
        assert len(tracker._dynamics_history) <= 256
    assert f.status == "ready"


@pytest.mark.parametrize("value", [True, 19, 1001, -1, .1, float("nan")])
def test_invalid_dynamics_window(value):
    with pytest.raises(ValueError, match="dynamics_window_ms"):
        CausalPredictabilityTracker(dynamics_window_ms=value)


def pose(index, ms=None, *, acceleration=False):
    ms = index * 20 if ms is None else ms
    p = np.zeros((17, 3))
    p[1:, 0] = .5 * (ms / 1000) ** 2 if acceleration else ms / 1000
    return PoseFrame(index, ms * 1_000_000, ms * 1_000_000 + 5_000_000,
                     p, np.ones(17), np.ones(17, bool), metadata={"clock_domain": "synthetic"})


def forecast(anchor, horizon=33, error=0.):
    positions = anchor.joint_positions.copy()
    positions[1:, 0] += horizon / 1000 + error
    return replace(anchor, joint_positions=positions, metadata={
        **anchor.metadata, "target_offset_ns": horizon * 1_000_000,
        "target_timestamp_ns": anchor.source_timestamp_ns + horizon * 1_000_000,
    })


def test_constant_velocity_has_known_speed_and_zero_acceleration():
    tracker = CausalPredictabilityTracker()
    first = tracker.observe(pose(0))
    assert first.status == "warmup" and first.speed_m_s is None
    assert tracker.observe(pose(1)).speed_m_s == pytest.approx(1)
    features = tracker.observe(pose(2))
    assert features.status == "ready"
    assert features.acceleration_m_s2 == pytest.approx(0, abs=1e-5)
    assert features.to_dict()["recent_observation_error_m"]["33"] is None


def test_acceleration_uses_velocity_midpoint_time_on_irregular_frames():
    tracker = CausalPredictabilityTracker()
    tracker.observe(pose(0, 0, acceleration=True))
    tracker.observe(pose(1, 10, acceleration=True))
    result = tracker.observe(pose(2, 40, acceleration=True))
    assert result.acceleration_m_s2 == pytest.approx(1, abs=1e-5)


def test_error_cannot_mature_before_target_and_uses_only_arrived_interpolation():
    tracker = CausalPredictabilityTracker()
    first = pose(0)
    tracker.observe(first)
    tracker.record_prediction(forecast(first, error=.1))
    before = tracker.observe(pose(1))
    assert before.recent_error_m[0] is None
    assert before.matured_counts[0] == 0
    assert before.pending_predictions == 1
    mature = tracker.observe(pose(2))
    assert mature.recent_error_m[0] == pytest.approx(.1, abs=1e-6)
    assert mature.matured_counts[0] == 1
    assert mature.pending_predictions == 0
    # Returned snapshots do not acquire information after a future observation.
    assert before.recent_error_m[0] is None


def test_four_horizons_mature_independently_at_explicit_times():
    tracker = CausalPredictabilityTracker()
    first = pose(0)
    tracker.observe(first)
    for horizon in (33, 66, 100, 150):
        tracker.record_prediction(forecast(first, horizon))
    at40 = tracker.observe(pose(1, 40))
    assert at40.matured_counts == (1, 0, 0, 0)
    at100 = tracker.observe(pose(2, 100))
    assert at100.matured_counts == (1, 1, 1, 0)
    at150 = tracker.observe(pose(3, 150))
    assert at150.matured_counts == (1, 1, 1, 1)
    assert at150.recent_error_m == pytest.approx((0, 0, 0, 0), abs=1e-6)


@pytest.mark.parametrize("loss", ["invalid", "confidence", "gap"])
def test_tracking_loss_or_long_gap_discards_pending_and_old_error_support(loss):
    tracker = CausalPredictabilityTracker(minimum_confidence=.3)
    first = pose(0)
    tracker.observe(first)
    tracker.record_prediction(forecast(first))
    current = pose(1)
    if loss == "invalid":
        current = replace(current, joint_validity=np.zeros(17, bool))
    elif loss == "confidence":
        current = replace(current, joint_confidence=np.full(17, .2))
    else:
        current = pose(1, 200)
    result = tracker.observe(current)
    assert result.pending_predictions == 0 and result.discarded_predictions == 1
    assert result.recent_error_m == (None,) * 4
    assert result.speed_m_s is None
    if loss != "gap":
        with pytest.raises(ValueError, match="latest usable"):
            tracker.record_prediction(forecast(current))


def test_pending_buffers_are_bounded_and_duplicate_registration_is_rejected():
    tracker = CausalPredictabilityTracker(pending_per_horizon=2)
    for index in range(5):
        current = pose(index, index)
        features = tracker.observe(current)
        tracker.record_prediction(forecast(current, 150))
    assert features.pending_predictions == 2 and features.discarded_predictions == 2
    with pytest.raises(ValueError, match="already recorded"):
        tracker.record_prediction(forecast(current, 150))
    tracker.reset()
    result = tracker.observe(pose(0))
    assert result.pending_predictions == 0 and result.discarded_predictions == 0


def test_future_predictions_cannot_be_used_as_observation_features():
    tracker = CausalPredictabilityTracker()
    with pytest.raises(ValueError, match="observations"):
        tracker.observe(forecast(pose(0)))
    tracker.observe(pose(0))  # Rejected input did not mutate order state.


def test_recording_older_or_different_clock_prediction_is_rejected():
    tracker = CausalPredictabilityTracker()
    tracker.observe(pose(0))
    tracker.observe(pose(1))
    with pytest.raises(ValueError, match="latest usable"):
        tracker.record_prediction(forecast(pose(0)))
    with pytest.raises(ValueError, match="anchor/clock"):
        tracker.record_prediction(replace(forecast(pose(1)), metadata={
            **forecast(pose(1)).metadata, "clock_domain": "another",
        }))


def test_recent_error_window_uses_only_latest_matured_predictions():
    tracker = CausalPredictabilityTracker(error_history=2)
    for index in range(5):
        anchor = pose(index, index * 40)
        result = tracker.observe(anchor)
        tracker.record_prediction(forecast(anchor, error=index * .1))
    assert result.matured_counts[0] == 4
    assert result.recent_error_m[0] == pytest.approx(.25, abs=1e-6)  # latest .2 and .3


def test_future_suffix_cannot_change_features_of_the_shared_prefix():
    trackers = [CausalPredictabilityTracker(), CausalPredictabilityTracker()]
    prefixes = []
    for tracker in trackers:
        first = pose(0)
        tracker.observe(first)
        tracker.record_prediction(forecast(first))
        prefixes.append(tracker.observe(pose(1)).to_dict())
    changed = pose(2)
    positions = changed.joint_positions.copy()
    positions[1:, 0] += 10
    trackers[0].observe(changed)
    trackers[1].observe(replace(changed, joint_positions=positions))
    assert prefixes[0] == prefixes[1]


@pytest.mark.parametrize("metadata", [
    {"target_offset_ns": 50_000_000, "target_timestamp_ns": 50_000_000},
    {"target_offset_ns": 33_000_000, "target_timestamp_ns": 34_000_000},
    {"target_offset_ns": 33_000_000.0, "target_timestamp_ns": 33_000_000},
    {"target_offset_ns": 0, "target_timestamp_ns": 0},
])
def test_invalid_prediction_target_is_rejected_without_queue_mutation(metadata):
    tracker = CausalPredictabilityTracker()
    first = pose(0)
    tracker.observe(first)
    with pytest.raises(ValueError, match="explicit supported"):
        tracker.record_prediction(replace(forecast(first), metadata={
            "clock_domain": "synthetic", **metadata,
        }))
    assert tracker.observe(pose(1)).pending_predictions == 0


def test_clock_or_frame_order_changes_require_explicit_reset():
    tracker = CausalPredictabilityTracker()
    tracker.observe(pose(0))
    with pytest.raises(ValueError, match="increasing frame"):
        tracker.observe(pose(0))
    with pytest.raises(ValueError, match="clock changed"):
        tracker.observe(replace(pose(1), metadata={"clock_domain": "changed"}))
    tracker.reset()
    assert tracker.observe(pose(0)).status == "warmup"


@pytest.mark.parametrize("parameters", [
    {"max_gap_ms": 0}, {"max_gap_ms": True}, {"minimum_confidence": float("nan")},
    {"minimum_confidence": True}, {"minimum_confidence": 1.1},
    {"error_history": 0}, {"error_history": True}, {"pending_per_horizon": 0},
    {"pending_per_horizon": True},
])
def test_invalid_tracker_parameters_fail_before_any_observation(parameters):
    with pytest.raises(ValueError):
        CausalPredictabilityTracker(**parameters)
