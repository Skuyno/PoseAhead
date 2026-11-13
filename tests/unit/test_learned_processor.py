from dataclasses import replace

import numpy as np
import pytest

from poseahead.core import PoseFrame, PoseSequence
from poseahead.processing.learned import LearnedProcessor
from poseahead.processing.simulation import simulate_processing


class FakePredictor:
    history_frames = 3
    step_ns = 20_000_000
    horizon_ms = 33

    def predict(self, positions, timestamps_ns):
        output = positions[-1].copy()
        output[1:, 0] += .033
        return output


def processor(monkeypatch, tmp_path, **parameters):
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "poseahead.processing.learned.FixedHorizonPredictor", lambda _: FakePredictor(),
    )
    return LearnedProcessor(checkpoint=tmp_path, **parameters)


def pose(i, milliseconds=None):
    timestamp = i * 20_000_000 if milliseconds is None else milliseconds * 1_000_000
    p = np.zeros((17, 3))
    p[1:, 0] = timestamp / 1e9
    return PoseFrame(i, timestamp, timestamp + 5_000_000, p, np.ones(17), np.ones(17, bool),
                     root_position=[1, 2, 3])


def test_warmup_then_prediction_keeps_observation_time_and_root(monkeypatch, tmp_path):
    p = processor(monkeypatch, tmp_path)
    first = p.process(pose(0))
    assert first.metadata["prediction_status"] == "warmup"
    assert first.metadata["target_offset_ns"] == 0
    p.process(pose(1))
    result = p.process(pose(2))
    assert result.metadata["prediction_status"] == "predicted"
    assert result.metadata["target_timestamp_ns"] == 73_000_000
    assert result.source_timestamp_ns == 40_000_000
    assert result.produced_timestamp_ns == 45_000_000
    np.testing.assert_array_equal(result.root_position, [1, 2, 3])
    assert result.joint_positions[1, 0] == pytest.approx(.073)
    assert p.process(pose(3)).joint_positions[1, 0] == pytest.approx(.093)


def test_prefilter_only_changes_history_not_raw_fallback_or_root(monkeypatch, tmp_path):
    p = processor(monkeypatch, tmp_path, history_prefilter_ms=25)
    first, second = pose(0), pose(1)
    p.process(first)
    fallback = p.process(second)
    np.testing.assert_array_equal(fallback.joint_positions, second.joint_positions)
    current = replace(pose(2), root_position=[3, 4, 5])
    result = p.process(current)
    assert .033 < result.joint_positions[1, 0] < .073
    np.testing.assert_array_equal(result.root_position, current.root_position)
    assert result.source_timestamp_ns == current.source_timestamp_ns
    p.reset()
    assert p.process(first).metadata["prediction_status"] == "warmup"


def test_prefilter_reset_on_loss_and_gap(monkeypatch, tmp_path):
    p = processor(monkeypatch, tmp_path, history_prefilter_ms=25,
                  history_sampling="linear_past")
    p.process(pose(0))
    p.process(pose(1))
    p.process(replace(pose(2), joint_validity=np.zeros(17, bool)))
    fresh = pose(3)
    p.process(fresh)
    np.testing.assert_array_equal(p._history[-1].joint_positions, fresh.joint_positions)
    gap = pose(4, 500)
    p.process(gap)
    np.testing.assert_array_equal(p._history[-1].joint_positions, gap.joint_positions)


def test_prefilter_prefix_is_causal_and_reset_reproducible(monkeypatch, tmp_path):
    p = processor(monkeypatch, tmp_path, history_prefilter_ms=25)
    frames = [pose(i) for i in range(10)]
    first = [p.process(f) for f in frames]
    p.reset()
    changed = [f if i < 5 else replace(f, joint_positions=f.joint_positions * 10)
               for i, f in enumerate(frames)]
    second = [p.process(f) for f in changed]
    for a, b in zip(first[:5], second[:5], strict=True):
        np.testing.assert_array_equal(a.joint_positions, b.joint_positions)


@pytest.mark.parametrize("value", [-1, float("nan"), 1001, True])
def test_invalid_prefilter_rejected(monkeypatch, tmp_path, value):
    with pytest.raises(ValueError, match="history_prefilter"):
        processor(monkeypatch, tmp_path, history_prefilter_ms=value)


def test_gaps_invalidity_and_reset_return_raw_without_future_credit(monkeypatch, tmp_path):
    p = processor(monkeypatch, tmp_path)
    for i in range(3):
        p.process(pose(i))
    gap = p.process(pose(3, 65))
    assert gap.metadata["prediction_status"] == "irregular_interval"
    assert gap.metadata["target_offset_ns"] == 0
    invalid = p.process(replace(pose(4, 85), joint_validity=np.zeros(17, bool)))
    assert invalid.metadata["prediction_status"] == "invalid_observation"
    assert not invalid.joint_validity.any()
    assert p.process(pose(5, 105)).metadata["prediction_status"] == "warmup"
    p.reset()
    assert p.process(pose(0)).metadata["prediction_status"] == "warmup"


def test_delay_and_reset_are_deterministic_without_future_input(monkeypatch, tmp_path):
    p = processor(monkeypatch, tmp_path)
    data = PoseSequence("test", "clock", tuple(pose(i) for i in range(10)))
    times = [f.produced_timestamp_ns + 66_000_000 for f in data.frames]
    first = list(simulate_processing(data, p, times, delay_ns=66_000_000))
    second = list(simulate_processing(data, p, times, delay_ns=66_000_000))
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a.pose.joint_positions, b.pose.joint_positions)
        assert a.pose.produced_timestamp_ns + 66_000_000 <= a.evaluation_timestamp_ns
    with pytest.raises(ValueError, match="prior predictions"):
        p.process(replace(pose(10), metadata={"target_timestamp_ns": 233_000_000}))


def test_checkpoint_pin_mismatch_rejected_before_predictor_load(tmp_path):
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        LearnedProcessor(checkpoint=tmp_path, checkpoint_sha256="wrong")


def test_inference_error_does_not_advance_processor_state(monkeypatch, tmp_path):
    p = processor(monkeypatch, tmp_path)
    p.process(pose(0))
    p.process(pose(1))
    original = p.predictor.predict

    def fail(*args):
        raise RuntimeError("synthetic inference failure")

    p.predictor.predict = fail
    with pytest.raises(RuntimeError, match="inference failure"):
        p.process(pose(2))
    assert p._previous.frame_id == 1
    assert len(p._history) == 2
    p.predictor.predict = original
    assert p.process(pose(2)).metadata["prediction_status"] == "predicted"


def test_linear_past_sampling_reconstructs_known_motion_without_lookahead(monkeypatch, tmp_path):
    p = processor(monkeypatch, tmp_path, history_sampling="linear_past")
    original = p.predictor.predict
    sampled = []

    def inspect(positions, times):
        sampled.append((positions.copy(), times))
        return original(positions, times)

    p.predictor.predict = inspect
    p.process(pose(0, 0))
    p.process(pose(1, 30))
    result = p.process(pose(2, 55))
    positions, times = sampled[-1]
    assert times == (15_000_000, 35_000_000, 55_000_000)
    np.testing.assert_allclose(positions[:, 1, 0], [.015, .035, .055], atol=1e-8)
    assert max(times) == result.source_timestamp_ns
    assert result.joint_positions[1, 0] == pytest.approx(.088)
    before = result.joint_positions.copy()
    p.process(pose(3, 90))
    np.testing.assert_array_equal(before, result.joint_positions)


def test_resampling_does_not_bridge_tracking_loss_and_bounds_memory(monkeypatch, tmp_path):
    p = processor(monkeypatch, tmp_path, history_sampling="linear_past",
                  max_observation_gap_ms=40, max_history_observations=4)
    for i in range(10):
        result = p.process(pose(i, i))
        assert len(p._history) <= 4
    assert result.metadata["prediction_status"] == "history_capacity"
    gap = p.process(pose(10, 70))
    assert gap.metadata["prediction_status"] == "tracking_gap"
    assert gap.metadata["target_offset_ns"] == 0
    p.process(pose(11, 100))
    assert p.process(pose(12, 125)).metadata["prediction_status"] == "predicted"
    p.process(replace(pose(13, 140), joint_validity=np.zeros(17, bool)))
    assert p.process(pose(14, 160)).metadata["prediction_status"] == "warmup"


@pytest.mark.parametrize("sampling", ["exact", "linear_past"])
def test_on_grid_resampling_matches_exact_predictor(monkeypatch, tmp_path, sampling):
    p = processor(monkeypatch, tmp_path, history_sampling=sampling)
    for i in range(10):
        result = p.process(pose(i))
        expected = i * .02 + (.033 if i >= 2 else 0)
        assert result.joint_positions[1, 0] == pytest.approx(expected)
