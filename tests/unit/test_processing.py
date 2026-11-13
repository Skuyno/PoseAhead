import math
from dataclasses import replace

import numpy as np
import pytest

from poseahead.core import PoseFrame, PoseSequence
from poseahead.processing import EmaProcessor, KalmanProcessor, OneEuroProcessor, RawProcessor
from poseahead.processing.adaptive_filters import KalmanExtrapolationProcessor
from poseahead.processing.config import METHODS, create_processor
from poseahead.recording import replay
from poseahead.recording.delay import DelayedReplay


def pose(i, milliseconds, value=0):
    positions = np.zeros((17, 3))
    positions[1:, 0] = value
    timestamp = int(milliseconds * 1e6)
    return PoseFrame(i, timestamp, timestamp + 5_000_000, positions,
                     np.ones(17), np.ones(17, dtype=bool), root_position=[value, 0, 3])


def test_ema_matches_continuous_time_step_response_at_variable_intervals():
    for times in ((0, 10, 20, 100), (0, 25, 75, 100), (0, 100)):
        processor = EmaProcessor(time_constant_ms=50)
        for i, t in enumerate(times):
            result = processor.process(pose(i, t, int(i > 0)))
        assert result.joint_positions[1, 0] == pytest.approx(1 - math.exp(-2), abs=1e-6)
        assert result.root_position[0] == pytest.approx(1 - math.exp(-2), abs=1e-6)
        np.testing.assert_array_equal(result.joint_positions[0], [0, 0, 0])


def test_missing_joint_reacquisition_and_gap_reset_do_not_blend_stale_state():
    processor = EmaProcessor()
    processor.process(pose(0, 0))
    frame = pose(1, 20, 10)
    validity = np.ones(17, dtype=bool)
    validity[1] = False
    lost = processor.process(replace(frame, joint_validity=validity, root_position=None))
    assert not lost.joint_validity[1]
    np.testing.assert_array_equal(lost.joint_positions[1], [0, 0, 0])
    reacquired = processor.process(pose(2, 40, 10))
    assert reacquired.joint_positions[1, 0] == 10
    assert reacquired.root_position[0] == 10
    after_gap = processor.process(pose(3, 500, 20))
    assert after_gap.joint_positions[2, 0] == 20


@pytest.mark.parametrize("factory", [RawProcessor, EmaProcessor, OneEuroProcessor,
                                     KalmanProcessor, KalmanExtrapolationProcessor])
def test_live_and_replay_processing_are_identical_and_resettable(factory):
    observations = tuple(pose(i, i * 20, i) for i in range(5))
    sequence = PoseSequence("clip", "clock", observations)
    processor = factory()
    live = [processor.process(f) for f in observations]
    with pytest.raises(ValueError):
        processor.process(observations[-1])
    processor.reset()
    offline = [processor.process(f) for f in replay(sequence)]
    for left, right in zip(live, offline, strict=True):
        np.testing.assert_array_equal(left.joint_positions, right.joint_positions)
        assert left.source_timestamp_ns == right.source_timestamp_ns
    assert observations[1].joint_positions[1, 0] == 1  # observations were not mutated


def test_injected_delay_respects_production_time_and_exact_boundary():
    sequence = PoseSequence("clip", "clock", (pose(0, 0), pose(3, 40), pose(9, 150)))
    delayed = DelayedReplay(sequence, delay_ns=33_000_000)
    assert delayed.advance(33_000_000) == ()  # inference took another 5 ms
    assert delayed.advance(37_999_999) == ()
    assert delayed.advance(38_000_000)[0] is sequence.frames[0]
    assert delayed.advance(38_000_000) == ()
    assert delayed.advance(78_000_000)[0] is sequence.frames[1]
    assert delayed.advance(187_999_999) == ()
    with pytest.raises(ValueError, match="backwards"):
        delayed.advance(0)
    delayed.reset()
    assert len(delayed.advance(188_000_000)) == 3


@pytest.mark.parametrize("method", METHODS)
def test_processor_never_receives_future_observations_from_delayed_replay(method):
    sequence = PoseSequence("clip", "clock", tuple(pose(i, i * 17, i) for i in range(10)))
    delayed = DelayedReplay(sequence, delay_ns=66_000_000)
    processor = create_processor(method)
    for evaluation in range(0, 250_000_000, 1_000_000):
        for observation in delayed.advance(evaluation):
            assert observation.source_timestamp_ns <= evaluation - 66_000_000
            assert observation.produced_timestamp_ns + 66_000_000 <= evaluation
            processor.process(observation)


@pytest.mark.parametrize("method", METHODS)
def test_future_suffix_cannot_change_processed_prefix(method):
    prefix = tuple(pose(i, i * 20, math.sin(i)) for i in range(10))
    left, right = create_processor(method), create_processor(method)
    expected = [left.process(frame) for frame in prefix]
    actual = [right.process(frame) for frame in prefix]
    for i in range(10, 20):
        right.process(pose(i, i * 20, 100 * math.cos(i)))
    for a, b in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(a.joint_positions, b.joint_positions)
        np.testing.assert_array_equal(a.root_position, b.root_position)


def test_one_euro_beta_zero_matches_fixed_lowpass_and_speed_reduces_lag():
    slow, fast = OneEuroProcessor(beta=0), OneEuroProcessor(beta=10)
    for processor in (slow, fast):
        processor.process(pose(0, 0))
    low = slow.process(pose(1, 20, 1))
    high = fast.process(pose(1, 20, 1))
    alpha = 1 / (1 + 1 / (2 * math.pi * .02))
    assert low.joint_positions[1, 0] == pytest.approx(alpha)
    assert low.joint_positions[1, 0] < high.joint_positions[1, 0] < 1


def test_kalman_linear_motion_prediction_and_covariance_stability():
    processor = KalmanProcessor()
    for i in range(100):
        processor.process(pose(i, i * 20, i * .02))
    prediction = processor.extrapolate(target_timestamp_ns=2_080_000_000)
    assert prediction.joint_positions[1, 0] == pytest.approx(2.08, abs=.005)
    assert prediction.source_timestamp_ns == 1_980_000_000
    assert prediction.metadata["target_timestamp_ns"] == 2_080_000_000
    assert np.linalg.eigvalsh(processor._covariance).min() >= -1e-12
    # Predictions must not feed themselves back into the estimate.
    again = processor.extrapolate(target_timestamp_ns=2_080_000_000)
    np.testing.assert_array_equal(prediction.joint_positions, again.joint_positions)
    next_result = processor.process(pose(100, 2000, 2))
    assert next_result.joint_positions[1, 0] == pytest.approx(2, abs=.005)


@pytest.mark.parametrize("factory", [OneEuroProcessor, KalmanProcessor])
def test_adaptive_filters_reinitialize_after_missing_joint_and_long_gap(factory):
    processor = factory()
    processor.process(pose(0, 0))
    frame = pose(1, 20, 1)
    processor.process(replace(frame, joint_validity=np.zeros(17, dtype=bool), root_position=None))
    result = processor.process(pose(2, 40, 5))
    assert result.joint_positions[1, 0] == 5
    assert result.root_position[0] == 5
    result = processor.process(pose(3, 500, 10))
    assert result.joint_positions[1, 0] == 10
