import numpy as np
import pytest

from poseahead.core import PoseFrame, PoseSequence
from poseahead.processing.config import METHODS, create_processor
from poseahead.processing.simulation import simulate_processing


def sequence():
    frames = tuple(PoseFrame(
        i, i * 20_000_000, i * 20_000_000 + 5_000_000,
        np.zeros((17, 3)), np.ones(17), np.ones(17, dtype=bool),
    ) for i in range(10))
    return PoseSequence("test", "dataset:test", frames)


@pytest.mark.parametrize("method", METHODS)
def test_delay_hold_and_catchup_have_no_duplicate_processing(method):
    data = sequence()
    snapshots = list(simulate_processing(
        data, create_processor(method), [37_999_999, 38_000_000, 38_000_000, 97_999_999],
        delay_ns=33_000_000,
    ))
    assert snapshots[0].pose is None
    assert snapshots[0].observation_age_ns is None
    assert snapshots[0].represented_timestamp_ns is None
    assert [s.observations_consumed for s in snapshots] == [0, 1, 1, 3]
    assert snapshots[1].pose is snapshots[2].pose
    assert snapshots[3].pose.frame_id == 2
    assert snapshots[3].observation_age_ns == 57_999_999
    assert snapshots[3].clock_domain == data.clock_domain
    assert snapshots[3].represented_timestamp_ns == (
        140_000_000 if method == "kalman_cv" else 40_000_000
    )


@pytest.mark.parametrize("method", METHODS)
def test_simulation_resets_and_matches_direct_causal_processing(method):
    data = sequence()
    processor = create_processor(method)
    direct = [processor.process(frame) for frame in data.frames]
    for _ in range(2):
        simulated = list(simulate_processing(
            data, processor, [frame.produced_timestamp_ns + 66_000_000 for frame in data.frames],
            delay_ns=66_000_000,
        ))
        for left, right in zip(direct, simulated, strict=True):
            np.testing.assert_array_equal(left.joint_positions, right.pose.joint_positions)
            assert left.metadata == right.pose.metadata


def test_backwards_evaluation_clock_rejected():
    with pytest.raises(ValueError, match="backwards"):
        list(simulate_processing(sequence(), create_processor("raw"), [50_000_000, 0]))
