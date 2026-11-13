from dataclasses import replace

import numpy as np
import pytest

from poseahead.core import PoseFrame
from poseahead.core.sequence import PoseSequence, frame_timestamp_ns


def pose(frame_id=0, source=0, produced=100):
    return PoseFrame(frame_id, source, produced, np.zeros((17, 3)),
                     np.ones(17), np.ones(17, dtype=bool))


def test_sequence_accepts_gaps_variable_rate_and_shared_production_time():
    frames = [pose(), pose(3, 25), pose(7, 90)]
    sequence = PoseSequence("clip", "test-clock", frames)
    frames.clear()
    assert len(sequence.frames) == 3
    assert sequence.duration_ns == 90


@pytest.mark.parametrize("following", [pose(0, 10), pose(1, 0), pose(1, 10, 99)])
def test_sequence_rejects_duplicate_id_event_time_or_backwards_production(following):
    with pytest.raises(ValueError):
        PoseSequence("clip", "test-clock", (pose(), following))


def test_sequence_rejects_mixed_clock_domains_and_empty_sequence():
    with pytest.raises(ValueError, match="clock_domain"):
        PoseSequence("clip", "a", (replace(pose(), metadata={"clock_domain": "b"}),))
    with pytest.raises(ValueError, match="at least one"):
        PoseSequence("clip", "a", ())


def test_rational_timestamps_do_not_accumulate_rounding_drift():
    assert frame_timestamp_ns(30_000, fps_numerator=30_000, fps_denominator=1001) == (
        1_001_000_000_000
    )
    assert frame_timestamp_ns(1, fps_numerator=30) == 33_333_333
    assert frame_timestamp_ns(3, fps_numerator=30) == 100_000_000


@pytest.mark.parametrize("options", [
    {"fps_numerator": 0}, {"fps_numerator": True},
    {"fps_numerator": 50, "fps_denominator": 0},
    {"fps_numerator": 1_000_000_001},
    {"fps_numerator": 50, "origin_ns": 2**64},
])
def test_invalid_clock_parameters(options):
    with pytest.raises(ValueError):
        frame_timestamp_ns(0, **options)
