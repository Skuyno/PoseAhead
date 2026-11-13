import base64
import json
import socket
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from poseahead.processing.config import create_processor
from poseahead.runtime.unity_replay import diagnostic_sequence, stream_sequence
from poseahead.transport import UdpPoseSender, decode_udp_packet, encode_udp_packet


class Clock:
    def __init__(self):
        self.now = 10_000_000_000

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(1, round(seconds * 1e9))


class Sender:
    def __init__(self):
        self.sent = []

    def send(self, frame, **fields):
        self.sent.append((frame, fields))


def test_unity_tpose_fixture_matches_python_wire_format():
    path = (Path(__file__).resolve().parents[2] / "unity/PoseAheadAvatar/Assets/PoseAhead"
            / "Tests/Fixtures/retarget_tpose.base64")
    packet = encode_udp_packet(diagnostic_sequence().frames[0], session_id=42, send_timestamp_ns=1)
    assert base64.b64decode(path.read_text().strip()) == packet


def test_unity_eight_phase_fixtures_match_python_wire_format():
    path = (Path(__file__).resolve().parents[2] / "unity/PoseAheadAvatar/Assets/PoseAhead"
            / "Tests/Fixtures/diagnostic_packets.json")
    fixtures = json.loads(path.read_text())
    assert fixtures["version"] == 1
    assert len(fixtures["packets"]) == 8
    sequence = diagnostic_sequence()
    for encoded, frame in zip(fixtures["packets"], sequence.frames[::20], strict=True):
        expected = encode_udp_packet(frame, session_id=42,
                                     send_timestamp_ns=frame.produced_timestamp_ns + 1)
        assert base64.b64decode(encoded) == expected


def test_diagnostic_rig_geometry_and_clock_rebase_do_not_mutate_originals():
    sequence, clock, sender = diagnostic_sequence(), Clock(), Sender()
    originals = tuple(f.joint_positions.copy() for f in sequence.frames)
    result = stream_sequence(sequence, create_processor("raw"), sender,
                             clock_ns=clock, sleep=clock.sleep)
    assert result["sent_frames"] == 160
    assert clock.now == 17_950_000_000
    for original, (wire, fields), points in zip(sequence.frames, sender.sent, originals,
                                               strict=True):
        assert wire.source_timestamp_ns == original.source_timestamp_ns + 10_000_000_000
        assert wire.produced_timestamp_ns <= fields["send_timestamp_ns"]
        assert fields["target_offset_ms"] == 0
        assert wire.metadata["original_clock_domain"] == "synthetic_20hz"
        assert "transport_clock_rebased" not in original.metadata
        np.testing.assert_array_equal(original.joint_positions, points)
    # Left wrist is deliberately invalid only during the tracking-loss phase.
    assert not sequence.frames[120].joint_validity[7]
    assert sequence.frames[140].joint_validity.all()


def test_rebase_preserves_processor_target_time():
    sequence, clock, sender = diagnostic_sequence(), Clock(), Sender()
    stream_sequence(sequence, create_processor("kalman_cv", {"target_offset_ms": 100}), sender,
                    duration_seconds=.2, clock_ns=clock, sleep=clock.sleep)
    assert len(sender.sent) == 4
    for wire, fields in sender.sent:
        assert wire.metadata["target_timestamp_ns"] == wire.source_timestamp_ns + 100_000_000
        assert fields["target_offset_ms"] == 100


def test_recording_from_another_monotonic_epoch_can_be_displayed():
    sequence, clock, sender = diagnostic_sequence(), Clock(), Sender()
    old_epoch = 2**60
    sequence = replace(sequence, frames=tuple(replace(
        f, source_timestamp_ns=f.source_timestamp_ns + old_epoch,
        produced_timestamp_ns=f.produced_timestamp_ns + old_epoch) for f in sequence.frames))
    stream_sequence(sequence, create_processor("raw"), sender, duration_seconds=.1,
                    clock_ns=clock, sleep=clock.sleep)
    assert sender.sent[0][0].source_timestamp_ns == 10_000_000_000
    assert sequence.frames[0].source_timestamp_ns == old_epoch


def test_duration_excludes_long_gap_without_waiting_for_next_frame():
    sequence, clock, sender = diagnostic_sequence(), Clock(), Sender()
    future = replace(sequence.frames[1], source_timestamp_ns=1_000_000_000_000,
                     produced_timestamp_ns=1_000_000_000_000)
    sequence = replace(sequence, frames=(sequence.frames[0], future))
    stream_sequence(sequence, create_processor("raw"), sender, duration_seconds=.1,
                    clock_ns=clock, sleep=clock.sleep)
    assert len(sender.sent) == 1
    assert clock.now == 10_000_000_000


@pytest.mark.parametrize("duration", [0, -1, True, float("nan"), float("inf")])
def test_invalid_duration_rejected_before_sending(duration):
    sender = Sender()
    with pytest.raises(ValueError, match="duration"):
        stream_sequence(diagnostic_sequence(), create_processor("raw"), sender,
                        duration_seconds=duration)
    assert not sender.sent


def test_paced_diagnostic_reaches_actual_localhost_decoder():
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(1)
    try:
        clock = Clock()
        with UdpPoseSender("127.0.0.1", receiver.getsockname()[1]) as sender:
            result = stream_sequence(diagnostic_sequence(), create_processor("raw"), sender,
                                     duration_seconds=.11, clock_ns=clock, sleep=clock.sleep)
        assert result["sent_frames"] == 3
        packets = [decode_udp_packet(receiver.recvfrom(1200)[0]) for _ in range(3)]
        assert [p.frame.frame_id for p in packets] == [0, 1, 2]
        assert len({p.session_id for p in packets}) == 1
        assert all(p.send_timestamp_ns >= p.frame.produced_timestamp_ns for p in packets)
        assert (packets[1].frame.source_timestamp_ns - packets[0].frame.source_timestamp_ns
                == 50_000_000)
    finally:
        receiver.close()


def test_ui_cancel_interrupts_long_recording_gap_without_mutating_frames():
    sequence, clock, sender = diagnostic_sequence(), Clock(), Sender()
    future = replace(sequence.frames[1], source_timestamp_ns=1_000_000_000_000,
                     produced_timestamp_ns=1_000_000_000_000)
    sequence = replace(sequence, frames=(sequence.frames[0], future))
    original = sequence.frames[0].joint_positions.copy()
    result = stream_sequence(sequence, create_processor("raw"), sender,
                             clock_ns=clock, sleep=clock.sleep,
                             stop_requested=lambda: clock.now >= 10_100_000_000)
    assert result["status"] == "stopped"
    assert len(sender.sent) == 1
    assert clock.now <= 10_150_000_000
    np.testing.assert_array_equal(original, sequence.frames[0].joint_positions)


def test_ui_cancel_before_replay_sends_nothing():
    sender, clock = Sender(), Clock()
    result = stream_sequence(diagnostic_sequence(), create_processor("raw"), sender,
                             clock_ns=clock, sleep=clock.sleep, stop_requested=lambda: True)
    assert result["status"] == "stopped"
    assert result["sent_frames"] == 0
    assert not sender.sent
