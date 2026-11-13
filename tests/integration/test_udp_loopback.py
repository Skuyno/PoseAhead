"""Loopback integration test for one-datagram pose delivery."""

from __future__ import annotations

import socket

import numpy as np

from poseahead.core import JOINT_COUNT, PoseFrame
from poseahead.transport import UdpPoseSender, decode_udp_packet


def test_udp_sender_delivers_one_decodable_datagram() -> None:
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(1.0)
    host, port = receiver.getsockname()
    frame = PoseFrame(
        frame_id=9,
        source_timestamp_ns=100,
        produced_timestamp_ns=200,
        joint_positions=np.zeros((JOINT_COUNT, 3)),
        joint_confidence=np.ones(JOINT_COUNT),
        joint_validity=np.ones(JOINT_COUNT, dtype=np.bool_),
    )
    try:
        with UdpPoseSender(host, port) as sender:
            sent = sender.send(frame, send_timestamp_ns=300)
        packet, _ = receiver.recvfrom(1_200)
    finally:
        receiver.close()

    assert sent == len(packet)
    assert decode_udp_packet(packet).frame.frame_id == 9
