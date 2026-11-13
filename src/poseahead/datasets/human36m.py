"""Adapter for absolute Human3.6M 32-joint CAMERA coordinates in millimetres.

Joint indices follow the Human3.6M skeleton used by VideoPose3D:
https://github.com/facebookresearch/VideoPose3D/blob/main/common/h36m_dataset.py
File decoding, dataset access and world-to-camera calibration are separate steps.
"""

from __future__ import annotations

import numpy as np

from poseahead.core import JOINT_COUNT, Joint, PoseFrame
from poseahead.core.sequence import PoseSequence, frame_timestamp_ns

# Canonical order, with a placeholder for the derived chest.
_SOURCE_INDICES = (0, 12, 12, 13, 14, 17, 18, 19, 25, 26, 27, 6, 7, 8, 1, 2, 3)


def camera_poses_to_sequence(
    positions_mm: object,
    *,
    sequence_id: str,
    camera_id: str,
    fps_numerator: int = 50,
    fps_denominator: int = 1,
    first_frame_index: int = 0,
) -> PoseSequence:
    """Convert prepared camera-space ground truth; never infer units or axes.

    Source index 13 maps to canonical neck, 14 to head; source 15 is unused.
    Chest is synthesized between source 12 and 13, matching the live convention.
    Timestamps represent ideal dataset event time, not measured inference latency.
    """
    if not isinstance(camera_id, str) or not camera_id.strip():
        raise ValueError("camera_id must be a nonempty string")
    if not isinstance(sequence_id, str) or not sequence_id.strip():
        raise ValueError("sequence_id must be a nonempty string")
    source = np.asarray(positions_mm, dtype=np.float64)
    if source.ndim != 3 or source.shape[1:] != (32, 3) or len(source) == 0:
        raise ValueError("Human3.6M camera positions must have shape (N, 32, 3), N > 0")
    if not np.isfinite(source).all():
        raise ValueError("ground-truth positions must be finite")
    # Validate the index/rate before arithmetic with frame indices.
    frame_timestamp_ns(
        first_frame_index, fps_numerator=fps_numerator, fps_denominator=fps_denominator
    )
    clock_domain = f"dataset:human36m:{sequence_id}:{camera_id}"
    frames = []
    for offset, pose in enumerate(source):
        frame_id = first_frame_index + offset
        event_ns = frame_timestamp_ns(
            frame_id, fps_numerator=fps_numerator, fps_denominator=fps_denominator
        )
        absolute = pose[list(_SOURCE_INDICES)] / 1000.0
        absolute[Joint.CHEST] = (absolute[Joint.SPINE] + absolute[Joint.NECK]) / 2
        root = absolute[Joint.PELVIS].copy()
        frames.append(PoseFrame(
            frame_id=frame_id,
            source_timestamp_ns=event_ns,
            produced_timestamp_ns=event_ns,
            joint_positions=absolute - root,
            root_position=root,
            joint_confidence=np.ones(JOINT_COUNT),
            joint_validity=np.ones(JOINT_COUNT, dtype=bool),
            metadata={
                "dataset": "Human3.6M",
                "source_skeleton": "human36m_32",
                "camera_id": camera_id,
                "clock_domain": clock_domain,
                "timestamp_semantics": "dataset_event_time_no_processing_delay",
                "root_translation_status": "ground_truth",
                "confidence_semantics": "ground_truth_validity_not_detector_probability",
                "derived_joints": ["chest"],
            },
        ))
    return PoseSequence(sequence_id, clock_domain, tuple(frames))
