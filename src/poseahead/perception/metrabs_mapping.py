"""Map MeTRAbs H36M-17 metric output to the canonical PoseAhead skeleton."""

from __future__ import annotations

from typing import Any

import numpy as np

from poseahead.core import JOINT_COUNT, Joint, PoseFrame

METRABS_SKELETON = "h36m_17"
METRABS_JOINT_NAMES = (
    "pelv",
    "rhip",
    "rkne",
    "rank",
    "lhip",
    "lkne",
    "lank",
    "spin",
    "neck",
    "head",
    "htop",
    "lsho",
    "lelb",
    "lwri",
    "rsho",
    "relb",
    "rwri",
)


def metrabs_to_pose_frame(
    keypoints_mm: object,
    *,
    detection_confidence: float,
    confidence_threshold: float,
    frame_id: int,
    source_timestamp_ns: int,
    produced_timestamp_ns: int,
    metadata: dict[str, Any] | None = None,
) -> PoseFrame:
    """Convert one absolute H36M-17 pose in millimetres to canonical metres.

    MeTRAbs does not publish per-joint confidences. The selected person's detector
    confidence is therefore carried as an explicitly labelled proxy for every joint.
    """
    source = np.asarray(keypoints_mm, dtype=np.float32)
    if source.shape == (1, len(METRABS_JOINT_NAMES), 3):
        source = source[0]
    if source.shape != (len(METRABS_JOINT_NAMES), 3):
        raise ValueError(f"MeTRAbs keypoints must have shape (17, 3), got {source.shape}")
    if (
        type(detection_confidence) not in (int, float)
        or not np.isfinite(detection_confidence)
        or not 0 <= detection_confidence <= 1
    ):
        raise ValueError("detection_confidence must be finite and in [0, 1]")
    if (
        type(confidence_threshold) not in (int, float)
        or not np.isfinite(confidence_threshold)
        or not 0 <= confidence_threshold <= 1
    ):
        raise ValueError("confidence_threshold must be finite and in [0, 1]")

    source = source / 1000.0
    output = np.zeros((JOINT_COUNT, 3), dtype=np.float32)
    direct = {
        Joint.PELVIS: 0,
        Joint.SPINE: 7,
        Joint.NECK: 8,
        Joint.HEAD: 9,
        Joint.LEFT_SHOULDER: 11,
        Joint.LEFT_ELBOW: 12,
        Joint.LEFT_WRIST: 13,
        Joint.RIGHT_SHOULDER: 14,
        Joint.RIGHT_ELBOW: 15,
        Joint.RIGHT_WRIST: 16,
        Joint.LEFT_HIP: 4,
        Joint.LEFT_KNEE: 5,
        Joint.LEFT_ANKLE: 6,
        Joint.RIGHT_HIP: 1,
        Joint.RIGHT_KNEE: 2,
        Joint.RIGHT_ANKLE: 3,
    }
    for target, source_index in direct.items():
        output[target] = source[source_index]
    output[Joint.CHEST] = (output[Joint.SPINE] + output[Joint.NECK]) * 0.5

    finite = np.isfinite(output).all(axis=1)
    output[~finite] = 0
    person_valid = detection_confidence >= confidence_threshold and detection_confidence > 0
    validity = finite & person_valid
    confidence = np.full(JOINT_COUNT, detection_confidence, dtype=np.float32)
    confidence[~validity] = 0
    root = output[Joint.PELVIS].copy()
    if validity[Joint.PELVIS]:
        output -= root
    else:
        output[:] = 0
        validity[:] = False
        confidence[:] = 0

    return PoseFrame(
        frame_id=frame_id,
        source_timestamp_ns=source_timestamp_ns,
        produced_timestamp_ns=produced_timestamp_ns,
        root_position=root if validity[Joint.PELVIS] else None,
        joint_positions=output,
        joint_confidence=confidence,
        joint_validity=validity,
        metadata={
            "source_skeleton": METRABS_SKELETON,
            "confidence_semantics": "person_detection_proxy_no_per_joint_scores",
            **(metadata or {}),
        },
    )
