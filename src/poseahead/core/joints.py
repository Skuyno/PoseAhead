"""Canonical PoseAhead skeleton definition."""

from __future__ import annotations

from enum import IntEnum


class Joint(IntEnum):
    """Stable wire/storage order of the canonical 17-joint skeleton."""

    PELVIS = 0
    SPINE = 1
    CHEST = 2
    NECK = 3
    HEAD = 4
    LEFT_SHOULDER = 5
    LEFT_ELBOW = 6
    LEFT_WRIST = 7
    RIGHT_SHOULDER = 8
    RIGHT_ELBOW = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    LEFT_KNEE = 12
    LEFT_ANKLE = 13
    RIGHT_HIP = 14
    RIGHT_KNEE = 15
    RIGHT_ANKLE = 16


JOINT_COUNT = len(Joint)

# Parent indices use -1 for the root. This tuple is part of the persisted format.
JOINT_PARENTS: tuple[int, ...] = (
    -1,  # pelvis
    Joint.PELVIS,
    Joint.SPINE,
    Joint.CHEST,
    Joint.NECK,
    Joint.CHEST,
    Joint.LEFT_SHOULDER,
    Joint.LEFT_ELBOW,
    Joint.CHEST,
    Joint.RIGHT_SHOULDER,
    Joint.RIGHT_ELBOW,
    Joint.PELVIS,
    Joint.LEFT_HIP,
    Joint.LEFT_KNEE,
    Joint.PELVIS,
    Joint.RIGHT_HIP,
    Joint.RIGHT_KNEE,
)
