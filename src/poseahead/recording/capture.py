"""Bounded in-memory recording of observations, including explicit detection gaps."""

from dataclasses import replace

import numpy as np

from poseahead.core import PoseFrame, PoseSequence


class ObservationRecorder:
    def __init__(self, *, sequence_id: str, clock_domain: str, max_frames: int = 100_000):
        if type(max_frames) is not int or max_frames < 1:
            raise ValueError("max_frames must be a positive integer")
        self.sequence_id = sequence_id
        self.clock_domain = clock_domain
        self.max_frames = max_frames
        self._frames: list[PoseFrame] = []

    def append(
        self, pose: PoseFrame | None, *, frame_id: int,
        source_timestamp_ns: int, completed_timestamp_ns: int,
    ) -> None:
        if len(self._frames) >= self.max_frames:
            raise RuntimeError("recording frame limit reached; increase the explicit limit")
        if pose is None:
            pose = PoseFrame(
                frame_id, source_timestamp_ns, completed_timestamp_ns,
                np.zeros((17, 3)), np.zeros(17), np.zeros(17, dtype=bool),
                metadata={"observation_status": "missing", "clock_domain": self.clock_domain},
            )
        elif pose.frame_id != frame_id or pose.source_timestamp_ns != source_timestamp_ns:
            raise ValueError("observation and capture identity differ")
        declared_clock = pose.metadata.get("clock_domain", self.clock_domain)
        if declared_clock != self.clock_domain:
            raise ValueError("observation clock differs from recorder clock")
        pose = replace(pose, metadata={**pose.metadata, "clock_domain": self.clock_domain})
        self._frames.append(pose)

    def sequence(self) -> PoseSequence:
        return PoseSequence(self.sequence_id, self.clock_domain, tuple(self._frames))
