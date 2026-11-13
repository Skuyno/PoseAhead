"""Ordered poses in one explicitly named event-clock domain."""

from __future__ import annotations

from dataclasses import dataclass

from poseahead.core.pose_frame import PoseFrame


@dataclass(frozen=True, slots=True)
class PoseSequence:
    sequence_id: str
    clock_domain: str
    frames: tuple[PoseFrame, ...]

    def __post_init__(self) -> None:
        for name in ("sequence_id", "clock_domain"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        frames = tuple(self.frames)
        if not frames:
            raise ValueError("a pose sequence must contain at least one frame")
        for index, frame in enumerate(frames):
            if not isinstance(frame, PoseFrame):
                raise ValueError("sequence entries must be PoseFrames")
            declared_clock = frame.metadata.get("clock_domain")
            if declared_clock is not None and declared_clock != self.clock_domain:
                raise ValueError("frame clock_domain differs from sequence clock_domain")
            if index:
                previous = frames[index - 1]
                if frame.frame_id <= previous.frame_id:
                    raise ValueError("frame IDs must be strictly increasing")
                if frame.source_timestamp_ns <= previous.source_timestamp_ns:
                    raise ValueError("source timestamps must be strictly increasing")
                if frame.produced_timestamp_ns < previous.produced_timestamp_ns:
                    raise ValueError("produced timestamps must be nondecreasing")
        object.__setattr__(self, "frames", frames)

    @property
    def duration_ns(self) -> int:
        return self.frames[-1].source_timestamp_ns - self.frames[0].source_timestamp_ns


def frame_timestamp_ns(
    frame_index: int, *, fps_numerator: int, fps_denominator: int = 1, origin_ns: int = 0
) -> int:
    """Exact rational frame time, rounded down once; no accumulated float drift."""
    for name, value in (
        ("frame_index", frame_index), ("fps_numerator", fps_numerator),
        ("fps_denominator", fps_denominator), ("origin_ns", origin_ns),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if fps_numerator == 0 or fps_denominator == 0:
        raise ValueError("frame rate must be positive")
    if fps_numerator > 1_000_000_000 * fps_denominator:
        raise ValueError("frame rate exceeds nanosecond clock resolution")
    timestamp = origin_ns + frame_index * 1_000_000_000 * fps_denominator // fps_numerator
    if timestamp >= 2**64:
        raise ValueError("timestamp exceeds uint64 storage range")
    return timestamp
