"""Offline observation availability with no look-ahead exposed to processors."""

from poseahead.core import PoseFrame, PoseSequence


class DelayedReplay:
    """Inject delay after observation creation, preserving original event timestamps.

    A frame is visible only at produced_timestamp_ns + delay_ns. This also enforces
    source_timestamp_ns <= evaluation_time_ns - delay_ns, including inference cost.
    """

    def __init__(self, sequence: PoseSequence, *, delay_ns: int):
        if type(delay_ns) is not int or not 0 <= delay_ns < 2**64:
            raise ValueError("delay_ns must be a nonnegative uint64 integer")
        self.sequence = sequence
        self.delay_ns = delay_ns
        self.reset()

    def reset(self) -> None:
        self._index = 0
        self._evaluation_time: int | None = None

    def advance(self, evaluation_time_ns: int) -> tuple[PoseFrame, ...]:
        if type(evaluation_time_ns) is not int or not 0 <= evaluation_time_ns < 2**64:
            raise ValueError("evaluation time must be a nonnegative uint64 integer")
        if self._evaluation_time is not None and evaluation_time_ns < self._evaluation_time:
            raise ValueError("evaluation time cannot move backwards without reset")
        ready = []
        while self._index < len(self.sequence.frames):
            frame = self.sequence.frames[self._index]
            if frame.produced_timestamp_ns + self.delay_ns > evaluation_time_ns:
                break
            ready.append(frame)
            self._index += 1
        self._evaluation_time = evaluation_time_ns
        return tuple(ready)
