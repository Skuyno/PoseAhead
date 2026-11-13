"""Replay immutable frames without replacing their original clock values."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator

from poseahead.core import PoseFrame, PoseSequence


def replay(
    sequence: PoseSequence,
    *,
    speed: float | None = None,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    sleep: Callable[[float], None] = time.sleep,
) -> Iterator[PoseFrame]:
    """None speed means accelerated replay; positive speed paces availability.

    Scheduling uses produced time because a live observation is unavailable before
    inference completes. Source/event timestamps remain untouched for processors.
    Slow consumers cause no accumulated scheduling drift and no frames are dropped.
    """
    if speed is not None and (
        type(speed) not in (int, float) or not math.isfinite(speed) or speed <= 0
    ):
        raise ValueError("speed must be None or finite and positive")
    start = clock_ns() if speed is not None else 0
    origin = sequence.frames[0].produced_timestamp_ns
    for frame in sequence.frames:
        if speed is not None:
            due = start + int((frame.produced_timestamp_ns - origin) / speed)
            while (remaining := due - clock_ns()) > 0:
                sleep(min(remaining / 1e9, 0.05))
        yield frame
