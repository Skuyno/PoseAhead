"""Shared interpolation of an already available observation history."""

from bisect import bisect_left

import numpy as np


def sample_history(history, grid, max_gap_ns):
    """Interpolate inside the supplied history; never extrapolate or read ahead.

    Callers must supply only observations available at the prediction anchor and
    enforce validity/reset semantics before calling this numerical helper.
    """
    if not history or grid[0] < history[0].source_timestamp_ns:
        return None
    times = [f.source_timestamp_ns for f in history]
    samples = []
    for timestamp in grid:
        right = bisect_left(times, timestamp)
        if right == len(times):
            return None
        if times[right] == timestamp:
            samples.append(history[right].joint_positions)
        else:
            left = right - 1
            gap = times[right] - times[left]
            if gap > max_gap_ns:
                return None
            alpha = (timestamp - times[left]) / gap
            samples.append((1 - alpha) * history[left].joint_positions.astype(np.float64)
                           + alpha * history[right].joint_positions.astype(np.float64))
    return np.stack(samples)
