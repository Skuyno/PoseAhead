"""Explicit bridge from causal simulation outputs to display-time metrics."""

from collections.abc import Iterable

import numpy as np

from poseahead.evaluation.temporal import MotionSeries
from poseahead.processing.simulation import ProcessingSnapshot


def snapshot_motion(snapshots: Iterable[ProcessingSnapshot]) -> MotionSeries:
    """Keep output positions at query times; do not relabel by prediction target.

    No output before the first available observation is represented as invalid,
    not a valid zero pose. Position-only metrics use the canonical 17-joint layout.
    Root translation is deliberately not mixed into these root-relative signals.
    """
    snapshots = tuple(snapshots)
    if not snapshots:
        raise ValueError("at least three snapshots are required")
    clock = snapshots[0].clock_domain
    if any(snapshot.clock_domain != clock for snapshot in snapshots):
        raise ValueError("snapshot clocks differ")
    positions = np.zeros((len(snapshots), 17, 3))
    valid = np.zeros((len(snapshots), 17), bool)
    for i, snapshot in enumerate(snapshots):
        if snapshot.pose is not None:
            if snapshot.pose.produced_timestamp_ns > snapshot.evaluation_timestamp_ns:
                raise ValueError("snapshot contains a not-yet-produced observation")
            positions[i] = snapshot.pose.joint_positions
            valid[i] = snapshot.pose.joint_validity & (snapshot.pose.joint_confidence > 0)
    return MotionSeries(
        tuple(s.evaluation_timestamp_ns for s in snapshots), positions, valid, clock,
    )
