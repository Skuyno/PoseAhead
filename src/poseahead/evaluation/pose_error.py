"""Unaligned canonical pose/root error at explicitly declared prediction targets."""

from bisect import bisect_left

import numpy as np

from poseahead.core import PoseSequence


def score_pose_error(
    estimates: PoseSequence, reference: PoseSequence, *, interpolation: str = "exact",
    max_gap_ns: int = 50_000_000,
) -> dict:
    """Score 17-joint pelvis-relative MPJPE and independent absolute root error.

    Linear reference interpolation is offline-only, explicitly selected, and
    never extrapolates or bridges a gap larger than max_gap_ns. A joint must be
    valid with positive confidence in the estimate and both reference endpoints.
    Missing pairs are excluded with coverage counts, never scored as zero error.
    Caller is responsible for supplying a genuine reference with matching camera
    geometry, not just a file sharing a clock label.
    """
    if estimates.clock_domain != reference.clock_domain:
        raise ValueError("estimate and reference clock domains differ")
    if interpolation not in ("exact", "linear"):
        raise ValueError("reference interpolation must be exact or linear")
    if type(max_gap_ns) is not int or max_gap_ns <= 0:
        raise ValueError("max_gap_ns must be a positive integer")
    times = [f.source_timestamp_ns for f in reference.frames]
    joint_errors, root_errors = [], []
    matched_frames = scored_frames = 0
    for estimate in estimates.frames:
        target = estimate.metadata.get("target_timestamp_ns", estimate.source_timestamp_ns)
        if type(target) is not int or not 0 <= target < 2**64:
            raise ValueError("estimate target must be a uint64 timestamp")
        offset = estimate.metadata.get("target_offset_ns")
        if offset is not None and (
            type(offset) is not int or target - estimate.source_timestamp_ns != offset
        ):
            raise ValueError("estimate target and offset disagree")
        index = bisect_left(times, target)
        if index < len(times) and times[index] == target:
            left = right = reference.frames[index]
            alpha = 0.0
        elif interpolation == "linear" and 0 < index < len(times):
            left, right = reference.frames[index - 1:index + 1]
            gap = right.source_timestamp_ns - left.source_timestamp_ns
            if gap > max_gap_ns:
                continue
            alpha = (target - left.source_timestamp_ns) / gap
        else:
            continue
        matched_frames += 1
        valid = (estimate.joint_validity & (estimate.joint_confidence > 0)
                 & left.joint_validity & (left.joint_confidence > 0)
                 & right.joint_validity & (right.joint_confidence > 0))
        expected = ((1 - alpha) * left.joint_positions.astype(np.float64)
                    + alpha * right.joint_positions.astype(np.float64))
        errors = np.linalg.norm(estimate.joint_positions.astype(np.float64) - expected, axis=1)
        joint_errors.extend(errors[valid].tolist())
        scored_frames += int(valid.any())
        if all(f.root_position is not None for f in (estimate, left, right)):
            root = ((1 - alpha) * left.root_position.astype(np.float64)
                    + alpha * right.root_position.astype(np.float64))
            root_errors.append(float(np.linalg.norm(estimate.root_position - root)))
    total = len(estimates.frames)
    return {
        "protocol": "canonical17_pelvis_relative_unaligned_including_pelvis",
        "reference_sequence_id": reference.sequence_id,
        "estimate_sequence_id": estimates.sequence_id,
        "clock_domain": estimates.clock_domain,
        "target_policy": "declared_target_or_source_timestamp",
        "reference_interpolation": interpolation,
        "max_interpolation_gap_ns": max_gap_ns if interpolation == "linear" else None,
        "mpjpe_mm": float(np.mean(joint_errors) * 1000) if joint_errors else None,
        "root_error_mm": float(np.mean(root_errors) * 1000) if root_errors else None,
        "estimate_frames": total, "matched_reference_frames": matched_frames,
        "scored_pose_frames": scored_frames, "valid_joint_pairs": len(joint_errors),
        "possible_joint_pairs": total * 17,
        "joint_coverage": len(joint_errors) / (total * 17),
        "root_pairs": len(root_errors), "root_coverage": len(root_errors) / total,
    }
