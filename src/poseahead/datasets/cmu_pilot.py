"""Explicit, limited 20-Hz CMU pilot preparation; no interpolation of inputs."""

import hashlib
from dataclasses import replace

import numpy as np

from poseahead.core import PoseSequence
from poseahead.datasets.splits import CMU_SUBJECT_SPLITS


def skeleton_fingerprint(skeleton):
    """Numeric rest skeleton hash to catch duplicated calibrations across IDs."""
    scale, bones, parents, _ = skeleton
    digest = hashlib.sha256(str(scale).encode())
    for name in sorted(bones):
        offset, basis, dof = bones[name]
        digest.update(repr((name, parents[name], dof)).encode())
        digest.update(np.round(offset, 7).astype("<f8").tobytes())
        digest.update(np.round(basis, 7).astype("<f8").tobytes())
    return digest.hexdigest()


def prepare_pilot(sequence, fingerprint, *, retain_native_rate=False):
    """Label by fixed pilot policy and retain every sixth 120-Hz source frame."""
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("expected skeleton SHA-256 fingerprint")
    if any(f.metadata.get("dataset") != "CMU" or f.metadata.get("fps_numerator") != 120
           for f in sequence.frames):
        raise ValueError("pilot decimation requires CMU at 120 Hz")
    subjects = {f.metadata.get("cmu_subject") for f in sequence.frames}
    if len(subjects) != 1:
        raise ValueError("mixed CMU subjects")
    subject = f"CMU:{next(iter(subjects))}"
    if subject not in CMU_SUBJECT_SPLITS:
        raise ValueError("subject is outside CMU pilot policy")
    for i, frame in enumerate(sequence.frames):
        if frame.source_timestamp_ns != i * 1_000_000_000 // 120:
            raise ValueError("pilot requires original contiguous source timing")
    frames = tuple(replace(f, metadata={
        **f.metadata, "subject": subject, "split": CMU_SUBJECT_SPLITS[subject],
        "usage": "cmu_subject_id_pilot_not_final_benchmark",
        "sampling": ("native_120hz" if retain_native_rate else
                     "every_sixth_source_frame_no_interpolation_20hz"),
        "skeleton_fingerprint": fingerprint,
    }) for f in sequence.frames[::1 if retain_native_rate else 6])
    return PoseSequence(sequence.sequence_id, sequence.clock_domain, frames)


def check_skeleton_partitions(sequences):
    """Reject known duplicated skeletons across splits, not proof of actor identity."""
    seen = {}
    for sequence in sequences:
        metadata = sequence.frames[0].metadata
        fingerprint, split = metadata["skeleton_fingerprint"], metadata["split"]
        if fingerprint in seen and seen[fingerprint] != split:
            raise ValueError("same skeleton calibration occurs across pilot splits")
        seen[fingerprint] = split
