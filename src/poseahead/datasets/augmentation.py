"""Clip-consistent training corruption; clean targets are handled separately."""

import hashlib
import math


def validate_noise(std_m, seed):
    if type(std_m) not in (int, float) or not math.isfinite(std_m) or not 0 <= std_m <= .1:
        raise ValueError("training noise std must be finite in [0, 0.1] metres")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("training noise seed must be a nonnegative int below 2**63")


def noisy_training_observations(sequence, std_m, seed):
    from poseahead.datasets.splits import sequence_split
    from poseahead.evaluation.experiment import degrade_observations

    validate_noise(std_m, seed)
    if sequence_split(sequence)[0] != "train":
        raise ValueError("training augmentation requires train split")
    identity = f"{seed}\0{sequence.sequence_id}\0{sequence.clock_domain}".encode()
    clip_seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], "little")
    return degrade_observations(sequence, seed=clip_seed, noise_std_m=std_m,
                                dropout_probability=0.)
