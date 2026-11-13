"""Reproducible observation stress; references and event-clock timestamps stay intact."""

from dataclasses import replace

import numpy as np

from poseahead.core import PoseSequence

CORRUPTION_DEFAULTS = {
    "noise_std_m": 0., "dropout_probability": 0.,
    "outlier_probability": 0., "outlier_std_m": 0.,
    "outlier_magnitude_m": 0., "tracking_gap_period_ms": 0, "tracking_gap_duration_ms": 0,
    "gap_period_ms": 0, "gap_duration_ms": 0, "observation_stride": 1,
}


def corruption_settings(config):
    """Canonical profile, omitting disabled extensions for legacy artifact compatibility."""
    values = {k: config.get(k, default) for k, default in CORRUPTION_DEFAULTS.items()}
    for name in ("noise_std_m", "dropout_probability", "outlier_probability", "outlier_std_m",
                 "outlier_magnitude_m"):
        value = values[name]
        if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    for name in ("dropout_probability", "outlier_probability"):
        if values[name] > 1:
            raise ValueError(f"{name} must not exceed one")
    modes = bool(values["outlier_std_m"]) + bool(values["outlier_magnitude_m"])
    if modes != int(bool(values["outlier_probability"])):
        raise ValueError("outliers require positive probability and exactly one std/magnitude mode")
    for name in ("gap_period_ms", "gap_duration_ms", "observation_stride",
                 "tracking_gap_period_ms", "tracking_gap_duration_ms"):
        if type(values[name]) is not int or values[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    for prefix in ("gap", "tracking_gap"):
        period, duration = values[f"{prefix}_period_ms"], values[f"{prefix}_duration_ms"]
        if (bool(period) != bool(duration) or period * 1_000_000 >= 2**64
                or (period and not 0 < duration < period)):
            raise ValueError(f"{prefix} requires 0 < duration < period within uint64 range")
    if not 1 <= values["observation_stride"] <= 120:
        raise ValueError("observation_stride must be an integer in [1, 120]")
    return {k: v for k, v in values.items()
            if k in ("noise_std_m", "dropout_probability") or v != CORRUPTION_DEFAULTS[k]}


def degrade_observations(sequence, *, seed, noise_std_m, dropout_probability,
                         outlier_probability=0., outlier_std_m=0., gap_period_ms=0,
                         gap_duration_ms=0, observation_stride=1, outlier_magnitude_m=0.,
                         tracking_gap_period_ms=0, tracking_gap_duration_ms=0):
    """Causal synthetic stress, not a calibrated camera error model.

    Noise/outliers affect root-relative non-pelvis XYZ only, not root translation.
    Dropout can invalidate any joint; outliers remain confident/valid on purpose.
    Periodic missing-arrival bursts start after one full period from the first
    timestamp. Stride keeps the first and every Nth original frame. No timestamps
    or IDs are compacted: the simulator holds the last available output in gaps.
    Every original frame consumes base RNG draws, even when its arrival is omitted;
    outliers use a separate deterministic stream, preserving legacy noise/dropout.
    Fixed-magnitude outliers use uniformly random directions. Delivered tracking
    gaps retain arrivals/times but invalidate all joints/confidence and remove root
    and rotations; they are distinct from missing-arrival bursts above.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    profile = corruption_settings({
        "noise_std_m": noise_std_m, "dropout_probability": dropout_probability,
        "outlier_probability": outlier_probability, "outlier_std_m": outlier_std_m,
        "gap_period_ms": gap_period_ms, "gap_duration_ms": gap_duration_ms,
        "observation_stride": observation_stride,
        "outlier_magnitude_m": outlier_magnitude_m,
        "tracking_gap_period_ms": tracking_gap_period_ms,
        "tracking_gap_duration_ms": tracking_gap_duration_ms,
    })
    rng = np.random.default_rng(seed)
    outlier_rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,)))
    origin = sequence.frames[0].source_timestamp_ns
    period_ns, duration_ns = gap_period_ms * 1_000_000, gap_duration_ms * 1_000_000
    tracking_period_ns = tracking_gap_period_ms * 1_000_000
    tracking_duration_ns = tracking_gap_duration_ms * 1_000_000
    frames = []
    for index, frame in enumerate(sequence.frames):
        noise = rng.normal(0, noise_std_m, (17, 3))
        valid = frame.joint_validity & (rng.random(17) >= dropout_probability)
        if outlier_probability:
            mask = outlier_rng.random(17) < outlier_probability
            if outlier_magnitude_m:
                direction = outlier_rng.normal(size=(17, 3))
                direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-15)
                noise += direction * outlier_magnitude_m * mask[:, None]
            else:
                noise += outlier_rng.normal(0, outlier_std_m, (17, 3)) * mask[:, None]
        noise[0] = 0
        elapsed = frame.source_timestamp_ns - origin
        missing = period_ns and elapsed >= period_ns and elapsed % period_ns < duration_ns
        if index % observation_stride or missing:
            continue
        tracking_lost = bool(tracking_period_ns and elapsed >= tracking_period_ns
                             and elapsed % tracking_period_ns < tracking_duration_ns)
        if tracking_lost:
            valid[:] = False
        positions = frame.joint_positions.astype(np.float64) + noise
        positions[~valid] = 0
        confidence = frame.joint_confidence.copy()
        confidence[~valid] = 0
        frames.append(replace(frame, joint_positions=positions, joint_validity=valid,
                              joint_confidence=confidence,
                              root_position=None if tracking_lost else frame.root_position,
                              root_rotation=None if tracking_lost else frame.root_rotation,
                              joint_rotations=None if noise_std_m or outlier_probability
                              or tracking_lost
                              else frame.joint_rotations,
                              metadata={**frame.metadata, "noise_profile": {
                                  "seed": seed, "std_m": noise_std_m,
                                  "dropout_probability": dropout_probability,
                                  "generator": "numpy_default_rng_PCG64",
                                  **{k: v for k, v in profile.items()
                                     if k not in ("noise_std_m", "dropout_probability")},
                                  **({"tracking_gap_active": tracking_lost}
                                     if tracking_period_ns else {}),
                              }}))
    return PoseSequence(sequence.sequence_id, sequence.clock_domain, tuple(frames))
