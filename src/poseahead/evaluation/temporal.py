"""Temporal metrics on explicitly aligned physical-time motion samples.

These arrays describe output at evaluation/display time, not observation IDs or
prediction target time. Resampling and clock alignment are caller responsibilities.
"""

from dataclasses import dataclass
from itertools import pairwise

import numpy as np


@dataclass(frozen=True, slots=True)
class MotionSeries:
    timestamps_ns: tuple[int, ...]
    positions_m: np.ndarray
    validity: np.ndarray
    clock_domain: str

    def __post_init__(self):
        times = tuple(self.timestamps_ns)
        if len(times) < 3 or any(type(t) is not int or not 0 <= t < 2**64 for t in times):
            raise ValueError("motion requires at least three uint64 timestamps")
        if any(b <= a for a, b in pairwise(times)):
            raise ValueError("motion timestamps must increase strictly")
        positions = np.asarray(self.positions_m, dtype=np.float64)
        valid = np.asarray(self.validity)
        if (positions.ndim != 3 or positions.shape[0] != len(times)
                or positions.shape[1] < 1 or positions.shape[2] != 3):
            raise ValueError("motion positions must have shape (time, joints, 3)")
        if not np.isfinite(positions).all():
            raise ValueError("motion positions must be finite")
        if valid.dtype != bool or valid.shape != positions.shape[:2]:
            raise ValueError("motion validity must be boolean (time, joints)")
        if not isinstance(self.clock_domain, str) or not self.clock_domain.strip():
            raise ValueError("motion clock_domain must be nonempty")
        object.__setattr__(self, "timestamps_ns", times)
        for name, array in (("positions_m", positions), ("validity", valid)):
            frozen = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
            object.__setattr__(self, name, frozen)


def _aligned(output, reference):
    if output.clock_domain != reference.clock_domain:
        raise ValueError("motion clock domains differ")
    if output.timestamps_ns != reference.timestamps_ns:
        raise ValueError("motion timestamps differ; resample explicitly before scoring")
    if output.positions_m.shape != reference.positions_m.shape:
        raise ValueError("motion layouts differ")


def score_jitter(output: MotionSeries, reference: MotionSeries) -> dict:
    """Three-point acceleration error on a shared, possibly irregular time grid.

    Primary metric is mean Euclidean acceleration residual in m/s^2. Plain output
    acceleration is secondary only. All three samples must be valid in both streams.
    """
    _aligned(output, reference)
    dt = np.array([b - a for a, b in pairwise(output.timestamps_ns)]) / 1e9

    def acceleration(series):
        velocity = np.diff(series.positions_m, axis=0) / dt[:, None, None]
        return 2 * np.diff(velocity, axis=0) / (dt[1:] + dt[:-1])[:, None, None]

    shared = output.validity & reference.validity
    valid = shared[:-2] & shared[1:-1] & shared[2:]
    actual, expected = acceleration(output), acceleration(reference)
    residual = np.linalg.norm(actual - expected, axis=2)[valid]
    smoothness = np.linalg.norm(actual, axis=2)[valid]
    return {
        "acceleration_error_m_s2": float(residual.mean()) if residual.size else None,
        "output_acceleration_m_s2": float(smoothness.mean()) if smoothness.size else None,
        "valid_joint_triples": int(valid.sum()), "possible_joint_triples": int(valid.size),
        "coverage": float(valid.mean()),
        "protocol": "three_point_acceleration_residual_shared_physical_time",
    }


def score_effective_latency(
    output: MotionSeries, reference: MotionSeries, *, max_lag_ms: float = 200,
    min_velocity_variation_m_s: float = 1e-4, min_correlation: float = .6,
    ambiguity_margin: float = .02,
) -> dict:
    """One moving window, joint-wise demeaned vector-velocity correlation.

    Positive lag means output trails reference. Every lag uses the same number of
    samples. Invalid/static, low-correlation, boundary and ambiguous windows are
    excluded. Median combines eligible joints; reports retain individual reasons.
    Parabolic interpolation refines the peak by at most half a sample.
    """
    _aligned(output, reference)
    for name, value in (("max_lag_ms", max_lag_ms),
                        ("min_velocity_variation_m_s", min_velocity_variation_m_s),
                        ("min_correlation", min_correlation),
                        ("ambiguity_margin", ambiguity_margin)):
        if type(value) not in (float, int) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if min_correlation > 1 or ambiguity_margin >= 1:
        raise ValueError("correlation thresholds must be within (0, 1]")
    intervals = [b - a for a, b in pairwise(output.timestamps_ns)]
    if len(set(intervals)) != 1:
        raise ValueError("latency requires uniform sampling; resample explicitly")
    step_ms = intervals[0] / 1e6
    if max_lag_ms >= len(intervals) * step_ms / 2:
        raise ValueError("window too short for requested lag range")
    radius = int(max_lag_ms / step_ms)
    count = len(intervals)
    if radius < 1 or count - 2 * radius < 5:
        raise ValueError("window too short or lag range below sample resolution")
    velocities = [np.diff(s.positions_m, axis=0) / (intervals[0] / 1e9)
                  for s in (output, reference)]
    results, accepted = [], []

    def centered(values):
        return values - values.mean(axis=0)

    for joint in range(output.positions_m.shape[1]):
        record = {"joint": joint, "lag_ms": None, "correlation": None, "status": "invalid"}
        results.append(record)
        if not (output.validity[:, joint].all() and reference.validity[:, joint].all()):
            continue
        observed = centered(velocities[0][radius:count - radius, joint])
        energy = float(np.linalg.norm(observed))
        if energy / np.sqrt(observed.size) < min_velocity_variation_m_s:
            record["status"] = "static_or_constant_velocity"
            continue
        scores = []
        for lag in range(-radius, radius + 1):
            truth = centered(velocities[1][radius - lag:count - radius - lag, joint])
            norm = float(np.linalg.norm(truth))
            scores.append(float(np.sum(observed * truth) / (energy * norm))
                          if norm / np.sqrt(truth.size) >= min_velocity_variation_m_s else -1.)
        peak = int(np.argmax(scores))
        record["correlation"] = float(np.clip(scores[peak], -1, 1))
        if scores[peak] < min_correlation:
            record["status"] = "low_correlation"
            continue
        if peak in (0, len(scores) - 1):
            record["status"] = "lag_boundary"
            continue
        # Separated competing local peaks indicate repeated/ambiguous motion.
        competitors = [i for i in range(1, len(scores) - 1)
                       if abs(i - peak) >= 3 and scores[i] >= scores[i - 1]
                       and scores[i] >= scores[i + 1]]
        competitors.extend(i for i in (0, len(scores) - 1) if abs(i - peak) >= 3)
        if any(scores[i] >= scores[peak] - ambiguity_margin for i in competitors):
            record["status"] = "ambiguous_peak"
            continue
        a, b, c = scores[peak - 1:peak + 2]
        curvature = a - 2 * b + c
        fractional = float(np.clip(.5 * (a - c) / curvature, -.5, .5)) if curvature < 0 else 0
        lag_ms = (peak - radius + fractional) * step_ms
        record.update(status="ok", lag_ms=lag_ms)
        accepted.append(lag_ms)
    return {
        "effective_latency_ms": float(np.median(accepted)) if accepted else None,
        "eligible_joint_windows": len(accepted), "total_joint_windows": len(results),
        "coverage": len(accepted) / len(results), "joint_windows": results,
        "positive_lag_means": "output_trails_reference", "sample_interval_ms": step_ms,
        "max_lag_ms": max_lag_ms, "searched_max_lag_ms": radius * step_ms,
        "min_velocity_variation_m_s": min_velocity_variation_m_s,
        "min_correlation": min_correlation, "ambiguity_margin": ambiguity_margin,
        "protocol": "demeaned_vector_velocity_equal_overlap_parabolic_peak_median_joints",
    }
