"""Versioned, deterministic thresholds; scores are margins, not probabilities."""

from dataclasses import dataclass
from math import isfinite

from poseahead.prediction.predictability import HORIZONS_MS, PredictabilityFeatures


@dataclass(frozen=True, slots=True)
class AdaptiveDecision:
    horizon_ms: int
    alpha: float
    score: float | None
    reason: str

    def to_dict(self):
        return {"horizon_ms": self.horizon_ms, "alpha": self.alpha,
                "score": self.score, "reason": self.reason,
                "score_semantics": "one_minus_observation_error_over_limit_not_probability"}


class ThresholdPolicy:
    """Choose the longest supported horizon within an explicit compensation budget.

    All research thresholds must be supplied. Construction does not imply they
    were selected on validation. Unknown/insufficient errors never mean zero.
    horizon_only and horizon_blend share gates, making blending an ablation.
    """

    def __init__(self, *, version, error_limits_m, max_horizon_ms,
                 minimum_matured, maximum_acceleration_m_s2, mode):
        if type(version) is not int or version != 1:
            raise ValueError("unsupported adaptive policy version")
        if mode not in ("horizon_only", "horizon_blend"):
            raise ValueError("adaptive mode must be horizon_only or horizon_blend")
        if type(max_horizon_ms) is not int or max_horizon_ms not in HORIZONS_MS:
            raise ValueError("max_horizon_ms must be a supported horizon")
        if type(minimum_matured) is not int or not 1 <= minimum_matured <= 256:
            raise ValueError("minimum_matured must be in [1, 256]")
        if (type(maximum_acceleration_m_s2) not in (int, float)
                or not isfinite(maximum_acceleration_m_s2) or maximum_acceleration_m_s2 <= 0):
            raise ValueError("maximum_acceleration_m_s2 must be finite and positive")
        if (not isinstance(error_limits_m, dict)
                or set(error_limits_m) != {str(h) for h in HORIZONS_MS}):
            raise ValueError("error_limits_m requires exactly string keys 33, 66, 100, 150")
        for limit in error_limits_m.values():
            if type(limit) not in (int, float) or not isfinite(limit) or limit <= 0:
                raise ValueError("error limits must be finite and positive metres")
        self._limits = tuple(float(error_limits_m[str(h)]) for h in HORIZONS_MS)
        self.max_horizon_ms, self.minimum_matured = max_horizon_ms, minimum_matured
        self.maximum_acceleration_m_s2, self.mode = maximum_acceleration_m_s2, mode

    @property
    def parameters(self):
        return {"version": 1, "error_limits_m": dict(zip(
                    map(str, HORIZONS_MS), self._limits, strict=True)),
                "max_horizon_ms": self.max_horizon_ms,
                "minimum_matured": self.minimum_matured,
                "maximum_acceleration_m_s2": self.maximum_acceleration_m_s2, "mode": self.mode}

    def decide(self, features: PredictabilityFeatures, available_horizons):
        available = tuple(available_horizons)
        if (len(set(available)) != len(available)
                or any(type(h) is not int or h not in HORIZONS_MS for h in available)):
            raise ValueError("available horizons must be unique supported integers")
        def fallback(reason):
            return AdaptiveDecision(0, 0., None, reason)
        if features.status != "ready":
            return fallback(features.status)
        acceleration = features.acceleration_m_s2
        if acceleration is None or not isfinite(acceleration) or acceleration < 0:
            return fallback("unknown_acceleration")
        if acceleration > self.maximum_acceleration_m_s2:
            return fallback("acceleration_limit")
        for i in reversed(range(len(HORIZONS_MS))):
            h, error = HORIZONS_MS[i], features.recent_error_m[i]
            if h > self.max_horizon_ms or h not in available:
                continue
            if features.matured_counts[i] < self.minimum_matured or error is None:
                continue
            if not isfinite(error) or error < 0 or error > self._limits[i]:
                continue
            score = max(0., min(1., 1 - error / self._limits[i]))
            alpha = score if self.mode == "horizon_blend" else 1.
            if alpha > 0:
                return AdaptiveDecision(h, alpha, score, "selected")
        return fallback("no_supported_forecast")
