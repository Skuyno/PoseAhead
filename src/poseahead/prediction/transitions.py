"""Causal smoothing of current-anchor forecast weights, never stale pose arrays."""

from dataclasses import dataclass
from math import exp, isfinite

from poseahead.prediction.predictability import HORIZONS_MS


@dataclass(frozen=True, slots=True)
class ForecastMix:
    weights: tuple[float, ...]

    @property
    def alpha(self):
        return sum(self.weights)

    @property
    def horizon_ms(self):
        return max((h for h, w in zip(HORIZONS_MS, self.weights, strict=True) if w > 0), default=0)


class ForecastTransition:
    def __init__(self, time_constant_ms=0):
        if (type(time_constant_ms) not in (int, float) or not isfinite(time_constant_ms)
                or not 0 <= time_constant_ms <= 1000):
            raise ValueError("transition_time_constant_ms must be finite in [0, 1000]")
        self.tau_ns = time_constant_ms * 1_000_000
        self.reset()

    def reset(self):
        self._time = None
        self._weights = (0.,) * len(HORIZONS_MS)

    def step(self, timestamp_ns, decision, available_horizons, *, hard_reset=False):
        if (type(timestamp_ns) is not int or not 0 <= timestamp_ns < 2**64
                or self._time is not None and timestamp_ns <= self._time):
            raise ValueError("transition needs increasing uint64 source timestamps")
        available = tuple(available_horizons)
        if (any(type(h) is not int or h not in HORIZONS_MS for h in available)
                or len(set(available)) != len(available)):
            raise ValueError("invalid available transition horizons")
        if (type(decision.alpha) not in (int, float) or not 0 <= decision.alpha <= 1
                or type(decision.horizon_ms) is not int
                or (decision.alpha == 0 and decision.horizon_ms != 0)
                or (decision.alpha > 0 and decision.horizon_ms not in available)):
            raise ValueError("inconsistent requested transition decision")
        target = tuple(decision.alpha if h == decision.horizon_ms else 0. for h in HORIZONS_MS)
        if hard_reset:
            weights = (0.,) * len(HORIZONS_MS)
        elif not self.tau_ns:
            weights = target
        else:
            beta = 0. if self._time is None else 1 - exp(-(timestamp_ns - self._time) / self.tau_ns)
            weights = tuple((1 - beta) * old + beta * new if h in available else 0.
                            for h, old, new in zip(HORIZONS_MS, self._weights, target, strict=True))
            # Finite release tail. No unsupported horizon or stale forecast survives.
            weights = tuple(w if w >= 1e-6 else 0. for w in weights)
        total = sum(weights)
        if total > 1:
            weights = tuple(w / total for w in weights)
        self._time, self._weights = timestamp_ns, weights
        return ForecastMix(weights)
