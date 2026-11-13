"""Descriptive seed stability, never a confidence interval over subjects."""

import math

import numpy as np

HORIZONS = (33, 66, 100, 150)
SEEDS = (42, 43, 44)


def summarize_stability(rows):
    expected = {(h, s) for h in HORIZONS for s in SEEDS}
    seen = set()
    for row in rows:
        pair = (row["horizon_ms"], row["seed"])
        if pair not in expected or pair in seen:
            raise ValueError("unexpected or duplicate stability run")
        seen.add(pair)
        for field in ("model_mm", "hold_mm", "velocity_mm"):
            if not math.isfinite(row[field]) or row[field] < 0:
                raise ValueError("invalid stability metric")
    if seen != expected:
        raise ValueError("incomplete stability experiment")
    summary = []
    for horizon in HORIZONS:
        group = [r for r in rows if r["horizon_ms"] == horizon]
        for field in ("hold_mm", "velocity_mm"):
            if max(r[field] for r in group) - min(r[field] for r in group) > 1e-6:
                raise ValueError("baselines differ across seeds; inputs may not match")
        errors = [r["model_mm"] for r in group]
        summary.append({"horizon_ms": horizon, "seeds": list(SEEDS),
                        "model_mean_mm": float(np.mean(errors)),
                        "model_min_mm": min(errors), "model_max_mm": max(errors),
                        "hold_mm": group[0]["hold_mm"],
                        "velocity_mm": group[0]["velocity_mm"],
                        "beats_hold_runs": sum(r["model_mm"] < r["hold_mm"] - 1e-5
                                               for r in group),
                        "beats_velocity_runs": sum(r["model_mm"] < r["velocity_mm"] - 1e-5
                                                   for r in group)})
    return summary
