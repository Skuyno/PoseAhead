import pytest

from poseahead.prediction.stability import HORIZONS, SEEDS, summarize_stability


def rows():
    return [{"horizon_ms": h, "seed": s, "model_mm": float(s - 22),
             "hold_mm": 30., "velocity_mm": 12.} for h in HORIZONS for s in SEEDS]


def test_summary_counts_and_range():
    result = summarize_stability(rows())
    assert len(result) == 4
    assert result[0]["model_mean_mm"] == 21
    assert result[0]["model_min_mm"] == 20
    assert result[0]["model_max_mm"] == 22
    assert result[0]["beats_hold_runs"] == 3
    assert result[0]["beats_velocity_runs"] == 0


def test_incomplete_duplicate_and_mismatched_inputs_rejected():
    with pytest.raises(ValueError, match="incomplete"):
        summarize_stability(rows()[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        summarize_stability(rows() + rows()[:1])
    data = rows()
    data[0]["velocity_mm"] = 13
    with pytest.raises(ValueError, match="baselines differ"):
        summarize_stability(data)


def test_nonfinite_result_rejected():
    data = rows()
    data[0]["model_mm"] = float("nan")
    with pytest.raises(ValueError, match="metric"):
        summarize_stability(data)
