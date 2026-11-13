from math import exp

import pytest

from poseahead.prediction.policy import AdaptiveDecision
from poseahead.prediction.predictability import HORIZONS_MS
from poseahead.prediction.transitions import ForecastTransition


def request(h=100, alpha=1.):
    return AdaptiveDecision(h, alpha, .5 if alpha else None, "selected" if alpha else "gate")


def test_off_is_exact_single_forecast_and_zero_target_fallback():
    transition = ForecastTransition()
    mix = transition.step(0, request(), HORIZONS_MS)
    assert mix.weights == (0, 0, 1, 0) and mix.horizon_ms == 100 and mix.alpha == 1
    raw = transition.step(1, request(0, 0), HORIZONS_MS)
    assert raw.horizon_ms == 0 and raw.alpha == 0


def test_ramp_switch_and_release_use_elapsed_source_time_not_frame_count():
    transition = ForecastTransition(100)
    transition.step(0, request(0, 0), HORIZONS_MS)
    first = transition.step(100_000_000, request(), HORIZONS_MS)
    assert first.alpha == pytest.approx(1 - exp(-1))
    second = transition.step(200_000_000, request(33, .5), HORIZONS_MS)
    assert second.weights[0] == pytest.approx(.5 * (1 - exp(-1)))
    assert second.weights[2] == pytest.approx(first.alpha * exp(-1))
    assert second.horizon_ms == 100  # Largest contributing horizon, not requested 33.
    released = transition.step(300_000_000, request(0, 0), HORIZONS_MS)
    assert released.alpha == pytest.approx(second.alpha * exp(-1))
    assert second.weights[2] > 0  # Immutable earlier snapshot.
    zero = transition.step(10_000_000_000, request(0, 0), HORIZONS_MS)
    assert zero.alpha == 0 and zero.horizon_ms == 0


def test_hard_reset_and_unavailable_forecasts_have_no_release_tail():
    transition = ForecastTransition(50)
    transition.step(0, request(0, 0), HORIZONS_MS)
    transition.step(100_000_000, request(), HORIZONS_MS)
    unavailable = transition.step(120_000_000, request(33), (33,))
    assert unavailable.weights[2] == 0
    mix = transition.step(140_000_000, request(0, 0), (), hard_reset=True)
    assert mix.alpha == 0
    transition.reset()
    assert transition.step(0, request(), HORIZONS_MS).alpha == 0


def test_subdivision_invariance_and_bounded_simplex():
    first, second = ForecastTransition(50), ForecastTransition(50)
    for t in (first, second):
        t.step(0, request(0, 0), HORIZONS_MS)
    a = first.step(100_000_000, request(), HORIZONS_MS)
    for i in range(1, 11):
        b = second.step(i * 10_000_000, request(), HORIZONS_MS)
    assert a.weights == pytest.approx(b.weights)
    for i in range(11, 1011):
        b = second.step(i * 10_000_000, request(HORIZONS_MS[i % 4], .7), HORIZONS_MS)
        assert 0 <= b.alpha <= 1 and all(0 <= w <= 1 for w in b.weights)


@pytest.mark.parametrize("value", [True, -1, 1001, float("nan"), float("inf")])
def test_invalid_tau(value):
    with pytest.raises(ValueError):
        ForecastTransition(value)


@pytest.mark.parametrize("h,alpha", [(0, 1), (33, 0), (101, .5), (True, .5), (33, True)])
def test_invalid_requested_decision_does_not_advance(h, alpha):
    transition = ForecastTransition(50)
    with pytest.raises(ValueError):
        transition.step(0, request(h, alpha), HORIZONS_MS)
    assert transition.step(0, request(0, 0), HORIZONS_MS).alpha == 0


def test_reordering_rejected_without_mutating_state():
    transition = ForecastTransition(50)
    transition.step(0, request(0, 0), HORIZONS_MS)
    transition.step(100_000_000, request(), HORIZONS_MS)
    with pytest.raises(ValueError):
        transition.step(100_000_000, request(), HORIZONS_MS)
    assert transition.step(200_000_000, request(), HORIZONS_MS).alpha == pytest.approx(1 - exp(-4))
