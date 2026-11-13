from dataclasses import replace

import numpy as np
import pytest

from poseahead.evaluation.temporal import MotionSeries, score_effective_latency, score_jitter


def motion(times, values, valid=None):
    positions = np.zeros((len(times), 1, 3))
    positions[:, 0, 0] = values
    return MotionSeries(tuple(round(t * 1e9) for t in times), positions,
                        np.ones((len(times), 1), bool) if valid is None else valid, "synthetic")


def signal(t):
    return (np.exp(-((t - 1.2) / .13) ** 2) - .7 * np.exp(-((t - 2.1) / .21) ** 2)
            + .4 * np.exp(-((t - 3.2) / .1) ** 2))


@pytest.mark.parametrize("delay", [-.066, 0, .033, .100, .150])
def test_known_delay_sign_and_subframe_magnitude(delay):
    times = np.arange(0, 4, .01)
    report = score_effective_latency(motion(times, signal(times - delay)),
                                     motion(times, signal(times)))
    assert report["coverage"] == 1
    assert report["effective_latency_ms"] == pytest.approx(delay * 1000, abs=.3)


def test_jitter_uses_irregular_physical_time_and_does_not_reward_frozen_output():
    times = np.array([0, .01, .025, .040, .10])
    reference = motion(times, times**2)
    same = score_jitter(reference, reference)
    assert same["acceleration_error_m_s2"] == pytest.approx(0)
    assert same["output_acceleration_m_s2"] == pytest.approx(2)
    frozen = score_jitter(motion(times, times * 0), reference)
    assert frozen["acceleration_error_m_s2"] == pytest.approx(2)
    assert frozen["output_acceleration_m_s2"] == 0


def test_jitter_excludes_invalid_triples():
    times = np.arange(5) * .01
    valid = np.ones((5, 1), bool)
    valid[2] = False
    report = score_jitter(motion(times, times, valid), motion(times, times))
    assert report["coverage"] == 0
    assert report["acceleration_error_m_s2"] is None


@pytest.mark.parametrize("values", [lambda t: t * 0, lambda t: t])
def test_static_and_constant_velocity_latency_are_unidentifiable(values):
    times = np.arange(0, 2, .01)
    series = motion(times, values(times))
    report = score_effective_latency(series, series)
    assert report["effective_latency_ms"] is None
    assert report["joint_windows"][0]["status"] == "static_or_constant_velocity"


def test_periodic_motion_has_ambiguous_lag():
    times = np.arange(0, 4, .01)
    series = motion(times, np.sin(2 * np.pi * times / .12))
    report = score_effective_latency(series, series)
    assert report["effective_latency_ms"] is None
    assert report["joint_windows"][0]["status"] == "ambiguous_peak"


def test_boundary_and_missing_windows_excluded():
    times = np.arange(0, 4, .01)
    reference = motion(times, signal(times))
    boundary = score_effective_latency(motion(times, signal(times - .2)), reference)
    assert boundary["joint_windows"][0]["status"] == "lag_boundary"
    valid = np.ones((len(times), 1), bool)
    valid[50] = False
    missing = score_effective_latency(motion(times, signal(times), valid), reference)
    assert missing["joint_windows"][0]["status"] == "invalid"


def test_bad_clock_nonuniform_grid_short_windows_and_immutable_arrays():
    times = np.arange(0, 1, .01)
    series = motion(times, signal(times))
    with pytest.raises(ValueError, match="clock"):
        score_jitter(series, replace(series, clock_domain="other"))
    with pytest.raises(ValueError):
        series.positions_m.setflags(write=True)
    irregular = motion([0, .01, .03], [0, 1, 2])
    with pytest.raises(ValueError, match="uniform"):
        score_effective_latency(irregular, irregular)
    with pytest.raises(ValueError, match="short"):
        score_effective_latency(series, series, max_lag_ms=500)


@pytest.mark.parametrize("parameter, value", [
    ("max_lag_ms", float("nan")), ("max_lag_ms", True),
    ("max_lag_ms", 1e308),
    ("min_correlation", 2), ("ambiguity_margin", 1),
    ("min_velocity_variation_m_s", -1),
])
def test_invalid_thresholds(parameter, value):
    times = np.arange(0, 1, .01)
    series = motion(times, signal(times))
    with pytest.raises(ValueError):
        score_effective_latency(series, series, **{parameter: value})


def test_uncorrelated_motion_rejected():
    times = np.arange(0, 4, .01)
    noisy = motion(times, np.random.default_rng(42).normal(size=len(times)))
    report = score_effective_latency(noisy, motion(times, signal(times)))
    assert report["coverage"] == 0
    assert report["joint_windows"][0]["status"] == "low_correlation"


def test_simulated_delay_is_measured_at_display_time_not_source_time():
    from poseahead.core import PoseFrame, PoseSequence
    from poseahead.evaluation.snapshots import snapshot_motion
    from poseahead.processing.config import create_processor
    from poseahead.processing.simulation import simulate_processing

    times = np.arange(0, 4, .01)
    frames = []
    for i, t in enumerate(times):
        positions = np.zeros((17, 3))
        positions[1:, 0] = signal(t)
        ns = round(t * 1e9)
        frames.append(PoseFrame(i, ns, ns, positions, np.ones(17), np.ones(17, bool)))
    sequence = PoseSequence("pulses", "synthetic", tuple(frames))
    queries = [f.source_timestamp_ns for f in frames]
    delayed = list(simulate_processing(sequence, create_processor("raw"), queries,
                                        delay_ns=100_000_000))
    immediate = list(simulate_processing(sequence, create_processor("raw"), queries))
    # Full-window validity requires excluding the explicitly unavailable warm-up.
    report = score_effective_latency(snapshot_motion(delayed[20:]),
                                     snapshot_motion(immediate[20:]))
    assert report["effective_latency_ms"] == pytest.approx(100, abs=.3)
    assert report["eligible_joint_windows"] == 16  # root-relative pelvis is static
    assert not snapshot_motion(delayed).validity[0].any()
