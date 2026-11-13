from dataclasses import replace

import numpy as np
import pytest

from poseahead.core import PoseFrame, PoseSequence
from poseahead.evaluation.corruption import corruption_settings, degrade_observations


@pytest.fixture
def sequence():
    rotations = np.tile([0., 0., 0., 1.], (17, 1))
    return PoseSequence("stress", "test", tuple(
        PoseFrame(i, 7_000_000_000 + i * 10_000_000, 7_000_000_000 + i * 10_000_000,
                  np.zeros((17, 3)), np.ones(17), np.ones(17, bool),
                  root_position=[1, 2, 3], joint_rotations=rotations)
        for i in range(401)
    ))


def degraded(sequence, **options):
    return degrade_observations(sequence, **{
        "seed": 42, "noise_std_m": .025, "dropout_probability": 0., **options,
    })


def test_legacy_rng_and_metadata_are_exactly_preserved(sequence):
    actual = degraded(sequence, dropout_probability=.01)
    rng = np.random.default_rng(42)
    for original, frame in zip(sequence.frames, actual.frames, strict=True):
        noise = rng.normal(0, .025, (17, 3))
        noise[0] = 0
        validity = original.joint_validity & (rng.random(17) >= .01)
        positions = original.joint_positions.astype(np.float64) + noise
        positions[~validity] = 0
        np.testing.assert_array_equal(frame.joint_positions, positions.astype(np.float32))
        np.testing.assert_array_equal(frame.joint_validity, validity)
        assert np.all(frame.joint_confidence[~validity] == 0)
        assert frame.metadata["noise_profile"] == {
            "seed": 42, "std_m": .025, "dropout_probability": .01,
            "generator": "numpy_default_rng_PCG64",
        }


def test_outliers_are_reproducible_confident_root_relative_and_not_mutating(sequence):
    kwargs = {"outlier_probability": .1, "outlier_std_m": .2, "noise_std_m": 0.}
    first, again = degraded(sequence, **kwargs), degraded(sequence, **kwargs)
    different = degraded(sequence, seed=43, **kwargs)
    assert any(not np.array_equal(a.joint_positions, b.joint_positions)
               for a, b in zip(first.frames, different.frames, strict=True))
    assert any(np.any(f.joint_positions) for f in first.frames)
    for original, left, right in zip(sequence.frames, first.frames, again.frames, strict=True):
        np.testing.assert_array_equal(left.joint_positions, right.joint_positions)
        np.testing.assert_array_equal(left.joint_positions[0], 0)
        np.testing.assert_array_equal(left.root_position, original.root_position)
        np.testing.assert_array_equal(left.joint_confidence, original.joint_confidence)
        assert left.joint_validity.all()
        assert left.joint_rotations is None
        assert original.joint_rotations is not None
        assert not np.any(original.joint_positions)


def test_stride_preserves_source_timing_ids_and_rng_at_retained_anchors(sequence):
    full = degraded(sequence, dropout_probability=.01)
    sparse = degraded(sequence, dropout_probability=.01, observation_stride=6)
    assert [f.frame_id for f in sparse.frames] == list(range(0, 401, 6))
    for frame in sparse.frames:
        original = full.frames[frame.frame_id]
        assert frame.source_timestamp_ns == original.source_timestamp_ns
        assert frame.produced_timestamp_ns == original.produced_timestamp_ns
        np.testing.assert_array_equal(frame.joint_positions, original.joint_positions)
        np.testing.assert_array_equal(frame.joint_validity, original.joint_validity)


def test_gap_is_half_open_after_first_period_and_keeps_long_elapsed_time(sequence):
    frames = degraded(sequence, gap_period_ms=1000, gap_duration_ms=250).frames
    ids = {f.frame_id for f in frames}
    assert 0 in ids and 99 in ids and 125 in ids
    assert not ids.intersection(range(100, 125))
    assert 199 in ids and 225 in ids
    assert not ids.intersection(range(200, 225))
    assert all(f.source_timestamp_ns == sequence.frames[f.frame_id].source_timestamp_ns
               for f in frames)
    assert any(b.source_timestamp_ns - a.source_timestamp_ns == 260_000_000
               for a, b in zip(frames, frames[1:], strict=False))


def test_prefix_causality_and_combined_gap_stride_outliers(sequence):
    options = {"outlier_probability": .01, "outlier_std_m": .2,
               "gap_period_ms": 1000, "gap_duration_ms": 250, "observation_stride": 6}
    prefix = PoseSequence(sequence.sequence_id, sequence.clock_domain, sequence.frames[:237])
    short = degraded(prefix, **options)
    long = [f for f in degraded(sequence, **options).frames if f.frame_id < 237]
    assert len(short.frames) == len(long)
    for left, right in zip(short.frames, long, strict=True):
        np.testing.assert_array_equal(left.joint_positions, right.joint_positions)
        np.testing.assert_array_equal(left.joint_validity, right.joint_validity)
        assert left.metadata == right.metadata


def test_existing_invalid_joints_and_confidence_are_not_repaired(sequence):
    frame = sequence.frames[0]
    validity, confidence = frame.joint_validity.copy(), frame.joint_confidence.copy()
    validity[3], confidence[4] = False, 0.
    original = replace(frame, joint_validity=validity, joint_confidence=confidence)
    result = degraded(PoseSequence("one", "test", (original,)),
                      outlier_probability=1., outlier_std_m=.2).frames[0]
    assert not result.joint_validity[3] and result.joint_confidence[4] == 0
    np.testing.assert_array_equal(result.joint_positions[3], 0)


@pytest.mark.parametrize("options", [
    {"seed": True}, {"seed": -1}, {"noise_std_m": True}, {"noise_std_m": float("inf")},
    {"outlier_probability": .01}, {"outlier_std_m": .2},
    {"outlier_probability": 2, "outlier_std_m": .2},
    {"outlier_probability": .1, "outlier_std_m": float("nan")},
    {"gap_period_ms": 3000}, {"gap_duration_ms": 100},
    {"gap_period_ms": 1000, "gap_duration_ms": 1000},
    {"gap_period_ms": 1000, "gap_duration_ms": -1},
    {"gap_period_ms": 1000, "gap_duration_ms": 1.5},
    {"gap_period_ms": 2**64, "gap_duration_ms": 1},
    {"observation_stride": True}, {"observation_stride": 0}, {"observation_stride": 121},
])
def test_invalid_corruption_is_rejected(sequence, options):
    with pytest.raises(ValueError):
        degraded(sequence, **options)


def test_disabled_extensions_canonicalize_to_legacy_profile():
    assert corruption_settings({"observation_stride": 1, "gap_period_ms": 0}) == {
        "noise_std_m": 0., "dropout_probability": 0.,
    }


def test_fixed_outliers_have_declared_displacement_not_gaussian_sigma(sequence):
    frames = degraded(sequence, noise_std_m=0., outlier_probability=1.,
                      outlier_magnitude_m=.150).frames
    for frame in frames:
        np.testing.assert_allclose(np.linalg.norm(frame.joint_positions[1:], axis=1), .150,
                                   atol=2e-8)
        np.testing.assert_array_equal(frame.joint_positions[0], 0)
        assert frame.joint_validity.all() and np.all(frame.joint_confidence == 1)


def test_delivered_tracking_gap_keeps_arrivals_but_removes_pose_confidence_and_root(sequence):
    frames = degraded(sequence, tracking_gap_period_ms=1000, tracking_gap_duration_ms=100).frames
    assert len(frames) == len(sequence.frames)
    for i in range(100, 110):
        assert frames[i].frame_id == i
        assert not frames[i].joint_validity.any()
        assert not frames[i].joint_confidence.any()
        assert frames[i].root_position is None and frames[i].root_rotation is None
        assert frames[i].joint_rotations is None
        assert frames[i].metadata["noise_profile"]["tracking_gap_active"] is True
    assert frames[99].joint_validity.all() and frames[110].joint_validity.all()
    assert frames[110].root_position is not None


@pytest.mark.parametrize("options", [
    {"outlier_magnitude_m": .15},
    {"outlier_probability": .01, "outlier_magnitude_m": .15, "outlier_std_m": .2},
    {"outlier_probability": .01, "outlier_magnitude_m": float("nan")},
    {"tracking_gap_period_ms": 1000},
    {"tracking_gap_period_ms": 1000, "tracking_gap_duration_ms": 1000},
    {"tracking_gap_period_ms": 1000, "tracking_gap_duration_ms": True},
])
def test_final_spec_profile_invalid_modes_fail(sequence, options):
    with pytest.raises(ValueError):
        degraded(sequence, **options)
