import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from poseahead.artifacts import sha256_file
from poseahead.core import PoseFrame, PoseSequence
from poseahead.core.metadata import metadata_to_json
from poseahead.evaluation.experiment import run_experiment, verify_experiment
from poseahead.prediction.policy import ThresholdPolicy
from poseahead.prediction.predictability import HORIZONS_MS, PredictabilityFeatures
from poseahead.prediction.transitions import ForecastTransition
from poseahead.processing.adaptive import AdaptiveProcessor
from poseahead.processing.adaptive_filters import KalmanExtrapolationProcessor
from poseahead.processing.config import create_processor, read_processor_config
from poseahead.recording import load_recording, save_recording
from poseahead.transport.processor_packet import processor_packet_fields
from poseahead.transport.udp_protocol import Algorithm, decode_udp_packet, encode_udp_packet


class LinearPredictor:
    history_frames = 3
    step_ns = 20_000_000

    def __init__(self, path):
        self.horizon_ms = int(Path(path).name)

    def predict(self, positions, times):
        return positions[-1] + (positions[-1] - positions[-2]) * (
            self.horizon_ms * 1_000_000 / (times[-1] - times[-2]))


@pytest.fixture
def bank_config(monkeypatch, tmp_path):
    monkeypatch.setattr("poseahead.processing.learned.FixedHorizonPredictor", LinearPredictor)
    checkpoints = {}
    for h in HORIZONS_MS:
        folder = tmp_path / str(h)
        folder.mkdir()
        (folder / "manifest.json").write_text("{}", encoding="utf-8")
        checkpoints[str(h)] = {"checkpoint": str(folder),
                               "checkpoint_sha256": sha256_file(folder / "manifest.json")}
    return {"checkpoints": checkpoints, "policy": {
        "version": 1, "error_limits_m": {str(h): .04 for h in HORIZONS_MS},
        "max_horizon_ms": 100, "minimum_matured": 3,
        "maximum_acceleration_m_s2": 20., "mode": "horizon_only",
    }}


def pose(i, *, ms=None):
    t = (i * 20 if ms is None else ms) * 1_000_000
    positions = np.zeros((17, 3))
    positions[1:, 0] = t / 1e9
    return PoseFrame(i, t, t + 5_000_000, positions, np.ones(17), np.ones(17, bool),
                     root_position=[1, 2, 3], root_rotation=[0, 0, 0, 1],
                     joint_rotations=np.tile([0, 0, 0, 1], (17, 1)),
                     metadata={"clock_domain": "synthetic"})


def process_many(p, count=30):
    return [p.process(pose(i)) for i in range(count)]


def test_bank_probes_all_horizons_raw_until_mature_then_selects_with_original_root(bank_config):
    p = AdaptiveProcessor(**bank_config)
    outputs = process_many(p)
    for i in (0, 1, 2, 3):
        assert outputs[i].metadata["target_offset_ns"] == 0
        np.testing.assert_array_equal(outputs[i].joint_positions, pose(i).joint_positions)
        assert outputs[i].root_rotation is not None
    last = outputs[-1]
    decision = last.metadata["adaptive_decision"]
    assert decision["horizon_ms"] == 100 and decision["alpha"] == 1
    assert decision["available_probe_horizons_ms"] == HORIZONS_MS
    assert decision["features"]["matured_counts"]["150"] > 0
    assert last.joint_positions[1, 0] == pytest.approx(.68)
    np.testing.assert_array_equal(last.root_position, [1, 2, 3])
    assert last.root_rotation is None and last.joint_rotations is None
    assert last.source_timestamp_ns == outputs[-1].source_timestamp_ns == 580_000_000
    fields = processor_packet_fields(p, last)
    packet = decode_udp_packet(encode_udp_packet(last, **fields, send_timestamp_ns=600_000_000))
    assert packet.algorithm == Algorithm.ADAPTIVE and packet.target_offset_ms == 100
    assert packet.predictability is None  # Margin is not a calibrated probability.
    assert processor_packet_fields(p, outputs[0])["algorithm"] == Algorithm.RAW


@pytest.mark.parametrize("loss", ["validity", "confidence", "gap"])
def test_loss_low_confidence_gap_reset_support_and_do_not_reuse_stale_bank(bank_config, loss):
    p = AdaptiveProcessor(**bank_config, minimum_confidence=.3)
    process_many(p)
    frame = pose(30)
    if loss == "validity":
        frame = replace(frame, joint_validity=np.zeros(17, bool))
    elif loss == "confidence":
        frame = replace(frame, joint_confidence=np.full(17, .2))
    else:
        frame = pose(30, ms=800)
    result = p.process(frame)
    assert result.metadata["target_offset_ns"] == 0
    np.testing.assert_array_equal(result.joint_positions, frame.joint_positions)
    assert result.root_rotation is not None
    f = result.metadata["adaptive_decision"]["features"]
    assert set(f["recent_observation_error_m"].values()) == {None}
    assert set(f["matured_counts"].values()) == {0}
    fresh = p.process(pose(31, ms=820 if loss == "gap" else 620))
    assert fresh.metadata["target_offset_ns"] == 0
    assert fresh.metadata["adaptive_decision"]["available_probe_horizons_ms"] == ()


def test_blend_keeps_nominal_forecast_target_not_fake_alpha_times_horizon(bank_config):
    bank_config["policy"]["mode"] = "horizon_blend"
    p = AdaptiveProcessor(**bank_config)
    for processor in p.bank.values():
        original = processor.predictor.predict

        def biased(positions, times, original=original):
            result = original(positions, times)
            result[1:, 0] += .02
            return result

        processor.predictor.predict = biased
    last = process_many(p)[-1]
    assert last.metadata["adaptive_decision"]["alpha"] == pytest.approx(.5, abs=1e-5)
    assert last.joint_positions[1, 0] == pytest.approx(.64, abs=1e-6)
    assert last.metadata["target_timestamp_ns"] == 680_000_000
    assert last.metadata["output_time_semantics"].startswith("partial_compensation")


def test_decisions_reconstruct_from_saved_features_and_resolved_policy(bank_config, tmp_path):
    p = AdaptiveProcessor(**bank_config)
    outputs = process_many(p)
    path = tmp_path / "processed.npz"
    save_recording(path, PoseSequence("adaptive", "synthetic", tuple(outputs)))
    restored = load_recording(path)
    policy = ThresholdPolicy(**p.parameters["policy"])
    for frame, original in zip(restored.frames, outputs, strict=True):
        d = frame.metadata["adaptive_decision"]
        values = d["features"]
        features = PredictabilityFeatures(
            values["source_timestamp_ns"], values["status"], values["minimum_confidence"],
            values["speed_m_s"], values["acceleration_m_s2"],
            tuple(values["recent_observation_error_m"][str(h)] for h in HORIZONS_MS),
            tuple(values["matured_counts"][str(h)] for h in HORIZONS_MS),
            values["pending_predictions"], values["discarded_predictions"],
        )
        decision = policy.decide(features, d["available_probe_horizons_ms"])
        assert decision.to_dict().items() <= metadata_to_json(d).items()
        assert frame.metadata == original.metadata


def test_future_suffix_cannot_change_prefix_and_reset_is_reproducible(bank_config):
    p = AdaptiveProcessor(**bank_config)
    first = process_many(p)
    p.reset()
    changed = [pose(i) if i < 15 else replace(pose(i), joint_positions=pose(i).joint_positions * 10)
               for i in range(30)]
    second = [p.process(f) for f in changed]
    for left, right in zip(first[:15], second[:15], strict=True):
        np.testing.assert_array_equal(left.joint_positions, right.joint_positions)
        assert left.metadata == right.metadata
    p.reset()
    assert [f.metadata for f in process_many(p)] == [f.metadata for f in first]


def test_model_failure_latches_bank_until_explicit_reset(bank_config):
    p = AdaptiveProcessor(**bank_config)
    p.process(pose(0))
    p.process(pose(1))
    predictor = p.bank[66].predictor
    original = predictor.predict

    def fail(*args):
        raise RuntimeError("inference failure")

    predictor.predict = fail
    with pytest.raises(RuntimeError, match="inference failure"):
        p.process(pose(2))
    predictor.predict = original
    with pytest.raises(RuntimeError, match="explicit reset"):
        p.process(pose(3))
    p.reset()
    assert process_many(p)[-1].metadata["target_offset_ns"] == 100_000_000


@pytest.mark.parametrize("kind", ["future", "order", "clock", "offset"])
def test_invalid_stream_requires_explicit_reset(bank_config, kind):
    p = AdaptiveProcessor(**bank_config)
    p.process(pose(0))
    f = pose(1)
    if kind == "future":
        f = replace(f, metadata={"target_timestamp_ns": 21_000_000})
    elif kind == "order":
        f = pose(0)
    elif kind == "offset":
        f = replace(f, metadata={"target_offset_ns": True})
    else:
        f = replace(f, metadata={"clock_domain": "other"})
    with pytest.raises(ValueError):
        p.process(f)
    with pytest.raises(RuntimeError, match="explicit reset"):
        p.process(pose(2))


def test_factory_and_paths_are_shared_and_input_config_not_mutated(bank_config, tmp_path):
    for h, entry in bank_config["checkpoints"].items():
        entry["checkpoint"] = h
    config = {"method": "adaptive", "parameters": bank_config}
    path = tmp_path / "adaptive.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    resolved = read_processor_config(path)
    assert config["parameters"]["checkpoints"]["33"]["checkpoint"] == "33"
    p = create_processor(resolved["method"], resolved["parameters"])
    assert process_many(p)[-1].metadata["prediction_status"] == "predicted"


@pytest.mark.parametrize("kind", ["missing", "hash", "horizon", "grid", "history"])
def test_bank_contracts_reject_unsafe_or_mismatched_checkpoints(bank_config, monkeypatch, kind):
    if kind == "missing":
        del bank_config["checkpoints"]["150"]
    elif kind == "hash":
        bank_config["checkpoints"]["33"]["checkpoint_sha256"] = "wrong"
    elif kind == "horizon":
        bank_config["checkpoints"]["33"] = bank_config["checkpoints"]["66"]
    elif kind == "history":
        bank_config["error_history"] = 2
    else:
        class BadGrid(LinearPredictor):
            def __init__(self, path):
                super().__init__(path)
                if self.horizon_ms == 150:
                    self.step_ns = 10_000_000

        monkeypatch.setattr("poseahead.processing.learned.FixedHorizonPredictor", BadGrid)
    with pytest.raises(ValueError):
        AdaptiveProcessor(**bank_config)


def test_target_overflow_rejected_before_advancing_state(bank_config):
    p = AdaptiveProcessor(**bank_config)
    f = replace(pose(0), source_timestamp_ns=2**64 - 1, produced_timestamp_ns=2**64 - 1)
    with pytest.raises(ValueError, match="timestamp range"):
        p.process(f)
    assert p.process(pose(0)).metadata["target_offset_ns"] == 0


@pytest.mark.parametrize("mode", ["horizon_only", "horizon_blend"])
@pytest.mark.parametrize("options", [{}, {"dynamics_window_ms": 100,
                                       "transition_time_constant_ms": 80},
                                      {"fallback_method": "kalman_cv",
                                       "maximum_neural_weight": .25,
                                       "neural_to_baseline_error_ratio": 1.,
                                       "transition_time_constant_ms": 80}])
def test_experiment_archives_decisions_and_replays_resolved_config(
    bank_config, tmp_path, mode, options,
):
    bank_config["policy"]["mode"] = mode
    bank_config.update(options)
    sequence = PoseSequence("linear", "synthetic", tuple(pose(i) for i in range(80)))
    save_recording(tmp_path / "source.npz", sequence)
    config = {"version": 1, "run_id": "adaptive-synthetic", "observations": "source.npz",
              "reference": "source.npz", "processor": {"method": "adaptive",
                                                          "parameters": bank_config},
              "seed": 42, "delay_ms": 100, "start_ns": 200_000_000,
              "end_ns": 1_580_000_000, "step_ns": 20_000_000, "window_frames": 20,
              "max_lag_ms": 100}
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    report = run_experiment(path, tmp_path / "run")
    assert verify_experiment(tmp_path / "run")["verified_files"] == 6
    assert report["adaptive_prediction"]["horizon_output_counts_ms"]["100"] > 0
    assert sum(report["adaptive_prediction"]["horizon_output_counts_ms"].values()) == 74
    assert report["adaptive_prediction"]["threshold_selection"].startswith("caller_supplied")
    expected_root = ("always_kalman_baseline_except_tracking_loss_raw"
                     if options.get("fallback_method") == "kalman_cv"
                     else "hold_latest_observation")
    assert report["learned_prediction"]["root_policy"] == expected_root
    again = run_experiment(tmp_path / "run/resolved_config.json", tmp_path / "rerun")
    assert report["display_error"] == again["display_error"]
    assert report["adaptive_prediction"] == again["adaptive_prediction"]
    a, b = [load_recording(tmp_path / name / "processed.npz") for name in ("run", "rerun")]
    assert [f.metadata for f in a.frames] == [f.metadata for f in b.frames]


def test_filtered_dynamics_transition_uses_current_probes_and_logs_nominal_mixture(bank_config):
    p = AdaptiveProcessor(**bank_config, dynamics_window_ms=100, transition_time_constant_ms=80)
    frames = process_many(p)
    assert frames[4].metadata["target_offset_ns"] == 0
    for frame in frames[10:]:
        d = frame.metadata["adaptive_decision"]
        assert 0 < d["alpha"] < 1 and d["horizon_ms"] == 100
        assert d["requested_decision"]["alpha"] == 1
        weights = d["forecast_weights"]
        expected = frame.source_timestamp_ns / 1e9 + sum(int(h) / 1000 * w
                                                        for h, w in weights.items())
        assert frame.joint_positions[1, 0] == pytest.approx(expected, abs=3e-7)
        assert frame.metadata["output_time_semantics"] == "mixed_current_anchor_forecast_weights"
        assert processor_packet_fields(p, frame)["target_offset_ms"] == 100


@pytest.mark.parametrize("loss", ["gap", "invalid", "confidence"])
def test_smoothed_mode_still_returns_exact_raw_immediately_on_tracking_loss(bank_config, loss):
    p = AdaptiveProcessor(**bank_config, dynamics_window_ms=100,
                          transition_time_constant_ms=80, minimum_confidence=.3)
    process_many(p)
    f = pose(30)
    if loss == "gap":
        f = pose(30, ms=800)
    elif loss == "invalid":
        f = replace(f, joint_validity=np.zeros(17, bool))
    else:
        f = replace(f, joint_confidence=np.full(17, .2))
    result = p.process(f)
    assert result.metadata["target_offset_ns"] == 0
    assert result.metadata["adaptive_decision"]["alpha"] == 0
    assert result.root_rotation is not None
    np.testing.assert_array_equal(result.joint_positions, f.joint_positions)


def test_smoothed_processor_prefix_reset_and_saved_weights_reproducible(bank_config, tmp_path):
    p = AdaptiveProcessor(**bank_config, dynamics_window_ms=100, transition_time_constant_ms=80)
    first = process_many(p)
    save_recording(tmp_path / "mix.npz", PoseSequence("mix", "synthetic", tuple(first)))
    restored = load_recording(tmp_path / "mix.npz")
    assert [f.metadata for f in restored.frames] == [f.metadata for f in first]
    policy = ThresholdPolicy(**p.parameters["policy"])
    transition = ForecastTransition(p.parameters["transition_time_constant_ms"])
    for frame in restored.frames:
        d = frame.metadata["adaptive_decision"]
        f = d["features"]
        features = PredictabilityFeatures(
            f["source_timestamp_ns"], f["status"], f["minimum_confidence"],
            f["speed_m_s"], f["acceleration_m_s2"],
            tuple(f["recent_observation_error_m"][str(h)] for h in HORIZONS_MS),
            tuple(f["matured_counts"][str(h)] for h in HORIZONS_MS),
            f["pending_predictions"], f["discarded_predictions"],
        )
        requested = policy.decide(features, d["available_probe_horizons_ms"])
        assert requested.to_dict() == metadata_to_json(d["requested_decision"])
        mix = transition.step(
            frame.source_timestamp_ns, requested, d["available_probe_horizons_ms"],
            hard_reset=features.status != "ready",
        )
        assert mix.horizon_ms == d["horizon_ms"] and mix.alpha == d["alpha"]
        assert mix.weights == tuple(d["forecast_weights"][str(h)] for h in HORIZONS_MS)
    p.reset()
    for i in range(30):
        f = pose(i) if i < 20 else replace(pose(i), joint_positions=pose(i).joint_positions * 3)
        current = p.process(f)
        if i < 20:
            assert current.metadata == first[i].metadata
            np.testing.assert_array_equal(current.joint_positions, first[i].joint_positions)


def test_soft_gate_release_keeps_current_anchor_forecasts_not_old_pose(bank_config, monkeypatch):
    from poseahead.prediction.policy import AdaptiveDecision

    p = AdaptiveProcessor(**bank_config, dynamics_window_ms=100, transition_time_constant_ms=80)
    before = process_many(p)[-1]
    monkeypatch.setattr(p.policy, "decide", lambda *args: AdaptiveDecision(0, 0., None, "gate"))
    current = p.process(pose(30))
    d = current.metadata["adaptive_decision"]
    assert d["requested_decision"]["alpha"] == 0
    assert 0 < d["alpha"] < before.metadata["adaptive_decision"]["alpha"]
    expected = .6 + sum(int(h) / 1000 * w for h, w in d["forecast_weights"].items())
    assert current.joint_positions[1, 0] == pytest.approx(expected, abs=3e-7)
    assert current.metadata["target_offset_ns"] == 100_000_000
    assert processor_packet_fields(p, current)["algorithm"] == Algorithm.ADAPTIVE


def test_kalman_only_fallback_exactly_matches_standalone_on_continuous_valid_input(bank_config):
    p = AdaptiveProcessor(**bank_config, fallback_method="kalman_cv", maximum_neural_weight=0)
    fixed = KalmanExtrapolationProcessor(target_offset_ms=100)
    for i in range(30):
        frame = replace(pose(i), root_position=[1 + i * .02, 2, 3])
        result, baseline = p.process(frame), fixed.process(frame)
        np.testing.assert_array_equal(result.joint_positions, baseline.joint_positions)
        np.testing.assert_array_equal(result.root_position, baseline.root_position)
        assert result.metadata["target_offset_ns"] == 100_000_000
        assert result.metadata["prediction_status"] == "baseline"
        assert processor_packet_fields(p, result)["algorithm"] == Algorithm.KALMAN_EXTRAPOLATION


def test_baseline_mixture_same_target_cap_paired_errors_and_root_policy(bank_config):
    p = AdaptiveProcessor(**bank_config, fallback_method="kalman_cv", maximum_neural_weight=.25,
                          neural_to_baseline_error_ratio=1., transition_time_constant_ms=80,
                          dynamics_window_ms=100)
    fixed = KalmanExtrapolationProcessor(target_offset_ms=100)
    for i in range(40):
        frame = replace(pose(i), root_position=[1 + i * .02, 2, 3])
        result, base = p.process(frame), fixed.process(frame)
        d = result.metadata["adaptive_decision"]
        alpha = d["alpha"]
        assert 0 <= alpha <= .25 and d["horizon_ms"] == d["baseline_horizon_ms"] == 100
        assert d["baseline_weight"] == 1 - alpha
        assert d["baseline_features"]["matured_counts"] == d["features"]["matured_counts"]
        np.testing.assert_array_equal(result.root_position, base.root_position)
        if alpha:
            expected = ((1 - alpha) * base.joint_positions.astype(np.float64)
                        + alpha * p.bank[100].predictor.predict(
                            np.stack([pose(i - 2).joint_positions, pose(i - 1).joint_positions,
                                      pose(i).joint_positions]),
                            ((i - 2) * 20_000_000, (i - 1) * 20_000_000, i * 20_000_000)))
            np.testing.assert_allclose(result.joint_positions, expected, atol=1e-7)
            assert result.metadata["output_time_semantics"] == "same_target_kalman_neural_mixture"
            assert processor_packet_fields(p, result)["algorithm"] == Algorithm.ADAPTIVE
    assert alpha > 0


def test_bad_neural_probes_never_replace_better_baseline(bank_config):
    bank_config["policy"]["error_limits_m"] = {str(h): .5 for h in HORIZONS_MS}
    p = AdaptiveProcessor(**bank_config, fallback_method="kalman_cv",
                          neural_to_baseline_error_ratio=1.)
    for processor in p.bank.values():
        original = processor.predictor.predict

        def bad(positions, times, original=original):
            output = original(positions, times)
            output[1:, 0] += .2
            return output

        processor.predictor.predict = bad
    outputs = process_many(p)
    assert all(f.metadata["adaptive_decision"]["alpha"] == 0 for f in outputs)
    d = outputs[-1].metadata["adaptive_decision"]
    assert d["available_probe_horizons_ms"] == HORIZONS_MS  # Still collects paired probes.
    assert d["policy_available_horizons_ms"] == ()
    assert d["baseline_features"]["matured_counts"]["100"] > 0


@pytest.mark.parametrize("loss", ["gap", "invalid", "confidence"])
def test_baseline_branch_tracking_loss_is_raw_and_clears_filter_support(bank_config, loss):
    p = AdaptiveProcessor(**bank_config, fallback_method="kalman_cv", minimum_confidence=.3,
                          transition_time_constant_ms=80, dynamics_window_ms=100)
    process_many(p)
    f = pose(30)
    if loss == "gap":
        f = pose(30, ms=800)
    elif loss == "invalid":
        f = replace(f, joint_validity=np.zeros(17, bool))
    else:
        f = replace(f, joint_confidence=np.full(17, .2))
    result = p.process(f)
    assert result.metadata["target_offset_ns"] == 0 and result.root_rotation is not None
    assert result.metadata["adaptive_decision"]["baseline_method"] == "raw"
    assert processor_packet_fields(p, result)["algorithm"] == Algorithm.RAW
    np.testing.assert_array_equal(result.joint_positions, f.joint_positions)
    d = result.metadata["adaptive_decision"]
    assert d["baseline_features"]["matured_counts"] == d["features"]["matured_counts"]
    assert set(d["features"]["matured_counts"].values()) == {0}


def test_baseline_prefix_and_reset_reproducibility(bank_config):
    p = AdaptiveProcessor(**bank_config, fallback_method="kalman_cv", maximum_neural_weight=.25,
                          neural_to_baseline_error_ratio=1., transition_time_constant_ms=80)
    first = process_many(p)
    p.reset()
    for i in range(30):
        f = pose(i) if i < 20 else replace(pose(i), joint_positions=pose(i).joint_positions * 3)
        output = p.process(f)
        if i < 20:
            assert output.metadata == first[i].metadata
            np.testing.assert_array_equal(output.joint_positions, first[i].joint_positions)


@pytest.mark.parametrize("missing_joint", [0, 3, 16])
def test_partial_recovery_preserves_usable_joints_root_and_clears_neural_weights(
    bank_config, missing_joint,
):
    p = AdaptiveProcessor(**bank_config, fallback_method="kalman_cv", maximum_neural_weight=0,
                          preserve_partial_baseline=True)
    fixed = KalmanExtrapolationProcessor(target_offset_ms=100)
    for i in range(45):
        frame = replace(pose(i), root_position=[1 + i * .02, 2, 3])
        if i in (20, 21, 30):
            validity, confidence = frame.joint_validity.copy(), frame.joint_confidence.copy()
            validity[missing_joint], confidence[missing_joint] = False, 0
            frame = replace(frame, joint_validity=validity, joint_confidence=confidence)
        result, base = p.process(frame), fixed.process(frame)
        np.testing.assert_array_equal(result.joint_positions, base.joint_positions)
        np.testing.assert_array_equal(result.joint_validity, frame.joint_validity)
        np.testing.assert_array_equal(result.root_position, base.root_position)
        assert result.metadata["prediction_status"] == "baseline"
        assert result.metadata["adaptive_decision"]["alpha"] == 0
        assert processor_packet_fields(p, result)["algorithm"] == Algorithm.KALMAN_EXTRAPOLATION


def test_partial_recovery_disables_neural_then_rewarms_without_global_filter_reset(bank_config):
    p = AdaptiveProcessor(**bank_config, fallback_method="kalman_cv", maximum_neural_weight=.25,
                          transition_time_constant_ms=80, preserve_partial_baseline=True)
    assert process_many(p)[-1].metadata["adaptive_decision"]["alpha"] > 0
    frame = pose(30)
    validity = frame.joint_validity.copy()
    validity[3] = False
    result = p.process(replace(frame, joint_validity=validity))
    d = result.metadata["adaptive_decision"]
    assert d["alpha"] == 0 and d["baseline_method"] == "kalman_cv"
    assert result.joint_validity[3] == 0 and result.metadata["target_offset_ns"] == 100_000_000
    assert all(n == 0 for n in d["features"]["matured_counts"].values())
    for i in range(31, 60):
        result = p.process(pose(i))
    assert result.metadata["adaptive_decision"]["alpha"] > 0


@pytest.mark.parametrize("loss", ["invalid", "confidence", "gap_after_partial"])
def test_partial_recovery_still_returns_raw_on_full_loss_or_any_arrival_gap(bank_config, loss):
    p = AdaptiveProcessor(**bank_config, fallback_method="kalman_cv", minimum_confidence=.3,
                          preserve_partial_baseline=True)
    process_many(p)
    frame = pose(30)
    if loss == "invalid":
        frame = replace(frame, joint_validity=np.zeros(17, bool))
    elif loss == "confidence":
        frame = replace(frame, joint_confidence=np.full(17, .2))
    else:
        validity = frame.joint_validity.copy()
        validity[3] = False
        p.process(replace(frame, joint_validity=validity))
        frame = pose(31, ms=1000)
    result = p.process(frame)
    assert result.metadata["target_offset_ns"] == 0
    assert result.metadata["adaptive_decision"]["alpha"] == 0
    assert processor_packet_fields(p, result)["algorithm"] == Algorithm.RAW
    np.testing.assert_array_equal(result.joint_positions, frame.joint_positions)


@pytest.mark.parametrize("options", [
    {"preserve_partial_baseline": True},
    {"fallback_method": "kalman_cv", "preserve_partial_baseline": 1},
])
def test_partial_recovery_options_are_explicit_and_baseline_only(bank_config, options):
    with pytest.raises(ValueError, match="preserve_partial_baseline"):
        AdaptiveProcessor(**bank_config, **options)


@pytest.mark.parametrize("options", [
    {"fallback_method": "ema"}, {"maximum_neural_weight": True, "fallback_method": "kalman_cv"},
    {"maximum_neural_weight": 1.1, "fallback_method": "kalman_cv"},
    {"neural_to_baseline_error_ratio": 0, "fallback_method": "kalman_cv"},
    {"neural_to_baseline_error_ratio": float("nan"), "fallback_method": "kalman_cv"},
    {"fallback_parameters": []}, {"maximum_neural_weight": .25},
    {"fallback_method": "kalman_cv", "fallback_parameters": {"target_offset_ms": 66}},
])
def test_invalid_baseline_mixture_options(bank_config, options):
    with pytest.raises(ValueError):
        AdaptiveProcessor(**bank_config, **options)
