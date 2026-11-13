"""Causal four-model bank with explicit partial-compensation output semantics."""

from dataclasses import replace
from string import hexdigits

import numpy as np

from poseahead.prediction.policy import ThresholdPolicy
from poseahead.prediction.predictability import HORIZONS_MS, CausalPredictabilityTracker
from poseahead.prediction.transitions import ForecastTransition
from poseahead.processing.adaptive_filters import KalmanExtrapolationProcessor
from poseahead.processing.learned import LearnedProcessor


class AdaptiveProcessor:
    method = "adaptive"

    def __init__(self, *, checkpoints, policy, history_sampling="linear_past",
                 max_observation_gap_ms=100, max_history_observations=4096,
                 history_prefilter_ms=0, minimum_confidence=0., error_history=8,
                 pending_per_horizon=64, dynamics_window_ms=0, transition_time_constant_ms=0,
                 fallback_method="raw", fallback_parameters=None, maximum_neural_weight=1.,
                 neural_to_baseline_error_ratio=None, preserve_partial_baseline=False):
        if not isinstance(policy, dict):
            raise ValueError("adaptive policy must be an explicit parameter object")
        self.policy = ThresholdPolicy(**policy)
        if fallback_method not in ("raw", "kalman_cv"):
            raise ValueError("fallback_method must be raw or kalman_cv")
        if type(preserve_partial_baseline) is not bool or (
            preserve_partial_baseline and fallback_method != "kalman_cv"
        ):
            raise ValueError("preserve_partial_baseline requires a boolean and kalman_cv fallback")
        self.preserve_partial_baseline = preserve_partial_baseline
        if (type(maximum_neural_weight) not in (int, float)
                or not np.isfinite(maximum_neural_weight) or not 0 <= maximum_neural_weight <= 1):
            raise ValueError("maximum_neural_weight must be finite in [0, 1]")
        if neural_to_baseline_error_ratio is not None and (
            type(neural_to_baseline_error_ratio) not in (int, float)
            or not np.isfinite(neural_to_baseline_error_ratio)
            or neural_to_baseline_error_ratio <= 0
        ):
            raise ValueError("neural_to_baseline_error_ratio must be finite and positive")
        if fallback_method == "raw" and (
            fallback_parameters is not None or maximum_neural_weight != 1
            or neural_to_baseline_error_ratio is not None
        ):
            raise ValueError("baseline mixture options require kalman_cv fallback")
        if fallback_parameters is not None and not isinstance(fallback_parameters, dict):
            raise ValueError("fallback_parameters must be a filter parameter object")
        if fallback_parameters and "target_offset_ms" in fallback_parameters:
            raise ValueError("baseline target is fixed by policy.max_horizon_ms")
        self.fallback_method = fallback_method
        self.maximum_neural_weight = maximum_neural_weight
        self.error_ratio = neural_to_baseline_error_ratio
        self.baseline = (KalmanExtrapolationProcessor(
            target_offset_ms=self.policy.max_horizon_ms, **(fallback_parameters or {}),
        ) if fallback_method == "kalman_cv" else None)
        self.baseline_tracker = (CausalPredictabilityTracker(
            max_gap_ms=max_observation_gap_ms, minimum_confidence=minimum_confidence,
            error_history=error_history, pending_per_horizon=pending_per_horizon,
        ) if self.baseline is not None else None)
        if type(error_history) is not int or self.policy.minimum_matured > error_history:
            raise ValueError("minimum_matured must not exceed error_history")
        self.tracker = CausalPredictabilityTracker(
            max_gap_ms=max_observation_gap_ms, minimum_confidence=minimum_confidence,
            error_history=error_history, pending_per_horizon=pending_per_horizon,
            dynamics_window_ms=dynamics_window_ms,
        )
        self.transition = ForecastTransition(transition_time_constant_ms)
        if (not isinstance(checkpoints, dict)
                or set(checkpoints) != {str(h) for h in HORIZONS_MS}):
            raise ValueError("adaptive bank requires exactly checkpoints 33, 66, 100, 150")
        for entry in checkpoints.values():
            if not isinstance(entry, dict) or set(entry) != {"checkpoint", "checkpoint_sha256"}:
                raise ValueError("each bank entry requires a checkpoint and pinned manifest hash")
            digest = entry["checkpoint_sha256"]
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(c not in hexdigits for c in digest)):
                raise ValueError("bank checkpoint_sha256 must be a SHA-256 hex digest")
        self.bank = {}
        for h in HORIZONS_MS:
            processor = LearnedProcessor(
                **checkpoints[str(h)], history_sampling=history_sampling,
                max_observation_gap_ms=max_observation_gap_ms,
                max_history_observations=max_history_observations,
                history_prefilter_ms=history_prefilter_ms,
            )
            if processor.predictor.horizon_ms != h:
                raise ValueError("checkpoint horizon disagrees with its bank key")
            self.bank[h] = processor
        grids = {(p.predictor.history_frames, p.predictor.step_ns) for p in self.bank.values()}
        if len(grids) != 1:
            raise ValueError("adaptive checkpoints must share their history sampling grid")
        self.history_sampling = history_sampling
        self.parameters = {
            "checkpoints": {str(h): {k: p.parameters[k] for k in (
                "checkpoint", "checkpoint_sha256")} for h, p in self.bank.items()},
            "policy": self.policy.parameters, "history_sampling": history_sampling,
            "max_observation_gap_ms": max_observation_gap_ms,
            "max_history_observations": max_history_observations,
            "history_prefilter_ms": history_prefilter_ms, "minimum_confidence": minimum_confidence,
            "error_history": error_history, "pending_per_horizon": pending_per_horizon,
        }
        # Preserve version-1 resolved configs/logs exactly when these options are off.
        if dynamics_window_ms:
            self.parameters["dynamics_window_ms"] = dynamics_window_ms
        if transition_time_constant_ms:
            self.parameters["transition_time_constant_ms"] = transition_time_constant_ms
        if self.baseline is not None:
            self.parameters.update(fallback_method=fallback_method,
                                   fallback_parameters=dict(self.baseline.parameters),
                                   maximum_neural_weight=maximum_neural_weight,
                                   neural_to_baseline_error_ratio=neural_to_baseline_error_ratio)
            if preserve_partial_baseline:
                self.parameters["preserve_partial_baseline"] = True
        self.reset()

    def reset(self):
        self.tracker.reset()
        self.transition.reset()
        for processor in self.bank.values():
            processor.reset()
        if self.baseline is not None:
            self.baseline.reset()
            self.baseline_tracker.reset()
        self._failed = False
        self._baseline_last_source_ns = None

    def process(self, frame):
        if self._failed:
            raise RuntimeError("adaptive bank failed; explicit reset required before reuse")
        if frame.source_timestamp_ns + max(HORIZONS_MS) * 1_000_000 >= 2**64:
            raise ValueError("adaptive forecast target exceeds timestamp range")
        try:
            return self._process(frame)
        except Exception:
            # One model can fail after others have advanced. Never silently reuse
            # a partially advanced bank or publish a fake successful fallback.
            self._failed = True
            raise

    def _process(self, frame):
        features = self.tracker.observe(frame)
        usable = features.status != "invalid_or_low_confidence"
        bank_input = frame if usable else replace(frame, joint_validity=np.zeros(17, bool))
        base, baseline_features = None, None
        if self.baseline is not None:
            baseline_features = self.baseline_tracker.observe(frame)
            baseline_usable = usable
            baseline_input = frame
            baseline_gap = features.status == "tracking_gap"
            if self.preserve_partial_baseline:
                mask = (frame.joint_validity & (frame.joint_confidence > 0)
                        & (frame.joint_confidence >= self.tracker.minimum_confidence))
                baseline_usable = bool(mask.any())
                baseline_input = replace(frame, joint_validity=mask)
                baseline_gap = (self._baseline_last_source_ns is not None
                                and frame.source_timestamp_ns - self._baseline_last_source_ns
                                > self.tracker.max_gap_ns)
            if not baseline_usable or baseline_gap:
                self.baseline.reset()
            if baseline_usable:
                base = self.baseline.process(baseline_input)
                if baseline_gap:
                    base = None  # Reinitialize from this anchor, but output Raw on the gap.
            self._baseline_last_source_ns = frame.source_timestamp_ns
        # Every model sees every anchor, even when the policy returns Raw. Probes
        # are counterfactual model forecasts, not necessarily displayed forecasts.
        forecasts = {h: p.process(bank_input) for h, p in self.bank.items()}
        available = tuple(h for h, p in forecasts.items()
                          if usable and p.metadata["prediction_status"] == "predicted")
        policy_available = available
        if self.baseline is not None:
            # Blend only estimates of the SAME future instant. Shorter forecasts
            # remain probes; this branch adapts strength, not the common target.
            h = self.policy.max_horizon_ms
            policy_available = (h,) if h in available else ()
            if self.error_ratio is not None and policy_available:
                i = HORIZONS_MS.index(h)
                neural_error, baseline_error = features.recent_error_m[i], (
                    baseline_features.recent_error_m[i])
                if (neural_error is None or baseline_error is None
                        or baseline_features.matured_counts[i] < self.policy.minimum_matured
                        or neural_error > baseline_error * self.error_ratio):
                    policy_available = ()
        decision = self.policy.decide(features, policy_available)
        if self.baseline is not None:
            alpha = min(decision.alpha, self.maximum_neural_weight)
            decision = replace(decision, alpha=alpha,
                               horizon_ms=decision.horizon_ms if alpha else 0,
                               reason="neural_weight_disabled" if self.maximum_neural_weight == 0
                               else decision.reason)
        mix = self.transition.step(frame.source_timestamp_ns, decision, available,
                                   hard_reset=features.status != "ready")
        for h in available:
            self.tracker.record_prediction(forecasts[h])
            if base is not None:
                self.baseline_tracker.record_prediction(self.baseline.extrapolate(
                    target_timestamp_ns=frame.source_timestamp_ns + h * 1_000_000))
        if self.baseline is not None:
            return self._baseline_result(frame, features, baseline_features, base, forecasts,
                                         available, policy_available, decision, mix)
        offset = mix.horizon_ms * 1_000_000
        logged_decision = decision.to_dict()
        if self.transition.tau_ns:
            logged_decision.update(
                horizon_ms=mix.horizon_ms, alpha=mix.alpha,
                requested_decision=decision.to_dict(),
                forecast_weights=dict(zip(map(str, HORIZONS_MS), mix.weights, strict=True)),
                transition_semantics="current_anchor_forecast_weights_not_old_pose_smoothing",
            )
        metadata = {
            **frame.metadata, "processor": self.method, "processor_parameters": self.parameters,
            "prediction_status": "predicted" if mix.alpha > 0 else decision.reason,
            "target_offset_ns": offset, "target_timestamp_ns": frame.source_timestamp_ns + offset,
            "root_prediction_policy": "hold_latest_observation",
            "processing_time_semantics": "input_availability_compute_cost_unmodelled",
            "adaptive_decision": {**logged_decision, "policy_version": 1,
                                  "mode": self.policy.mode,
                                  "available_probe_horizons_ms": available,
                                  "features": features.to_dict(),
                                  "feature_snapshot": "before_current_anchor_probe_registration"},
            "output_time_semantics": ("observed_pose" if not offset else
                                      "mixed_current_anchor_forecast_weights"
                                      if self.transition.tau_ns else
                                      "single_future_forecast" if mix.alpha == 1 else
                                      "partial_compensation_observation_plus_future_forecast"),
        }
        if not offset:
            return replace(frame, metadata=metadata)  # Exact raw data, including rotations.
        positions = (1 - mix.alpha) * frame.joint_positions.astype(np.float64)
        for h, weight in zip(HORIZONS_MS, mix.weights, strict=True):
            if weight:
                positions += weight * forecasts[h].joint_positions
        # No inferred endpoint orientation, root extrapolation or quaternion averaging.
        return replace(frame, joint_positions=positions, root_rotation=None,
                       joint_rotations=None, metadata=metadata)

    def _baseline_result(self, frame, features, baseline_features, base, forecasts,
                         available, policy_available, decision, mix):
        usable = base is not None and features.status != "tracking_gap"
        h = self.policy.max_horizon_ms if usable else 0
        alpha = mix.alpha if usable else 0.
        metadata = {
            **frame.metadata, "processor": self.method, "processor_parameters": self.parameters,
            "prediction_status": ("predicted" if alpha else "baseline" if usable
                                  else decision.reason),
            "target_offset_ns": h * 1_000_000,
            "target_timestamp_ns": frame.source_timestamp_ns + h * 1_000_000,
            "root_prediction_policy": "kalman_cv_baseline" if usable else "hold_latest_observation",
            "processing_time_semantics": "input_availability_compute_cost_unmodelled",
            "output_time_semantics": ("same_target_kalman_neural_mixture" if alpha else
                                      "single_future_baseline_forecast" if usable
                                      else "observed_pose"),
            "adaptive_decision": {
                **decision.to_dict(), "horizon_ms": h, "alpha": alpha,
                "requested_decision": decision.to_dict(), "policy_version": 1,
                "mode": self.policy.mode, "available_probe_horizons_ms": available,
                "policy_available_horizons_ms": policy_available,
                "forecast_weights": dict(zip(map(str, HORIZONS_MS), mix.weights, strict=True)),
                "baseline_method": "kalman_cv" if usable else "raw",
                "baseline_horizon_ms": h, "baseline_weight": 1 - alpha if usable else 0.,
                "baseline_features": baseline_features.to_dict(), "features": features.to_dict(),
                "baseline_error_scope": "paired_same_anchor_targets_raw_observation_consistency",
                "feature_snapshot": "before_current_anchor_probe_registration",
            },
        }
        if not usable:
            return replace(frame, metadata=metadata)
        positions = base.joint_positions.astype(np.float64)
        if alpha:
            positions = ((1 - alpha) * positions
                         + alpha * forecasts[h].joint_positions.astype(np.float64))
        return replace(base, joint_positions=positions, metadata=metadata)
