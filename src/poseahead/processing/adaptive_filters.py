"""One-Euro and constant-velocity Kalman baselines over position observations."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from poseahead.core import PoseFrame
from poseahead.processing.filters import RawProcessor


def _positive(name, value, *, allow_zero=False):
    if type(value) not in (float, int) or not math.isfinite(value) or (
        value < 0 if allow_zero else value <= 0
    ):
        raise ValueError(f"invalid {name}")


def _observations(frame):
    values = np.zeros((18, 3), dtype=np.float64)
    values[:17] = frame.joint_positions
    valid = np.zeros(18, dtype=bool)
    valid[:17] = frame.joint_validity & (frame.joint_confidence > 0)
    if frame.root_position is not None:
        values[17] = frame.root_position
        valid[17] = True
    values[~valid] = 0
    return values, valid


def _result(frame, values, method, parameters):
    return replace(
        frame, joint_positions=values[:17],
        root_position=values[17] if frame.root_position is not None else None,
        root_rotation=None, joint_rotations=None,
        metadata={**frame.metadata, "processor": method, "processor_parameters": parameters,
                  "processing_time_semantics": "input_availability_compute_cost_unmodelled"},
    )


class OneEuroProcessor(RawProcessor):
    """Per-axis speed-adaptive low-pass filtering in metres and seconds.

    Derivative uses successive raw observations. See https://gery.casiez.net/1euro/.
    """

    method = "one_euro"

    def __init__(self, *, min_cutoff_hz=1.0, beta=0.0, derivative_cutoff_hz=1.0,
                 reset_gap_ms=250.0):
        for name, value in (("min_cutoff_hz", min_cutoff_hz), ("beta", beta),
                            ("derivative_cutoff_hz", derivative_cutoff_hz),
                            ("reset_gap_ms", reset_gap_ms)):
            _positive(name, value, allow_zero=name == "beta")
        self.parameters = dict(min_cutoff_hz=float(min_cutoff_hz), beta=float(beta),
                               derivative_cutoff_hz=float(derivative_cutoff_hz),
                               reset_gap_ms=float(reset_gap_ms))
        super().__init__()

    def reset(self):
        super().reset()
        self._raw = np.zeros((18, 3))
        self._filtered = np.zeros((18, 3))
        self._derivative = np.zeros((18, 3))
        self._valid = np.zeros(18, dtype=bool)

    @staticmethod
    def _alpha(cutoff, dt):
        return 1 / (1 + 1 / (2 * math.pi * cutoff * dt))

    def process(self, frame):
        self._validate(frame)
        values, valid = _observations(frame)
        filtered = values.copy()
        derivative = np.zeros_like(values)
        if self._previous is not None:
            dt = (frame.source_timestamp_ns - self._previous.source_timestamp_ns) / 1e9
            if dt * 1000 <= self.parameters["reset_gap_ms"]:
                active = valid & self._valid
                raw_derivative = (values[active] - self._raw[active]) / dt
                a = self._alpha(self.parameters["derivative_cutoff_hz"], dt)
                derivative[active] = a * raw_derivative + (1 - a) * self._derivative[active]
                cutoff = self.parameters["min_cutoff_hz"] + (
                    self.parameters["beta"] * np.abs(derivative[active])
                )
                a = self._alpha(cutoff, dt)
                filtered[active] = a * values[active] + (1 - a) * self._filtered[active]
        result = _result(frame, filtered, "one_euro", self.parameters)
        self._raw, self._filtered, self._derivative, self._valid = (
            values, filtered, derivative, valid
        )
        self._previous = result
        return result


class KalmanProcessor(RawProcessor):
    """Independent XYZ constant-velocity states, shared 2x2 covariance per joint.

    Process noise is continuous white acceleration, spectral density in m^2/s^3.
    Measurement sigma is metres. Confidence is a validity gate, not a covariance.
    """

    method = "kalman"

    def __init__(self, *, measurement_sigma_m=.025, acceleration_spectral_density=1.0,
                 initial_velocity_sigma_m_s=1.0, reset_gap_ms=250.0):
        for name, value in (("measurement_sigma_m", measurement_sigma_m),
                            ("acceleration_spectral_density", acceleration_spectral_density),
                            ("initial_velocity_sigma_m_s", initial_velocity_sigma_m_s),
                            ("reset_gap_ms", reset_gap_ms)):
            _positive(name, value)
        self.parameters = dict(measurement_sigma_m=float(measurement_sigma_m),
                               acceleration_spectral_density=float(acceleration_spectral_density),
                               initial_velocity_sigma_m_s=float(initial_velocity_sigma_m_s),
                               reset_gap_ms=float(reset_gap_ms))
        super().__init__()

    def reset(self):
        super().reset()
        self._positions = np.zeros((18, 3))
        self._velocity = np.zeros((18, 3))
        self._covariance = np.zeros((18, 2, 2))
        self._valid = np.zeros(18, dtype=bool)

    def process(self, frame):
        self._validate(frame)
        values, valid = _observations(frame)
        positions, velocity = values.copy(), np.zeros_like(values)
        r = self.parameters["measurement_sigma_m"] ** 2
        initial_p = np.diag([r, self.parameters["initial_velocity_sigma_m_s"] ** 2])
        covariance = np.broadcast_to(initial_p, (18, 2, 2)).copy()
        if self._previous is not None:
            dt = (frame.source_timestamp_ns - self._previous.source_timestamp_ns) / 1e9
            if dt * 1000 <= self.parameters["reset_gap_ms"]:
                active = valid & self._valid
                transition = np.array([[1, dt], [0, 1]])
                noise = self.parameters["acceleration_spectral_density"] * np.array([
                    [dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]
                ])
                p = transition @ self._covariance[active] @ transition.T + noise
                gain = p[:, :, 0] / (p[:, 0, 0, None] + r)
                predicted = self._positions[active] + dt * self._velocity[active]
                residual = values[active] - predicted
                positions[active] = predicted + gain[:, 0, None] * residual
                velocity[active] = self._velocity[active] + gain[:, 1, None] * residual
                # Joseph form preserves covariance symmetry/positive semidefiniteness.
                correction = np.broadcast_to(np.eye(2), p.shape).copy()
                correction[:, :, 0] -= gain
                covariance[active] = (
                    correction @ p @ correction.transpose(0, 2, 1)
                    + r * gain[:, :, None] * gain[:, None, :]
                )
        result = _result(frame, positions, "kalman", self.parameters)
        self._positions, self._velocity, self._covariance, self._valid = (
            positions, velocity, covariance, valid
        )
        self._previous = result
        return result

    def extrapolate(self, *, target_timestamp_ns: int) -> PoseFrame:
        """Predict a target without advancing or feeding predictions into filter state.

        PoseFrame timestamps stay attached to the latest observation. The distinct
        prediction target is metadata and must be used by an evaluator explicitly.
        """
        if self._previous is None:
            raise ValueError("extrapolation requires an observation")
        if type(target_timestamp_ns) is not int or not 0 <= target_timestamp_ns < 2**64:
            raise ValueError("target must be a uint64 timestamp")
        elapsed = target_timestamp_ns - self._previous.source_timestamp_ns
        if elapsed < 0 or elapsed > 150_000_000:
            raise ValueError("prediction horizon must be between 0 and 150 ms")
        values = self._positions + self._velocity * (elapsed / 1e9)
        return replace(_result(self._previous, values, "kalman_cv", self.parameters), metadata={
            **self._previous.metadata, "processor": "kalman_cv",
            "target_timestamp_ns": target_timestamp_ns, "target_offset_ns": elapsed,
        })


class KalmanExtrapolationProcessor(KalmanProcessor):
    method = "kalman_cv"

    def __init__(self, *, target_offset_ms=100.0, **parameters):
        _positive("target_offset_ms", target_offset_ms, allow_zero=True)
        if target_offset_ms > 150:
            raise ValueError("target_offset_ms must not exceed 150")
        self.target_offset_ns = round(target_offset_ms * 1e6)
        super().__init__(**parameters)

    def process(self, frame):
        # Reject overflow before mutating the Kalman state.
        if frame.source_timestamp_ns + self.target_offset_ns >= 2**64:
            raise ValueError("prediction target exceeds uint64 timestamp range")
        super().process(frame)
        return self.extrapolate(
            target_timestamp_ns=frame.source_timestamp_ns + self.target_offset_ns
        )
