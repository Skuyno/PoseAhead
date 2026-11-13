"""Map actual processor output to UDP identity, including learned fallback."""

from collections.abc import Mapping
from math import isclose, isfinite

from poseahead.transport.udp_protocol import Algorithm

_BASELINES = {"raw": Algorithm.RAW, "ema": Algorithm.EMA, "one_euro": Algorithm.ONE_EURO,
              "kalman": Algorithm.KALMAN, "kalman_cv": Algorithm.KALMAN_EXTRAPOLATION}
_PREDICTORS = {33: Algorithm.PREDICTOR_33_MS, 66: Algorithm.PREDICTOR_66_MS,
               100: Algorithm.PREDICTOR_100_MS, 150: Algorithm.PREDICTOR_150_MS}


def processor_packet_fields(processor, frame):
    offset = frame.metadata.get("target_offset_ns", 0)
    if type(offset) is not int or offset < 0 or offset % 1_000_000:
        raise ValueError("UDP requires a nonnegative whole-millisecond target offset")
    if "target_timestamp_ns" in frame.metadata and (
        frame.metadata["target_timestamp_ns"] != frame.source_timestamp_ns + offset
    ):
        raise ValueError("processor target time and offset disagree")
    if processor.method == "adaptive":
        decision = frame.metadata.get("adaptive_decision", {})
        alpha = decision.get("alpha")
        horizon = decision.get("horizon_ms")
        if (type(alpha) not in (int, float) or not 0 <= alpha <= 1
                or type(horizon) is not int or horizon * 1_000_000 != offset):
            raise ValueError("inconsistent adaptive decision/target")
        baseline_horizon = 0
        if decision.get("baseline_method") == "kalman_cv":
            weight = decision.get("baseline_weight")
            baseline_horizon = decision.get("baseline_horizon_ms")
            if (getattr(processor, "fallback_method", None) != "kalman_cv"
                    or type(baseline_horizon) is not int
                    or baseline_horizon != processor.policy.max_horizon_ms
                    or horizon != baseline_horizon
                    or alpha > processor.maximum_neural_weight + 1e-12
                    or type(weight) not in (int, float) or not 0 <= weight <= 1
                    or not isclose(weight, 1 - alpha, rel_tol=1e-9, abs_tol=1e-12)):
                raise ValueError("inconsistent adaptive Kalman baseline")
            if "forecast_weights" not in decision:
                raise ValueError("adaptive Kalman mixture requires explicit forecast weights")
        if "forecast_weights" in decision:
            weights = decision["forecast_weights"]
            if (not isinstance(weights, Mapping) or set(weights) != {str(h) for h in _PREDICTORS}
                    or any(type(w) not in (int, float) or not isfinite(w) or not 0 <= w <= 1
                           for w in weights.values())
                    or not isclose(sum(weights.values()), alpha, rel_tol=1e-9, abs_tol=1e-12)
                    or max([baseline_horizon] + [int(h) for h, w in weights.items()
                                                  if w > 0]) != horizon
                    or baseline_horizon and any(int(h) != horizon and w > 0
                                                for h, w in weights.items())):
                raise ValueError("adaptive forecast weights and nominal target disagree")
        if alpha == 0 and baseline_horizon:
            if frame.metadata.get("prediction_status") != "baseline":
                raise ValueError("adaptive Kalman-only output needs baseline status")
            algorithm = Algorithm.KALMAN_EXTRAPOLATION
        elif alpha == 0:
            if offset != 0 or frame.metadata.get("prediction_status") == "predicted":
                raise ValueError("adaptive raw fallback cannot claim a future prediction")
            algorithm = Algorithm.RAW
        else:
            if (horizon not in _PREDICTORS or horizon > processor.policy.max_horizon_ms
                    or frame.metadata.get("prediction_status") != "predicted"):
                raise ValueError("unsupported or inconsistent adaptive horizon")
            algorithm = Algorithm.ADAPTIVE
        # Policy margin is not a calibrated probability: do not send it as predictability.
    elif processor.method == "learned":
        if frame.metadata.get("prediction_status") == "predicted":
            if offset != processor.target_offset_ns or offset // 1_000_000 not in _PREDICTORS:
                raise ValueError("unsupported or inconsistent learned horizon")
            algorithm = _PREDICTORS[offset // 1_000_000]
        else:
            if offset != 0:
                raise ValueError("learned fallback cannot claim a future target")
            algorithm = Algorithm.RAW
    else:
        algorithm = _BASELINES[processor.method]
    return {"algorithm": algorithm, "target_offset_ms": offset // 1_000_000}
