from types import SimpleNamespace

import numpy as np
import pytest

from poseahead.core import PoseFrame
from poseahead.transport.processor_packet import processor_packet_fields
from poseahead.transport.udp_protocol import Algorithm, decode_udp_packet, encode_udp_packet


def pose(metadata):
    return PoseFrame(0, 0, 1, np.zeros((17, 3)), np.ones(17), np.ones(17, bool), metadata=metadata)


@pytest.mark.parametrize("horizon, algorithm", [
    (33, Algorithm.PREDICTOR_33_MS), (66, Algorithm.PREDICTOR_66_MS),
    (100, Algorithm.PREDICTOR_100_MS), (150, Algorithm.PREDICTOR_150_MS),
])
def test_learned_udp_identity_and_fallback(horizon, algorithm):
    processor = SimpleNamespace(method="learned", target_offset_ns=horizon * 1_000_000)
    frame = pose({"prediction_status": "predicted", "target_offset_ns": processor.target_offset_ns,
                  "target_timestamp_ns": processor.target_offset_ns})
    fields = processor_packet_fields(processor, frame)
    packet = decode_udp_packet(encode_udp_packet(frame, **fields, send_timestamp_ns=2))
    assert packet.algorithm == algorithm
    assert packet.target_offset_ms == horizon
    fallback = pose({"prediction_status": "warmup", "target_offset_ns": 0,
                     "target_timestamp_ns": 0})
    fields = processor_packet_fields(processor, fallback)
    assert fields == {"algorithm": Algorithm.RAW, "target_offset_ms": 0}


@pytest.mark.parametrize("metadata", [
    {"prediction_status": "predicted", "target_offset_ns": 66_000_000},
    {"prediction_status": "warmup", "target_offset_ns": 33_000_000},
    {"prediction_status": "predicted", "target_offset_ns": 33_500_000},
    {"prediction_status": "predicted", "target_offset_ns": 33_000_000,
     "target_timestamp_ns": 34_000_000},
])
def test_inconsistent_prediction_metadata_is_not_sent(metadata):
    processor = SimpleNamespace(method="learned", target_offset_ns=33_000_000)
    with pytest.raises(ValueError):
        processor_packet_fields(processor, pose(metadata))


@pytest.mark.parametrize("alpha,horizon,status", [
    (0, 33, "warmup"), (0, 0, "predicted"), (.5, 0, "predicted"),
    (.5, 66, "warmup"), (.5, 150, "predicted"), (True, 33, "predicted"),
    (float("nan"), 33, "predicted"), (1.1, 33, "predicted"),
])
def test_inconsistent_adaptive_output_is_not_sent(alpha, horizon, status):
    processor = SimpleNamespace(method="adaptive", policy=SimpleNamespace(max_horizon_ms=100))
    with pytest.raises(ValueError):
        frame = pose({"prediction_status": status, "target_offset_ns": horizon * 1_000_000,
                      "target_timestamp_ns": horizon * 1_000_000,
                      "adaptive_decision": {"alpha": alpha, "horizon_ms": horizon}})
        processor_packet_fields(processor, frame)


@pytest.mark.parametrize("weights", [
    {"33": .5}, {"33": .5, "66": 0, "100": 0, "150": 0},
    {"33": 0, "66": 0, "100": .4, "150": 0},
    {"33": 0, "66": 0, "100": .5, "150": True},
])
def test_adaptive_mixture_rejects_wrong_weights_sum_horizon_or_schema(weights):
    processor = SimpleNamespace(method="adaptive", policy=SimpleNamespace(max_horizon_ms=100))
    frame = pose({"prediction_status": "predicted", "target_offset_ns": 100_000_000,
                  "target_timestamp_ns": 100_000_000,
                  "adaptive_decision": {"alpha": .5, "horizon_ms": 100,
                                        "forecast_weights": weights}})
    with pytest.raises(ValueError, match="weights"):
        processor_packet_fields(processor, frame)


@pytest.mark.parametrize("change", [
    {"baseline_weight": .9}, {"baseline_horizon_ms": 66}, {"horizon_ms": 66},
    {"alpha": .5, "baseline_weight": .5},
    {"forecast_weights": {"33": .2, "66": 0, "100": 0, "150": 0}},
])
def test_baseline_mixture_inconsistent_identity_weights_or_cap_rejected(change):
    processor = SimpleNamespace(method="adaptive", fallback_method="kalman_cv",
                                maximum_neural_weight=.25,
                                policy=SimpleNamespace(max_horizon_ms=100))
    d = {"alpha": .2, "horizon_ms": 100, "baseline_method": "kalman_cv",
         "baseline_horizon_ms": 100, "baseline_weight": .8,
         "forecast_weights": {"33": 0, "66": 0, "100": .2, "150": 0}, **change}
    with pytest.raises(ValueError):
        processor_packet_fields(processor, pose({"prediction_status": "predicted",
            "target_offset_ns": 100_000_000, "target_timestamp_ns": 100_000_000,
            "adaptive_decision": d}))
