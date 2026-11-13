"""Stage boundaries reconcile complete sends without adding unrelated percentiles."""

import pytest

from poseahead.runtime.pipeline_timing import PipelineTiming


def observe(timing, **overrides):
    values = dict(source_ns=1_000_000, inference_start_ns=3_000_000,
                  inference_end_ns=13_000_000, processed_ns=15_000_000,
                  send_start_ns=16_000_000, send_end_ns=18_000_000)
    timing.observe(**{**values, **overrides})


def test_single_frame_stage_sum_matches_exact_total():
    timing = PipelineTiming()
    observe(timing)
    report = timing.report()
    assert report["frame_age_before_inference_ms"]["median"] == 2
    assert report["backend_return_ms"]["median"] == 10
    assert report["recording_and_processing_ms"]["median"] == 2
    assert report["validity_bookkeeping_before_send_ms"]["median"] == 1
    assert report["packet_and_send_ms"]["median"] == 2
    assert report["capture_available_to_send_complete_ms"]["median"] == 17


def test_unsent_frame_is_not_a_zero_send_latency():
    timing = PipelineTiming()
    observe(timing, send_start_ns=None, send_end_ns=None)
    report = timing.report()
    assert report["backend_return_ms"]["samples"] == 1
    assert report["packet_and_send_ms"]["samples"] == 0
    assert report["packet_and_send_ms"]["median"] is None


@pytest.mark.parametrize("overrides", [
    {"source_ns": -1}, {"source_ns": True}, {"source_ns": 1.5},
    {"source_ns": 4_000_000}, {"inference_end_ns": 2_000_000},
    {"processed_ns": 12_000_000}, {"send_start_ns": None},
    {"send_end_ns": None}, {"send_end_ns": 14_000_000},
])
def test_invalid_boundaries_do_not_mutate_timing(overrides):
    timing = PipelineTiming()
    with pytest.raises(ValueError):
        observe(timing, **overrides)
    assert timing.report()["backend_return_ms"]["samples"] == 0


def test_empty_report_is_json_finite():
    import json

    json.dumps(PipelineTiming().report(), allow_nan=False)
