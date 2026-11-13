import json
from pathlib import Path

import pytest

from poseahead.processing.config import (
    METHODS,
    create_processor,
    load_processor,
    read_processor_config,
)


@pytest.mark.parametrize("method", METHODS)
def test_checked_in_processing_config(method):
    path = Path(__file__).resolve().parents[2] / "configs" / "processing" / f"{method}.json"
    processor = load_processor(path)
    assert processor.method == method


@pytest.mark.parametrize(
    "method, parameters",
    [
        ("unknown", {}),
        (True, {}),
        ([], {}),
        ("ema", []),
        ("raw", {"time_constant_ms": 50}),
        ("ema", {"time_constant_ms": 0}),
        ("one_euro", {"beta": -1}),
        ("one_euro", {"min_cutoff_hz": True}),
        ("kalman", {"measurement_sigma_m": float("nan")}),
        ("kalman_cv", {"target_offset_ms": 151}),
    ],
)
def test_invalid_processor_parameters(method, parameters):
    with pytest.raises(ValueError):
        create_processor(method, parameters)


@pytest.mark.parametrize(
    "config",
    [
        [],
        {},
        {"method": "raw"},
        {"method": "raw", "parameters": None},
        {"method": "raw", "parameters": {}, "typo": 1},
    ],
)
def test_invalid_config_schema(tmp_path, config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        load_processor(path)


def test_read_config_resolves_paths_without_loading_a_checkpoint(tmp_path):
    path = tmp_path / "learned.json"
    original = json.dumps({"method": "learned", "parameters": {"checkpoint": "missing-model"}})
    path.write_text(original, encoding="utf-8")
    values = read_processor_config(path)
    assert values["parameters"]["checkpoint"] == str((tmp_path / "missing-model").resolve())
    assert not (tmp_path / "missing-model").exists()
    assert path.read_text(encoding="utf-8") == original
