"""Precision storage preserves normalization; comparison is numerical, not accuracy."""

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from poseahead.config import LiveConfig, load_live_config
from poseahead.core import PoseFrame
from poseahead.perception.precision import set_convolution_storage


def comparison():
    script = Path(__file__).resolve().parents[2] / "scripts/benchmark_metrabs_precision.py"
    spec = importlib.util.spec_from_file_location("precision_probe", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compare_poses


def test_only_convolution_and_linear_storage_changes_on_cpu():
    torch = pytest.importorskip("torch")
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.BatchNorm2d(4),
                                    torch.nn.Flatten(), torch.nn.Linear(4, 2)).eval()
        original_bn = {k: v.clone() for k, v in model[1].state_dict().items()}
        assert set_convolution_storage(model, "fp16_conv", torch) == 2
        assert model[0].weight.dtype == model[3].weight.dtype == torch.float16
        assert model[1].weight.dtype == model[1].running_mean.dtype == torch.float32
        for key, value in model[1].state_dict().items():
            assert torch.equal(value, original_bn[key])
        with pytest.raises(ValueError, match="eval"):
            set_convolution_storage(model.train(), "fp16_conv", torch)


@pytest.mark.parametrize("value", ["fp16", "unknown", None, True, []])
def test_invalid_precision_does_not_load_a_model(value):
    with pytest.raises(ValueError, match="crop_precision"):
        LiveConfig(crop_precision=value)


def test_fast_profile_changes_no_other_camera_settings():
    from dataclasses import asdict

    slow, fast = LiveConfig(), load_live_config("configs/live/fp16_conv.json")
    assert fast.crop_precision == "fp16_conv"
    assert asdict(replace(fast, crop_precision="autocast")) == asdict(slow)


def frame():
    points = np.arange(51, dtype=np.float32).reshape(17, 3) * .03
    points[0] = 0
    return PoseFrame(0, 0, 1, points, np.ones(17), np.ones(17, dtype=bool),
                     root_position=np.array([.123, .445, 3.321]),
                     metadata={"detector_mode": "detected_current_frame"})


def test_identical_arrays_have_exact_zero_absolute_delta():
    pose = frame()
    result = comparison()([pose], [pose])
    assert result["arrays_exact"]
    assert result["absolute_joint_delta_mm"]["max"] == 0  # avoid subtractive cancellation


def test_comparison_detects_changed_presence_schedule_and_position():
    pose = frame()
    points = pose.joint_positions.copy()
    points[1, 0] += .01
    modified = replace(pose, joint_positions=points, metadata={"detector_mode": "reused_past_box"})
    result = comparison()([pose], [modified])
    assert not result["arrays_exact"] and not result["detection_schedule_match"]
    assert result["body_delta_mm"]["max"] == pytest.approx(10, abs=.001)
    assert not comparison()([pose], [None])["validity_and_presence_match"]
