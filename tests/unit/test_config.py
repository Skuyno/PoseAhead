import pytest

from poseahead.config import LiveConfig, load_live_config


def test_default_config_loads() -> None:
    assert load_live_config("configs/live/default.json") == LiveConfig()


@pytest.mark.parametrize(
    "values",
    [
        {"num_aug": 0},
        {"num_aug": "1"},
        {"internal_batch_size": 0},
        {"internal_batch_size": 1, "num_aug": 2},
        {"detector_interval": 0},
        {"detector_interval": True},
        {"max_input_side": 32},
        {"max_input_side": True},
        {"default_fov_degrees": float("inf")},
        {"udp_port": 70000},
        {"confidence_threshold": float("nan")},
        {"confidence_threshold": True},
        {"confidence_threshold": False},
    ],
)
def test_bad_config_rejected(values: dict) -> None:
    with pytest.raises(ValueError):
        LiveConfig(**values)


def test_intrinsics_config_roundtrip():
    config = LiveConfig(camera_intrinsics={
        "fx": 500, "fy": 500, "cx": 320, "cy": 240, "width": 640, "height": 480,
    })
    assert config.camera_intrinsics.fx == 500
    assert LiveConfig(**config.to_dict()) == config
