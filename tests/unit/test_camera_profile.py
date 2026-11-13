"""The UI speed button must not discard the current camera calibration/settings."""

import json

from poseahead.config import LiveConfig, load_live_config
from poseahead.desktop.camera_profile import fast_camera_profile


def test_only_precision_changes_and_source_is_preserved(tmp_path):
    original = LiveConfig(
        camera_index=7, udp_port=5010, num_aug=2, detector_interval=6,
        confidence_threshold=.5,
        camera_intrinsics={"fx": 800, "fy": 810, "cx": 320, "cy": 240,
                           "width": 640, "height": 480},
    )
    source = tmp_path / "calibrated.json"
    source.write_text(json.dumps(original.to_dict()), encoding="utf-8")
    before = source.read_bytes()
    path = fast_camera_profile(source, tmp_path / "derived")
    fast = load_live_config(path).to_dict()
    assert fast.pop("crop_precision") == "fp16_conv"
    slow = original.to_dict()
    slow.pop("crop_precision")
    assert fast == slow
    assert source.read_bytes() == before
    assert fast_camera_profile(path, tmp_path / "derived") == path
    assert list((tmp_path / "derived").iterdir()) == [path]
