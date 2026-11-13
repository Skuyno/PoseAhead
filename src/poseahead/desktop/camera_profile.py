"""Derive a fast camera profile without overwriting calibration or user config."""

import json
import uuid
from pathlib import Path

from poseahead.config import load_live_config


def fast_camera_profile(source, directory):
    source = Path(source).resolve()
    config = load_live_config(source)
    if config.crop_precision == "fp16_conv":
        return source
    values = {**config.to_dict(), "crop_precision": "fp16_conv"}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("fp16_conv_" + uuid.uuid4().hex + ".json")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(values, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return path.resolve()
