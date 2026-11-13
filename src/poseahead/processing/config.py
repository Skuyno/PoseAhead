"""Shared baseline construction for live and offline paths."""

import json
from pathlib import Path

from poseahead.processing.adaptive_filters import (
    KalmanExtrapolationProcessor,
    KalmanProcessor,
    OneEuroProcessor,
)
from poseahead.processing.filters import EmaProcessor, RawProcessor

_BASELINE_FACTORIES = {
    "raw": RawProcessor,
    "ema": EmaProcessor,
    "one_euro": OneEuroProcessor,
    "kalman": KalmanProcessor,
    "kalman_cv": KalmanExtrapolationProcessor,
}
METHODS = tuple(_BASELINE_FACTORIES)


def create_processor(method: str, parameters: dict | None = None):
    if not isinstance(method, str) or method not in (*METHODS, "learned", "adaptive"):
        raise ValueError("unknown processor method")
    if parameters is not None and not isinstance(parameters, dict):
        raise ValueError("processor parameters must be an object")
    if method == "adaptive":
        from poseahead.processing.adaptive import AdaptiveProcessor

        factory = AdaptiveProcessor
    elif method == "learned":
        from poseahead.processing.learned import LearnedProcessor

        factory = LearnedProcessor
    else:
        factory = _BASELINE_FACTORIES[method]
    try:
        return factory(**(parameters or {}))
    except TypeError as error:
        raise ValueError(f"invalid {method} parameters: {error}") from error


def load_processor(path: str | Path):
    config = read_processor_config(path)
    return create_processor(config["method"], config["parameters"])


def validate_processor_config(config):
    """Check the lightweight schema without loading any model or checkpoint."""
    if not isinstance(config, dict) or set(config) != {"method", "parameters"}:
        raise ValueError("processor config requires exactly method and parameters")
    if not isinstance(config["parameters"], dict):
        raise ValueError("processor parameters must be an object")
    if (not isinstance(config["method"], str)
            or config["method"] not in (*METHODS, "learned", "adaptive")):
        raise ValueError("unknown processor method")
    if config["method"] == "adaptive" and not all(
        isinstance(config["parameters"].get(key), dict) for key in ("checkpoints", "policy")
    ):
        raise ValueError("adaptive config requires checkpoints and policy objects")


def read_processor_config(path: str | Path):
    """Read/validate once and resolve checkpoint paths relative to their config file."""
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_processor_config(config)
    return {
        "method": config["method"],
        "parameters": resolve_processor_paths(
            config["method"],
            config["parameters"],
            path.parent,
        ),
    }


def resolve_processor_paths(method, parameters, base):
    parameters = dict(parameters)
    if method == "learned" and isinstance(parameters.get("checkpoint"), str):
        parameters["checkpoint"] = str((Path(base) / parameters["checkpoint"]).resolve())
    if method == "adaptive" and isinstance(parameters.get("checkpoints"), dict):
        parameters["checkpoints"] = {
            h: resolve_processor_paths("learned", entry, base) if isinstance(entry, dict) else entry
            for h, entry in parameters["checkpoints"].items()
        }
    return parameters
