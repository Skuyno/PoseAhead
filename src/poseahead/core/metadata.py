"""Immutable JSON metadata, with an explicit conversion for persistence."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


def freeze_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("metadata keys must be strings")
        return MappingProxyType({key: freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_metadata(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("metadata must contain only finite JSON-compatible values")


def metadata_to_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: metadata_to_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [metadata_to_json(item) for item in value]
    return value
