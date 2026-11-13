from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from poseahead.datasets.window_comparison import verify_same_validation_windows


def store():
    manifest = {"split": "validation", "horizon_ms": 100, "history_frames": 10,
                "step_ns": 20_000_000, "windows": 1,
                "sequences": [{"sequence_id": "clip", "clock_domain": "clock",
                               "subject": "CMU:13"}], "shards": [{"sequence_index": 0}]}
    arrays = {"target_timestamp_ns": np.array([100]), "history_positions_m": np.zeros((1, 3))}
    return SimpleNamespace(manifest=manifest, manifest_sha256="hash",
                           iter_shards=lambda: iter([arrays])), arrays


def test_exact_validation_check_and_changes():
    left, _ = store()
    right, arrays = store()
    assert verify_same_validation_windows(left, right)["windows"] == 1
    arrays["history_positions_m"][0, 0] = 1
    with pytest.raises(ValueError, match="numeric"):
        verify_same_validation_windows(left, right)


@pytest.mark.parametrize("field,value", [("split", "test"), ("step_ns", 50_000_000),
                                       ("windows", 2)])
def test_protocol_mismatch_rejected(field, value):
    left, _ = store()
    right = deepcopy(left)
    right.manifest[field] = value
    with pytest.raises(ValueError):
        verify_same_validation_windows(left, right)
