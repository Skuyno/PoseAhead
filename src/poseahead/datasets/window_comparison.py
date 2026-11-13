"""Exact equality checks for matched validation-window comparisons."""

from itertools import zip_longest

import numpy as np


def verify_same_validation_windows(left, right):
    """Ignore archive/path metadata, but compare every numeric input and target."""
    for store in (left, right):
        if store.manifest["split"] != "validation":
            raise ValueError("matched validation check rejects other splits")
    for field in ("horizon_ms", "history_frames", "step_ns", "windows"):
        if left.manifest[field] != right.manifest[field]:
            raise ValueError(f"validation setting differs: {field}")
    def identities(store):
        return [(s["sequence_id"], s["clock_domain"], s["subject"])
                for s in store.manifest["sequences"]]
    if identities(left) != identities(right):
        raise ValueError("validation source identities differ")
    if [s["sequence_index"] for s in left.manifest["shards"]] != [
        s["sequence_index"] for s in right.manifest["shards"]
    ]:
        raise ValueError("validation shard layout differs")
    count = 0
    for a, b in zip_longest(left.iter_shards(), right.iter_shards()):
        if a is None or b is None or a.keys() != b.keys():
            raise ValueError("validation shard schemas differ")
        for key in a:
            if not np.array_equal(a[key], b[key]):
                raise ValueError(f"validation numeric data differs: {key}")
        count += len(a["target_timestamp_ns"])
    return {"status": "identical_numeric_validation_windows", "windows": count,
            "left_manifest_sha256": left.manifest_sha256,
            "right_manifest_sha256": right.manifest_sha256}
