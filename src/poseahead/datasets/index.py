"""No-overwrite dataset-specific indexes, validated before window generation."""

import json
from pathlib import Path

from poseahead.artifacts import sha256_file
from poseahead.datasets.splits import CMU_SUBJECT_SPLITS, SUBJECT_SPLITS, sequence_split
from poseahead.recording import load_recording


def _entry(path):
    path = Path(path).resolve()
    digest = sha256_file(path)
    sequence = load_recording(path)
    if sha256_file(path) != digest:
        raise ValueError("recording changed while indexing")
    split, subject = sequence_split(sequence)
    if split == "unlabelled":
        raise ValueError("dataset index requires labelled supported recordings")
    entry = {"path": str(path), "sha256": digest, "sequence_id": sequence.sequence_id,
            "clock_domain": sequence.clock_domain, "subject": subject, "split": split,
            "frames": len(sequence.frames), "start_ns": sequence.frames[0].source_timestamp_ns,
            "end_ns": sequence.frames[-1].source_timestamp_ns}
    if subject.startswith("CMU:"):
        fingerprints = {f.metadata.get("skeleton_fingerprint") for f in sequence.frames}
        if (len(fingerprints) != 1 or not isinstance(next(iter(fingerprints)), str)
                or len(next(iter(fingerprints))) != 64):
            raise ValueError("CMU index requires a consistent skeleton fingerprint")
        entry["skeleton_fingerprint"] = next(iter(fingerprints))
    return entry


def _unique(entries):
    identities, hashes = set(), set()
    skeleton_splits = {}
    for entry in entries:
        identity = (entry["sequence_id"], entry["clock_domain"])
        if identity in identities or entry["sha256"] in hashes:
            raise ValueError("duplicate recording or sequence/clock in dataset index")
        identities.add(identity)
        hashes.add(entry["sha256"])
        fingerprint = entry.get("skeleton_fingerprint")
        if fingerprint is not None:
            if fingerprint in skeleton_splits and skeleton_splits[fingerprint] != entry["split"]:
                raise ValueError("duplicate skeleton calibration across dataset splits")
            skeleton_splits[fingerprint] = entry["split"]


def freeze_dataset(recordings, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("dataset index already exists")
    recordings = tuple(recordings)
    if not recordings:
        raise ValueError("dataset index requires at least one recording")
    entries = [_entry(path) for path in recordings]
    _unique(entries)
    is_cmu = {e["subject"].startswith("CMU:") for e in entries}
    if len(is_cmu) != 1:
        raise ValueError("mixed datasets require separate indexes")
    cmu = next(iter(is_cmu))
    manifest = {"format": "poseahead-cmu-index" if cmu else "poseahead-human36m-index",
                "version": 1,
                "subject_splits": dict(CMU_SUBJECT_SPLITS if cmu else SUBJECT_SPLITS),
                "recordings": entries}
    encoded = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    with output.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    return {"recordings": len(entries), "index": str(output.resolve()),
            "split_counts": {split: sum(e["split"] == split for e in entries)
                             for split in ("train", "validation", "test")}}


def verify_dataset(index_path):
    manifest = json.loads(Path(index_path).read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or set(manifest) != {
        "format", "version", "subject_splits", "recordings",
    } or manifest["format"] not in ("poseahead-human36m-index", "poseahead-cmu-index")
            or type(manifest["version"]) is not int or manifest["version"] != 1
            or manifest["subject_splits"] != dict(CMU_SUBJECT_SPLITS if
                manifest["format"] == "poseahead-cmu-index" else SUBJECT_SPLITS)):
        raise ValueError("invalid dataset index schema or split policy")
    entries = manifest["recordings"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("dataset index requires recordings")
    actual = []
    for entry in entries:
        if (not isinstance(entry, dict) or not isinstance(entry.get("path"), str)
                or not Path(entry["path"]).is_absolute()):
            raise ValueError("invalid dataset recording entry")
        current = _entry(entry["path"])
        if current["subject"] not in manifest["subject_splits"]:
            raise ValueError("recording dataset differs from index policy")
        if current != entry:
            raise ValueError("dataset recording checksum or metadata changed")
        actual.append(current)
    _unique(actual)
    return {"status": "verified", "recordings": len(actual),
            "split_counts": {split: sum(e["split"] == split for e in actual)
                             for split in ("train", "validation", "test")}}
