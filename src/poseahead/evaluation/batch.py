"""Sequential experiment suites with split checks and descriptive macro summaries."""

import json
import re
from pathlib import Path

import numpy as np

from poseahead.artifacts import sha256_file
from poseahead.datasets.splits import sequence_split
from poseahead.evaluation.corruption import corruption_settings
from poseahead.evaluation.experiment import (
    load_experiment_config,
    run_experiment,
    verify_experiment,
)
from poseahead.recording import load_recording


def _comparison_key(config):
    # Keep noise, delay and metric protocol separate; never mix incomparable cells.
    fields = ("processor", "delay_ms", "step_ns",
              "window_frames", "max_lag_ms", "min_velocity_variation_m_s", "min_correlation",
              "ambiguity_margin")
    return {**{field: config[field] for field in fields}, **corruption_settings(config)}


def _summary(rows):
    groups = {}
    for row in rows:
        key = json.dumps(row["condition"], sort_keys=True)
        groups.setdefault(key, []).append(row)
    results = []
    for group in groups.values():
        metrics = {}
        for metric in ("display_mpjpe_mm", "root_error_mm", "jitter_m_s2", "latency_ms"):
            values = [row[metric] for row in group if row[metric] is not None]
            metrics[metric] = {
                "macro_mean": float(np.mean(values)) if values else None,
                "eligible_sequences": len(values), "total_sequences": len(group),
            }
        results.append({"condition": group[0]["condition"], "metrics": metrics,
                        "run_ids": [r["run_id"] for r in group]})
    return {"groups": results, "runs": rows,
            "aggregation": "equal_weight_sequence_means_excluding_nulls_with_counts",
            "inference": "descriptive_only_no_confidence_intervals_or_winner_selection"}


def run_batch(manifest_path, output_directory):
    """Run one split at a time; validate every entry before creating output.

    Manifest schema: version=1, batch_id, split, experiments=[relative config paths].
    Each condition accepts each reference sequence/clock once. Repeated crops/seeds
    must not inflate its sequence sample count. No auto-selection uses test data.
    """
    manifest_path, output_directory = Path(manifest_path), Path(output_directory)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest) != {
        "version", "batch_id", "split", "experiments",
    }:
        raise ValueError("invalid batch manifest fields")
    if type(manifest["version"]) is not int or manifest["version"] != 1:
        raise ValueError("unsupported batch version")
    if not isinstance(manifest["batch_id"], str) or not manifest["batch_id"].strip():
        raise ValueError("batch_id must be nonempty")
    if manifest["split"] not in ("train", "validation", "test", "unlabelled"):
        raise ValueError("unsupported batch split")
    entries = manifest["experiments"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= 500:
        raise ValueError("batch requires 1 to 500 experiment configs")
    if output_directory.exists():
        raise FileExistsError("batch output directory already exists")
    prepared, seen, run_ids = [], set(), set()
    for entry in entries:
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError("experiment entry must be a config path")
        config = load_experiment_config(manifest_path.parent / entry)
        if config["run_id"] in run_ids:
            raise ValueError("duplicate run_id in batch")
        run_ids.add(config["run_id"])
        sequences = {}
        for field in ("reference", "observations"):
            digest = sha256_file(config[field])
            if config.get(f"{field}_sha256", digest) != digest:
                raise ValueError(f"{field} checksum mismatch")
            sequences[field] = load_recording(config[field])
            if sha256_file(config[field]) != digest:
                raise ValueError("input changed during batch preflight")
            config[f"{field}_sha256"] = digest
        reference, observations = sequences["reference"], sequences["observations"]
        split, subject = sequence_split(reference)
        if split != manifest["split"]:
            raise ValueError("reference split differs from batch split")
        observed_split, observed_subject = sequence_split(observations)
        if (observed_split != "unlabelled"
                and (observed_split, observed_subject) != (split, subject)):
            raise ValueError("observation and reference subjects/splits differ")
        if reference.clock_domain != observations.clock_domain:
            raise ValueError("observation and reference clocks differ")
        available = {f.source_timestamp_ns for f in reference.frames}
        if any(t not in available for t in range(config["start_ns"], config["end_ns"] + 1,
                                                 config["step_ns"])):
            raise ValueError("reference does not cover the evaluation grid")
        condition = _comparison_key(config)
        identity = (reference.sequence_id, reference.clock_domain)
        key = (json.dumps(condition, sort_keys=True), identity)
        if key in seen:
            raise ValueError("duplicate reference sequence within a comparison condition")
        seen.add(key)
        prepared.append((config, condition, identity, subject))
    output_directory.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, (config, condition, identity, subject) in enumerate(prepared):
        name = f"run-{index:04d}"
        config_path = output_directory / f"{name}.json"
        with config_path.open("x", encoding="utf-8") as stream:
            json.dump(config, stream, indent=2, allow_nan=False)
        report = run_experiment(config_path, output_directory / name)
        verify_experiment(output_directory / name)
        rows.append({
            "run_id": config["run_id"], "directory": name, "condition": condition,
            "sequence_id": identity[0], "clock_domain": identity[1], "subject": subject,
            "display_mpjpe_mm": report["display_error"]["mpjpe_mm"],
            "root_error_mm": report["display_error"]["root_error_mm"],
            "jitter_m_s2": report["jitter"]["acceleration_error_m_s2"],
            "latency_ms": report["latency"]["effective_latency_ms"],
            "joint_coverage": report["display_error"]["joint_coverage"],
            "root_coverage": report["display_error"]["root_coverage"],
            "jitter_coverage": report["jitter"]["coverage"],
            "latency_coverage": report["latency"]["coverage"],
        })
    summary = {"batch_id": manifest["batch_id"], "split": manifest["split"], **_summary(rows)}
    with (output_directory / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    files = {p.relative_to(output_directory).as_posix(): sha256_file(p)
             for p in output_directory.rglob("*") if p.is_file()}
    with (output_directory / "batch_completion.json").open("x", encoding="utf-8") as stream:
        json.dump({"batch_id": manifest["batch_id"], "sha256": files}, stream, indent=2)
    return summary


def verify_batch(directory):
    """Check the suite and every child run without allowing manifest path traversal."""
    directory = Path(directory)
    manifest = json.loads((directory / "batch_completion.json").read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or set(manifest) != {"batch_id", "sha256"}
            or not isinstance(manifest["sha256"], dict)):
        raise ValueError("invalid batch completion manifest")
    hashes = manifest["sha256"]
    for name, digest in hashes.items():
        if not isinstance(name, str) or not re.fullmatch(
            r"summary\.json|run-\d{4}\.json|run-\d{4}/[a-z_]+\.(?:json|npz)", name,
        ):
            raise ValueError("invalid batch artifact path")
        path = directory / name
        if path.is_symlink() or path.parent.is_symlink() or not path.is_file():
            raise ValueError("batch artifact is absent or symlinked")
        if sha256_file(path) != digest:
            raise ValueError(f"batch checksum mismatch: {name}")
    if "summary.json" not in hashes:
        raise ValueError("batch summary missing from manifest")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if (not isinstance(summary, dict) or summary.get("batch_id") != manifest["batch_id"]
            or not isinstance(summary.get("runs"), list) or not summary["runs"]):
        raise ValueError("invalid batch summary")
    expected = {"summary.json"}
    seen = set()
    for row in summary["runs"]:
        if not isinstance(row, dict) or not isinstance(row.get("directory"), str):
            raise ValueError("invalid batch run row")
        name = row["directory"]
        if not re.fullmatch(r"run-\d{4}", name) or name in seen:
            raise ValueError("invalid or repeated child run directory")
        seen.add(name)
        verified = verify_experiment(directory / name)
        if verified["run_id"] != row.get("run_id"):
            raise ValueError("batch and child run IDs disagree")
        child = json.loads((directory / name / "completion.json").read_text(encoding="utf-8"))
        expected.update(f"{name}/{p}" for p in child["sha256"])
        expected.update((f"{name}/completion.json", f"{name}.json"))
    if set(hashes) != expected:
        raise ValueError("batch manifest does not cover exactly the declared child artifacts")
    return {"batch_id": manifest["batch_id"], "verified_runs": len(seen),
            "verified_files": len(hashes), "status": "verified"}
