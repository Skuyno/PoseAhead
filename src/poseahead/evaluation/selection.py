"""Validation-only robust comparator selection, not a three-metric victory claim."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from poseahead.artifacts import sha256_file
from poseahead.evaluation.batch import verify_batch
from poseahead.evaluation.corruption import corruption_settings
from poseahead.processing.config import validate_processor_config

SELECTION_RULE = {
    "maximum_clean_relative_growth": .10,
    "minimum_pose_coverage": .95,
    "score": "worst_corrupted_profile_max_error_or_jitter_ratio_to_baseline",
    "aggregation": "equal_sequence_and_noise_seed_weight",
    "tie_break": "baseline_then_declared_candidate_order",
    "latency": "report_only_when_all_masks_match_and_metrics_exist",
    "advantage_claim": "requires_separate_frozen_20ms_10percent_paired_comparison",
}
_METRICS = ("display_mpjpe_mm", "jitter_m_s2")


def validate_selection_protocol(protocol):
    fields = {"version", "protocol_id", "baseline_id", "clean_profile", "candidate_ids",
              "profile_ids", "seeds", "sequences", "rule"}
    if (not isinstance(protocol, dict) or set(protocol) != fields
            or type(protocol["version"]) is not int or protocol["version"] != 1
            or protocol["rule"] != SELECTION_RULE):
        raise ValueError("invalid robust-selection protocol/rule")
    for field in ("candidate_ids", "profile_ids", "sequences"):
        values = protocol[field]
        if (not isinstance(values, list) or not values
                or any(not isinstance(v, str) or not v.strip() for v in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"{field} must contain unique nonempty IDs")
    if (not isinstance(protocol["protocol_id"], str) or not protocol["protocol_id"].strip()
            or protocol["baseline_id"] not in protocol["candidate_ids"]
            or protocol["clean_profile"] not in protocol["profile_ids"]
            or len(protocol["profile_ids"]) < 2):
        raise ValueError("protocol needs a declared baseline, clean and corrupted profiles")
    seeds = protocol["seeds"]
    if (not isinstance(seeds, list) or not seeds
            or any(type(s) is not int or s < 0 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("noise seeds must be unique nonnegative integers")
    if (len(protocol["candidate_ids"]) * len(protocol["profile_ids"]) * len(seeds)
            * len(protocol["sequences"]) > 10000):
        raise ValueError("selection panel is too large")


def _finite(value, *, signed=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (signed or value >= 0))


def _ratio(candidate, baseline):
    # Perfect comparator: zero remains equal; positive error is not finite growth.
    value = candidate / baseline if baseline else (1. if not candidate else None)
    return value if value is None or math.isfinite(value) else None


def _mean(values):
    return sum(v / len(values) for v in values)


def select_validation_panel(protocol, rows):
    """Select a fixed comparator on a complete, matched validation panel only.

    Input fingerprints cover observations, reference, seed, evaluation grid and
    metric settings. Pose/latency masks must be digested from actual artifacts.
    Repeated noise seeds are averaged, never treated as independent subjects.
    """
    validate_selection_protocol(protocol)
    keys = ("candidate_id", "profile_id", "seed", "sequence_id")
    expected = set(itertools.product(protocol["candidate_ids"], protocol["profile_ids"],
                                     protocol["seeds"], protocol["sequences"]))
    cells = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("split") != "validation":
            raise ValueError("selection requires validation rows only; test is forbidden")
        try:
            key = tuple(row[k] for k in keys)
            if type(row["seed"]) is not int or key not in expected or key in cells:
                raise ValueError("unexpected or duplicate selection cell")
            for field in ("input_fingerprint", "pose_mask_sha256", "latency_mask_sha256"):
                value = row[field]
                if (not isinstance(value, str) or len(value) != 64
                        or any(c not in "0123456789abcdef" for c in value)):
                    raise ValueError("invalid selection fingerprint")
            for field in (*_METRICS, "latency_ms"):
                if row[field] is not None and not _finite(row[field], signed=field == "latency_ms"):
                    raise ValueError("nonfinite/invalid selection metric")
            for field in ("joint_coverage", "jitter_coverage", "latency_coverage"):
                if not _finite(row[field]) or row[field] > 1:
                    raise ValueError("invalid metric coverage")
        except (KeyError, TypeError) as error:
            raise ValueError("invalid selection cell fields") from error
        cells[key] = row
    if set(cells) != expected:
        raise ValueError("incomplete selection panel; do not rank missing conditions")
    baseline, clean = protocol["baseline_id"], protocol["clean_profile"]
    for (_candidate, profile, seed, sequence), row in cells.items():
        base = cells[baseline, profile, seed, sequence]
        if row["input_fingerprint"] != base["input_fingerprint"]:
            raise ValueError("candidate inputs or evaluation settings differ from baseline")

    candidates = []
    for candidate in protocol["candidate_ids"]:
        reasons, profiles = [], {}
        for profile in protocol["profile_ids"]:
            group = [cells[candidate, profile, seed, sequence]
                     for seed in protocol["seeds"] for sequence in protocol["sequences"]]
            bases = [cells[baseline, profile, seed, sequence]
                     for seed in protocol["seeds"] for sequence in protocol["sequences"]]
            for row, base in zip(group, bases, strict=True):
                if any(row[m] is None or base[m] is None for m in _METRICS):
                    reasons.append(f"{profile}:missing_pose_metric")
                if any(min(row[c], base[c]) < SELECTION_RULE["minimum_pose_coverage"]
                       for c in ("joint_coverage", "jitter_coverage")):
                    reasons.append(f"{profile}:insufficient_pose_coverage")
                if row["pose_mask_sha256"] != base["pose_mask_sha256"]:
                    reasons.append(f"{profile}:different_pose_masks")
            means = {m: (_mean([r[m] for r in group])
                         if all(r[m] is not None for r in group) else None) for m in _METRICS}
            base_means = {m: (_mean([r[m] for r in bases])
                              if all(r[m] is not None for r in bases) else None) for m in _METRICS}
            ratios = {m: (_ratio(means[m], base_means[m])
                          if means[m] is not None and base_means[m] is not None else None)
                      for m in _METRICS}
            if any(v is None for v in ratios.values()):
                reasons.append(f"{profile}:undefined_ratio")
            if profile == clean and any(
                means[m] is not None and base_means[m] is not None
                and means[m] > base_means[m] * (1 + SELECTION_RULE["maximum_clean_relative_growth"])
                for m in _METRICS
            ):
                reasons.append("clean:accuracy_or_jitter_budget_exceeded")
            latency_matched = all(
                r["latency_ms"] is not None and b["latency_ms"] is not None
                and r["latency_coverage"] > 0 and b["latency_coverage"] > 0
                and r["latency_mask_sha256"] == b["latency_mask_sha256"]
                for r, b in zip(group, bases, strict=True)
            )
            profiles[profile] = {
                "means": means, "baseline_means": base_means, "ratios": ratios,
                "cells": len(group),
                "latency_status": ("matched_descriptive_only" if latency_matched
                                   else "not_comparable"),
                "latency_mean_ms": (_mean([r["latency_ms"] for r in group])
                                    if latency_matched else None),
                "minimum_latency_coverage": min(r["latency_coverage"] for r in group),
            }
        reasons = sorted(set(reasons))
        score = (max(profiles[p]["ratios"][m] for p in protocol["profile_ids"] if p != clean
                     for m in _METRICS) if not reasons else None)
        candidates.append({"candidate_id": candidate, "eligible": not reasons,
                           "exclusions": reasons, "robust_score": score, "profiles": profiles})
    order = {name: index for index, name in enumerate(protocol["candidate_ids"])}
    ranking = sorted((c for c in candidates if c["eligible"]), key=lambda c: (
        c["robust_score"], c["candidate_id"] != baseline, order[c["candidate_id"]],
    ))
    return {
        "protocol_id": protocol["protocol_id"], "selection_split": "validation",
        "selected_candidate_id": ranking[0]["candidate_id"] if ranking else None,
        "status": "comparator_selected" if ranking else "no_eligible_comparator",
        "ranked_candidate_ids": [c["candidate_id"] for c in ranking],
        "rule": dict(SELECTION_RULE), "candidates": candidates,
        "advantage_claim": "not_assessed", "test_scored": False,
        "limitations": [
            "A selection objective is not proof of superiority in latency/accuracy/jitter.",
            "Noise seeds are not independent people; validation was repeatedly consulted.",
            "Latency with differing eligibility masks is not ranked or called an improvement.",
            "The report alone cannot prove prospective protocol chronology.",
        ],
    }


def _digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def select_from_batches(protocol_path, manifest_path):
    """Read pinned, completed batches without loading checkpoints or scoring test."""
    protocol_path, manifest_path = Path(protocol_path), Path(manifest_path)
    protocol_hash = sha256_file(protocol_path)
    manifest_hash = sha256_file(manifest_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_selection_protocol(protocol)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or set(manifest) != {
        "version", "protocol_sha256", "declaration", "declaration_sha256", "batches",
    } or type(manifest["version"]) is not int or manifest["version"] != 1
            or manifest["protocol_sha256"] != protocol_hash
            or not isinstance(manifest["batches"], list) or not manifest["batches"]):
        raise ValueError("invalid pinned selection manifest")
    if not isinstance(manifest["declaration"], str):
        raise ValueError("invalid study declaration path")
    declaration_path = (manifest_path.parent / manifest["declaration"]).resolve()
    if sha256_file(declaration_path) != manifest["declaration_sha256"]:
        raise ValueError("study declaration checksum mismatch")
    declaration = json.loads(declaration_path.read_text(encoding="utf-8"))
    if (not isinstance(declaration, dict) or set(declaration) != {
        "protocol_sha256", "candidate_processors", "profile_settings", "evaluation_settings",
        "sequence_inputs",
    } or declaration["protocol_sha256"] != protocol_hash):
        raise ValueError("invalid pinned study declaration")
    rows, provenance, seen = [], [], set()
    processors, profiles, inputs = {}, {}, {}
    common_evaluation = None
    for entry in manifest["batches"]:
        if not isinstance(entry, dict) or set(entry) != {"directory", "sha256", "cells"}:
            raise ValueError("invalid selection batch entry")
        if not isinstance(entry["directory"], str) or not isinstance(entry["cells"], dict):
            raise ValueError("invalid selection batch path/cells")
        directory = (manifest_path.parent / entry["directory"]).resolve()
        if directory in seen:
            raise ValueError("selection batch repeated")
        seen.add(directory)
        completion = directory / "batch_completion.json"
        if sha256_file(completion) != entry["sha256"]:
            raise ValueError("selection batch checksum mismatch")
        verify_batch(directory)
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        if summary["split"] != "validation":
            raise ValueError("selection batch must be validation, never test")
        if set(entry["cells"]) != {r["run_id"] for r in summary["runs"]}:
            raise ValueError("selection mapping must cover exactly every batch run")
        for row in summary["runs"]:
            labels = entry["cells"][row["run_id"]]
            if not isinstance(labels, dict) or set(labels) != {"candidate_id", "profile_id"}:
                raise ValueError("invalid selection cell labels")
            child = directory / row["directory"]
            config = json.loads((child / "resolved_config.json").read_text(encoding="utf-8"))
            metrics = json.loads((child / "metrics.json").read_text(encoding="utf-8"))
            environment = json.loads((child / "environment.json").read_text(encoding="utf-8"))
            candidate, profile = labels["candidate_id"], labels["profile_id"]
            validate_processor_config(config["processor"])
            if candidate in processors and processors[candidate] != config["processor"]:
                raise ValueError("candidate processor changes across selection cells")
            processors[candidate] = config["processor"]
            stress = corruption_settings(config)
            if profile in profiles and profiles[profile] != stress:
                raise ValueError("profile corruption settings change across selection cells")
            if profile == protocol["clean_profile"] and stress != corruption_settings({}):
                raise ValueError("clean profile is corrupted")
            profiles[profile] = stress
            evaluation = {k: config[k] for k in (
                "delay_ms", "step_ns", "window_frames", "max_lag_ms",
                "min_velocity_variation_m_s", "min_correlation", "ambiguity_margin",
            )}
            if common_evaluation is not None and evaluation != common_evaluation:
                raise ValueError("evaluation protocol changes across selection cells")
            common_evaluation = evaluation
            identity = {k: config[k] for k in (
                "observations_sha256", "reference_sha256", "start_ns", "end_ns",
            )}
            identity["clock_domain"] = row["clock_domain"]
            identity["python_source_sha256"] = environment["python_source_sha256"]
            sequence = row["sequence_id"]
            if sequence in inputs and inputs[sequence] != identity:
                raise ValueError("sequence inputs/clock/grid/code change across selection cells")
            inputs[sequence] = identity
            with np.load(child / "display_samples.npz", allow_pickle=False) as samples:
                validity = samples["validity"]
                pose_mask = _digest_json({"shape": validity.shape, "dtype": str(validity.dtype),
                                          "bytes": hashlib.sha256(validity.tobytes()).hexdigest()})
            latency_mask = _digest_json([
                [j["status"] == "ok" for j in w["joint_windows"]]
                for w in metrics["latency"]["windows"]
            ])
            condition = dict(config)
            for field in ("processor", "run_id", "observations", "reference"):
                condition.pop(field)
            condition["python_source_sha256"] = environment["python_source_sha256"]
            rows.append({
                **labels, "split": summary["split"], "seed": config["seed"],
                "sequence_id": row["sequence_id"],
                "display_mpjpe_mm": metrics["display_error"]["mpjpe_mm"],
                "jitter_m_s2": metrics["jitter"]["acceleration_error_m_s2"],
                "latency_ms": metrics["latency"]["effective_latency_ms"],
                "joint_coverage": metrics["display_error"]["joint_coverage"],
                "jitter_coverage": metrics["jitter"]["coverage"],
                "latency_coverage": metrics["latency"]["coverage"],
                "input_fingerprint": _digest_json(condition),
                "pose_mask_sha256": pose_mask, "latency_mask_sha256": latency_mask,
            })
        if sha256_file(completion) != entry["sha256"]:
            raise ValueError("selection batch changed during reading")
        provenance.append({"directory": str(directory), "sha256": entry["sha256"]})
    result = select_validation_panel(protocol, rows)
    declared_inputs = {
        name: {k: v for k, v in value.items() if k != "python_source_sha256"}
        for name, value in inputs.items()
    }
    if (processors != declaration["candidate_processors"]
            or profiles != declaration["profile_settings"]
            or common_evaluation != declaration["evaluation_settings"]
            or declared_inputs != declaration["sequence_inputs"]):
        raise ValueError("completed panel does not match the predeclared study")
    if sha256_file(protocol_path) != protocol_hash or sha256_file(manifest_path) != manifest_hash:
        raise ValueError("selection inputs changed during reading")
    if sha256_file(declaration_path) != manifest["declaration_sha256"]:
        raise ValueError("study declaration changed during reading")
    return {**result, "protocol_sha256": protocol_hash, "manifest_sha256": manifest_hash,
            "declaration_sha256": manifest["declaration_sha256"],
            "batch_provenance": provenance, "candidate_processors": processors,
            "profile_settings": profiles, "evaluation_settings": common_evaluation}
