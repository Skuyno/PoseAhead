"""Config-driven single-sequence baseline experiments with auditable artifacts."""

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from poseahead import __version__
from poseahead.artifacts import sha256_file
from poseahead.core import PoseSequence
from poseahead.evaluation import (
    MotionSeries,
    score_effective_latency,
    score_jitter,
    score_pose_error,
)
from poseahead.evaluation.corruption import (
    CORRUPTION_DEFAULTS,
    corruption_settings,
    degrade_observations,
)
from poseahead.evaluation.snapshots import snapshot_motion
from poseahead.processing.config import (
    create_processor,
    resolve_processor_paths,
    validate_processor_config,
)
from poseahead.processing.simulation import simulate_processing
from poseahead.recording import load_recording, save_recording


def _integer(name, value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _processor_config(processor):
    parameters = dict(getattr(processor, "parameters", {}))
    if processor.method == "ema":
        parameters = {"time_constant_ms": processor.time_constant_ms,
                      "reset_gap_ms": processor.reset_gap_ms}
    if processor.method == "kalman_cv":
        parameters["target_offset_ms"] = processor.target_offset_ns / 1e6
    return {"method": processor.method, "parameters": parameters}


def load_experiment_config(path):
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"version", "run_id", "observations", "reference", "processor",
                "seed", "delay_ms", "start_ns", "end_ns", "step_ns", "window_frames"}
    optional = {*CORRUPTION_DEFAULTS, "observations_sha256", "reference_sha256", "max_lag_ms",
                "min_velocity_variation_m_s", "min_correlation", "ambiguity_margin"}
    if not isinstance(config, dict) or not required <= config.keys() or config.keys() - (
        required | optional
    ):
        raise ValueError("invalid experiment config fields")
    if type(config["version"]) is not int or config["version"] != 1:
        raise ValueError("unsupported experiment config version")
    if not isinstance(config["run_id"], str) or not config["run_id"].strip():
        raise ValueError("run_id must be a nonempty string")
    for field in ("observations", "reference"):
        if not isinstance(config[field], str) or not config[field]:
            raise ValueError(f"{field} must be a path")
        config[field] = str((path.parent / config[field]).resolve())
    for field in ("seed", "delay_ms", "start_ns", "end_ns"):
        _integer(field, config[field])
    for field in ("step_ns", "window_frames"):
        _integer(field, config[field], 1)
    if not config["start_ns"] < config["end_ns"] < 2**64:
        raise ValueError("evaluation requires start_ns < end_ns < 2**64 (inclusive endpoints)")
    if config["delay_ms"] * 1_000_000 >= 2**64:
        raise ValueError("delay exceeds timestamp range")
    count = (config["end_ns"] - config["start_ns"]) // config["step_ns"] + 1
    if not 3 <= count <= 1_000_000 or not 3 <= config["window_frames"] <= count:
        raise ValueError("evaluation grid/window size outside supported bounds")
    profile = corruption_settings(config)
    for field in CORRUPTION_DEFAULTS:
        config.pop(field, None)
    config.update(profile)
    defaults = {"max_lag_ms": 200., "min_velocity_variation_m_s": 1e-4,
                "min_correlation": .6, "ambiguity_margin": .02}
    for field, value in defaults.items():
        config.setdefault(field, value)
        if type(config[field]) not in (int, float) or not np.isfinite(config[field]):
            raise ValueError(f"{field} must be finite numeric")
    specification = config["processor"]
    validate_processor_config(specification)
    processor = create_processor(specification["method"], resolve_processor_paths(
        specification["method"], specification["parameters"], path.parent,
    ))
    config["processor"] = _processor_config(processor)
    # Validate latency options/window before creating any run artifacts.
    dummy = MotionSeries(tuple(i * config["step_ns"] for i in range(config["window_frames"])),
                         np.zeros((config["window_frames"], 1, 3)),
                         np.ones((config["window_frames"], 1), bool), "validation")
    score_effective_latency(dummy, dummy, **_latency_options(config))
    return config


def _latency_options(config):
    return {key: config[key] for key in ("max_lag_ms", "min_velocity_variation_m_s",
                                        "min_correlation", "ambiguity_margin")}


class _TimedProcessor:
    def __init__(self, processor):
        self.processor = processor

    def reset(self):
        self.processor.reset()
        self.frames, self.durations_ns = [], []

    def process(self, frame):
        start = time.perf_counter_ns()
        result = self.processor.process(frame)
        self.durations_ns.append(time.perf_counter_ns() - start)
        self.frames.append(result)
        return result


def _environment():
    repo = Path(__file__).resolve().parents[3]
    def git(*args):
        try:
            result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                                    text=True, encoding="utf-8", errors="replace",
                                    timeout=5, check=True)
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    # Includes uncommitted/untracked Python source; revision alone is insufficient.
    source_hash = hashlib.sha256()
    for path in sorted((repo / "src" / "poseahead").rglob("*.py")):
        source_hash.update(path.relative_to(repo).as_posix().encode() + b"\0")
        source_hash.update(path.read_bytes() + b"\0")
    return {"python": platform.python_version(), "numpy": np.__version__,
            "poseahead": __version__, "platform": platform.platform(),
            "git_revision": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
            "python_source_sha256": source_hash.hexdigest()}


def run_experiment(config_path, output_directory):
    """Create a new run directory; completion.json is written only after all artifacts.

    Existing directories are never reused. Failures after creation leave an
    incomplete directory for diagnosis, without a completion marker.
    """
    config = load_experiment_config(config_path)
    output_directory = Path(output_directory)
    if output_directory.exists():
        raise FileExistsError("experiment output directory already exists")
    sequences = {}
    for field in ("observations", "reference"):
        digest = sha256_file(config[field])
        expected = config.get(f"{field}_sha256", digest)
        if expected != digest:
            raise ValueError(f"{field} checksum mismatch")
        sequences[field] = load_recording(config[field])
        if sha256_file(config[field]) != digest:
            raise ValueError(f"{field} changed during loading")
        config[f"{field}_sha256"] = digest
    source, reference = sequences["observations"], sequences["reference"]
    if source.clock_domain != reference.clock_domain:
        raise ValueError("observation and reference clocks differ")
    times = tuple(range(config["start_ns"], config["end_ns"] + 1, config["step_ns"]))
    lookup = {frame.source_timestamp_ns: frame for frame in reference.frames}
    if any(t not in lookup for t in times):
        raise ValueError("reference must contain every evaluation timestamp; resample explicitly")
    truth = [lookup[t] for t in times]
    reference_motion = MotionSeries(times, np.stack([f.joint_positions for f in truth]),
                                   np.stack([f.joint_validity & (f.joint_confidence > 0)
                                             for f in truth]), source.clock_domain)
    observations = degrade_observations(source, seed=config["seed"], **corruption_settings(config))
    processor = _TimedProcessor(create_processor(**config["processor"]))
    snapshots = list(simulate_processing(observations, processor, times,
                                         delay_ns=config["delay_ms"] * 1_000_000))
    motion = snapshot_motion(snapshots)
    shared = motion.validity & reference_motion.validity
    error = np.linalg.norm(motion.positions_m - reference_motion.positions_m, axis=2)[shared]
    root_errors = [float(np.linalg.norm(s.pose.root_position - f.root_position))
                   for s, f in zip(snapshots, truth, strict=True)
                   if s.pose is not None and s.pose.root_position is not None
                   and f.root_position is not None]
    window_size = config["window_frames"]
    windows = []
    for start in range(0, len(times) - window_size + 1, window_size):
        stop = start + window_size
        pair = [MotionSeries(times[start:stop], m.positions_m[start:stop],
                             m.validity[start:stop], m.clock_domain)
                for m in (motion, reference_motion)]
        windows.append({"start_ns": times[start], "end_ns": times[stop - 1],
                        **score_effective_latency(*pair, **_latency_options(config))})
    lags = [joint["lag_ms"] for w in windows for joint in w["joint_windows"]
            if joint["status"] == "ok"]
    processed = (PoseSequence(f"{source.sequence_id}/runs/{config['run_id']}",
                             source.clock_domain, tuple(processor.frames))
                 if processor.frames else None)
    report = {
        "run_id": config["run_id"], "status": "complete",
        "protocol": "canonical17_pelvis_relative_including_zero_pelvis_no_alignment",
        "display_time_policy": "evaluation_grid_with_last_available_output_held",
        "target_error_scope": "all_observations_processed_through_last_evaluation_including_warmup",
        "display_error": {"mpjpe_mm": float(error.mean() * 1000) if error.size else None,
                          "joint_coverage": float(shared.mean()),
                          "valid_joint_pairs": int(shared.sum()),
                          "possible_joint_pairs": int(shared.size),
                          "root_error_mm": float(np.mean(root_errors) * 1000)
                          if root_errors else None, "root_pairs": len(root_errors),
                          "root_coverage": len(root_errors) / len(times)},
        "target_error": score_pose_error(processed, reference, interpolation="linear",
                                          max_gap_ns=config["step_ns"]) if processed else None,
        "jitter": score_jitter(motion, reference_motion),
        "latency": {"effective_latency_ms": float(np.median(lags)) if lags else None,
                    "eligible_joint_windows": len(lags), "total_joint_windows": len(windows) * 17,
                    "coverage": len(lags) / (len(windows) * 17), "windows": windows,
                    "discarded_tail_frames": len(times) % window_size,
                    "aggregation": "median_eligible_joint_windows_nonoverlapping"},
        "runtime": {"processor_calls": len(processor.frames),
                    "processor_p50_ms": float(np.percentile(processor.durations_ns, 50) / 1e6)
                    if processor.frames else None,
                    "processor_p95_ms": float(np.percentile(processor.durations_ns, 95) / 1e6)
                    if processor.frames else None,
                    "scope": "CPU_process_call_including_first_call_not_injected_into_simulation"},
        "reference_role": "caller_supplied_reference_not_automatically_verified_ground_truth",
    }
    if processor.processor.method in ("learned", "adaptive"):
        statuses = {}
        for frame in processor.frames:
            status = frame.metadata["prediction_status"]
            statuses[status] = statuses.get(status, 0) + 1
        report["learned_prediction"] = {
            "observation_output_counts": statuses,
            "prediction_fraction": statuses.get("predicted", 0) / len(processor.frames)
            if processor.frames else 0.,
            "root_policy": "hold_latest_observation",
            "history_policy": processor.processor.history_sampling,
        }
    if processor.processor.method == "adaptive":
        choices, alphas = {}, []
        partial = 0
        for frame in processor.frames:
            decision = frame.metadata["adaptive_decision"]
            h = str(decision["horizon_ms"])
            choices[h] = choices.get(h, 0) + 1
            alphas.append(decision["alpha"])
            partial += int(0 < decision["alpha"] < 1)
        report["adaptive_prediction"] = {
            "horizon_output_counts_ms": choices,
            "mean_alpha": float(np.mean(alphas)) if alphas else None,
            "partial_compensation_frames": partial,
            "decision_log": "processed.npz: each frame metadata.adaptive_decision",
            "probe_error_scope": "matured_observation_consistency_not_ground_truth_accuracy",
            "score_semantics": "threshold_margin_not_probability",
            "target_error_semantics": "nominal_future_target_for_partial_compensation",
            "threshold_selection": "caller_supplied_not_automatically_validation_verified",
        }
        if getattr(processor.processor, "fallback_method", "raw") == "kalman_cv":
            report["learned_prediction"]["root_policy"] = (
                "always_kalman_baseline_except_tracking_loss_raw"
            )
            report["adaptive_prediction"].update(
                baseline_method="kalman_cv",
                baseline_only_frames=sum(f.metadata["prediction_status"] == "baseline"
                                         for f in processor.frames),
                target_error_semantics="same_future_target_kalman_neural_estimates",
                alpha_semantics="neural_weight_against_same_target_kalman_not_raw",
                root_policy="always_kalman_baseline_except_tracking_loss_raw",
            )
    environment = _environment()
    output_directory.mkdir(parents=True, exist_ok=False)
    def write_json(name, value):
        with (output_directory / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
    write_json("resolved_config.json", config)
    write_json("environment.json", environment)
    save_recording(output_directory / "observations_used.npz", observations)
    if processed is not None:
        save_recording(output_directory / "processed.npz", processed)
    with (output_directory / "display_samples.npz").open("xb") as stream:
        np.savez_compressed(stream, timestamps_ns=np.array(times, dtype=np.uint64),
                            positions_m=motion.positions_m, validity=motion.validity,
                            observation_present=np.array([s.pose is not None for s in snapshots]),
                            source_ns=np.array([s.pose.source_timestamp_ns if s.pose else 0
                                                for s in snapshots], dtype=np.uint64),
                            target_ns=np.array([s.represented_timestamp_ns or 0 for s in snapshots],
                                               dtype=np.uint64))
    write_json("metrics.json", report)
    hashes = {p.name: sha256_file(p) for p in output_directory.iterdir() if p.is_file()}
    write_json("completion.json", {"run_id": config["run_id"], "sha256": hashes})
    return report


def verify_experiment(directory):
    """Verify all completed artifacts without trusting paths in the manifest."""
    directory = Path(directory)
    manifest = json.loads((directory / "completion.json").read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or set(manifest) != {"run_id", "sha256"}
            or not isinstance(manifest["sha256"], dict)):
        raise ValueError("invalid completion manifest")
    hashes = manifest["sha256"]
    required = {"resolved_config.json", "environment.json", "observations_used.npz",
                "display_samples.npz", "metrics.json"}
    if not required <= hashes.keys() or hashes.keys() - required - {"processed.npz"}:
        raise ValueError("invalid artifact names in completion manifest")
    for name, digest in hashes.items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"artifact checksum mismatch: {name}")
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    config = json.loads((directory / "resolved_config.json").read_text(encoding="utf-8"))
    if (not isinstance(metrics, dict) or not isinstance(config, dict)
            or manifest["run_id"] != metrics.get("run_id")
            or manifest["run_id"] != config.get("run_id")
            or metrics.get("status") != "complete"):
        raise ValueError("artifact run IDs or completion status disagree")
    return {"run_id": manifest["run_id"], "verified_files": len(hashes), "status": "verified"}
