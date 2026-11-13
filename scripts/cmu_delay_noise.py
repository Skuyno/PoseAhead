"""Validation-only display-time comparison with 100-ms delay and synthetic noise."""

import argparse
import json
from pathlib import Path

from poseahead.artifacts import sha256_file
from poseahead.datasets.index import verify_dataset
from poseahead.evaluation.batch import run_batch, verify_batch


def conditions(checkpoint, prefilter_sweep=False, learned_only=False):
    processors = [
        {"method": method, "parameters": {}}
        for method in ("raw", "ema", "one_euro", "kalman")
    ]
    processors += [
        {"method": "kalman_cv", "parameters": {"target_offset_ms": 100}},
        {"method": "learned", "parameters": {
            "checkpoint": str(Path(checkpoint).resolve()),
            "checkpoint_sha256": sha256_file(Path(checkpoint) / "manifest.json"),
            "history_sampling": "linear_past"}},
    ]
    if learned_only:
        processors = [processors[-1]]
    if prefilter_sweep:
        learned = processors[-1]
        processors = [{"method": "learned", "parameters": {
            **learned["parameters"], "history_prefilter_ms": tau}}
            for tau in (0, 10, 25, 50)]
    return [(noise, p) for noise in (0., .01) for p in processors]


def config_for(entry, noise, processor):
    # 120-Hz source has an exact timestamp every three frames: 25 ms / 40 Hz.
    # No fabricated timestamp or resampled reference is needed.
    start, step = 500_000_000, 25_000_000
    end = entry["end_ns"] // step * step
    if entry["start_ns"] != 0 or end - start < 39 * step:
        raise ValueError("pilot needs zero-origin clips and a full post-warmup window")
    run_id = f"{entry['sequence_id'].replace('/', '-')}-{processor['method']}-{noise}"
    if "history_prefilter_ms" in processor["parameters"]:
        run_id += f"-prefilter{processor['parameters']['history_prefilter_ms']}"
    return {"version": 1,
            "run_id": run_id,
            "observations": entry["path"], "reference": entry["path"],
            "observations_sha256": entry["sha256"], "reference_sha256": entry["sha256"],
            "processor": processor, "seed": 42, "delay_ms": 100,
            "noise_std_m": noise, "dropout_probability": 0.,
            "start_ns": start, "end_ns": end, "step_ns": step,
            "window_frames": 40, "max_lag_ms": 200}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="expanded pilot index.json")
    parser.add_argument("checkpoint", type=Path, help="fixed seed-42 100-ms checkpoint")
    parser.add_argument("output", type=Path)
    parser.add_argument("--prefilter-sweep", action="store_true")
    parser.add_argument("--learned-only", action="store_true")
    args = parser.parse_args()
    before = sha256_file(args.dataset)
    verify_dataset(args.dataset)
    manifest = json.loads(args.dataset.read_text())
    if sha256_file(args.dataset) != before:
        raise ValueError("dataset index changed")
    model = json.loads((args.checkpoint / "manifest.json").read_text())
    if model["horizon_ms"] != 100 or model["config"]["seed"] != 42:
        raise ValueError("protocol requires preselected 100-ms seed-42 model")
    entries = [e for e in manifest["recordings"] if e["split"] == "validation"]
    if len(entries) != 3:
        raise ValueError("protocol requires exactly three validation clips")
    cases = [config_for(entry, noise, processor) for entry in entries
             for noise, processor in conditions(args.checkpoint, args.prefilter_sweep,
                                                  args.learned_only)]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = []
    for i, config in enumerate(cases):
        name = f"config-{i:03d}.json"
        with (output / name).open("x", encoding="utf-8") as stream:
            json.dump(config, stream, indent=2)
        paths.append(name)
    with (output / "suite.json").open("x", encoding="utf-8") as stream:
        json.dump({"version": 1, "batch_id": "cmu-validation-delay100-noise",
                   "split": "validation", "experiments": paths}, stream, indent=2)
    with (output / "protocol.json").open("x", encoding="utf-8") as stream:
        json.dump({"dataset_index_sha256": before, "test_scored": False,
                   "learned_only": args.learned_only,
                   "prefilter_sweep_ms": [0, 10, 25, 50] if args.prefilter_sweep else None,
                   "noise": "independent joint XYZ Gaussian sigma 0 or 10mm; pelvis/root unchanged",
                   "source_hz": 120, "display_hz": 40, "delay_ms": 100,
                   "warmup_excluded_ms": 500, "model_seed": 42,
                   "scope": "synthetic stress, not calibrated camera noise or measured live delay",
                   "tuning": ("predeclared validation prefilter sweep" if args.prefilter_sweep
                              else "no parameter tuning; existing filter defaults"),
                   "comparison": "Kalman CV is not last-two-sample extrapolation"},
                  stream, indent=2)
    import torch

    torch.set_num_threads(2)
    result = run_batch(output / "suite.json", output / "batch")
    verified = verify_batch(output / "batch")
    with (output / "verified.json").open("x", encoding="utf-8") as stream:
        json.dump(verified, stream, indent=2)
    print(json.dumps({"verification": verified, "groups": result["groups"]}, indent=2))


if __name__ == "__main__":
    main()
