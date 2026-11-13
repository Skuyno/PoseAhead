"""Freeze/run an initial M8 validation threshold sweep; no training or held-out scoring."""

import argparse
import json
from pathlib import Path

from cmu_robust_selection import prepare_panel, run_study, write_json

from poseahead.artifacts import sha256_file
from poseahead.prediction.predictability import HORIZONS_MS


def candidates(bank_root):
    bank = {}
    for h in HORIZONS_MS:
        path = (Path(bank_root) / f"h{h}_seed42").resolve()
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest["horizon_ms"] != h or manifest["config"]["seed"] != 42:
            raise ValueError("adaptive study requires preselected seed-42 checkpoints")
        bank[str(h)] = {"checkpoint": str(path),
                        "checkpoint_sha256": sha256_file(path / "manifest.json")}
    processors = {
        "kalman_cv": {"method": "kalman_cv", "parameters": {"target_offset_ms": 100}},
        "raw": {"method": "raw", "parameters": {}},
    }
    # This small grid is declared before this panel, not calibrated or a final
    # motion-class threshold. Both modes share the same feature gates/model bank.
    for limit_mm in (30, 60):
        for mode in ("horizon_only", "horizon_blend"):
            processors[f"adaptive_e{limit_mm}_{mode}"] = {
                "method": "adaptive", "parameters": {
                    "checkpoints": bank, "history_sampling": "linear_past",
                    "policy": {"version": 1,
                               "error_limits_m": {str(h): limit_mm / 1000 for h in HORIZONS_MS},
                               "max_horizon_ms": 100, "minimum_matured": 3,
                               "maximum_acceleration_m_s2": 20., "mode": mode},
                },
            }
    return processors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("bank_root", type=Path)
    parser.add_argument("output", type=Path)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--prepare-only", action="store_true")
    actions.add_argument("--run-prepared", action="store_true")
    args = parser.parse_args()
    import torch

    torch.set_num_threads(2)
    if args.run_prepared:
        study = json.loads((args.output / "adaptive_study.json").read_text(encoding="utf-8"))
        if sha256_file(args.dataset) != study["dataset_index_sha256"]:
            raise ValueError("prepared study dataset changed")
        if str(args.bank_root.resolve()) != study.get("adaptive_bank_root"):
            raise ValueError("prepared adaptive bank root differs")
    else:
        study = prepare_panel(
            args.dataset, candidates(args.bank_root), args.output,
            protocol_id="cmu-adaptive-threshold-pilot-v1", seeds=(42,),
            checkpoint_selection="unchanged clean seed-42 four-horizon bank; no retraining",
        )
        study.update(adaptive_bank_root=str(args.bank_root.resolve()),
                     limitation="one noise seed; repeatedly consulted validation; pilot only")
        # prepare_panel already published study.json. Record adaptive-specific
        # identity separately, preserving its exclusive-write guarantee.
        write_json(args.output / "adaptive_study.json", study)
    print("Frozen M8 pilot: 6 candidates x 4 Gaussian profiles x 3 validation clips = 72 runs",
          flush=True)
    if not args.prepare_only:
        run_study(args.output, study)


if __name__ == "__main__":
    main()
