"""Predeclare same-target Kalman/neural strength adaptation with the existing noisy h100 model."""

import argparse
import json
from pathlib import Path

from cmu_adaptive_stability import candidates as stability_candidates
from cmu_robust_selection import prepare_panel, run_study, write_json

from poseahead.artifacts import sha256_file


def candidates(bank_root, noisy_checkpoint):
    old = stability_candidates(bank_root)
    base = old["adaptive_window200_history10_transition80"]
    noisy_checkpoint = Path(noisy_checkpoint).resolve()
    manifest = json.loads((noisy_checkpoint / "manifest.json").read_text(encoding="utf-8"))
    if manifest["horizon_ms"] != 100 or manifest["config"]["seed"] != 42:
        raise ValueError("study requires the unchanged seed-42 noise-trained h100 checkpoint")
    noisy_entry = {"checkpoint": str(noisy_checkpoint),
                   "checkpoint_sha256": sha256_file(noisy_checkpoint / "manifest.json")}

    def variant(noisy, tau, cap):
        params = {**base["parameters"], "fallback_method": "kalman_cv",
                  "neural_to_baseline_error_ratio": 1., "maximum_neural_weight": cap,
                  "history_prefilter_ms": tau,
                  "policy": {**base["parameters"]["policy"],
                             "error_limits_m": {str(h): .150 for h in (33, 66, 100, 150)}}}
        if noisy:
            # No noisy checkpoints exist for other horizons. Keep the bank pinned,
            # but this SAME-target branch mixes only h100 into Kalman h100 output.
            params["checkpoints"] = {**params["checkpoints"], "100": noisy_entry}
        return {"method": "adaptive", "parameters": params}

    return {
        "kalman_cv": old["kalman_cv"],
        "previous_raw_mixture": base,
        "kalman_clean_h100_tau10_cap25": variant(False, 10, .25),
        "kalman_noisy_h100_tau10_cap25": variant(True, 10, .25),
        "kalman_noisy_h100_tau25_cap25": variant(True, 25, .25),
        "kalman_noisy_h100_tau25_cap10": variant(True, 25, .10),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("bank_root", type=Path)
    parser.add_argument("noisy_checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--prepare-only", action="store_true")
    actions.add_argument("--run-prepared", action="store_true")
    args = parser.parse_args()
    import torch

    torch.set_num_threads(2)
    if args.run_prepared:
        study = json.loads((args.output / "adaptive_study.json").read_text(encoding="utf-8"))
        if (sha256_file(args.dataset) != study["dataset_index_sha256"]
                or str(args.bank_root.resolve()) != study["adaptive_bank_root"]
                or str(args.noisy_checkpoint.resolve()) != study["noisy_checkpoint"]):
            raise ValueError("prepared study input identity differs")
    else:
        study = prepare_panel(
            args.dataset, candidates(args.bank_root, args.noisy_checkpoint), args.output,
            protocol_id="cmu-adaptive-baseline-mixture-v1", seeds=(42,),
            checkpoint_selection="existing clean bank and noise-trained h100; no training or test",
        )
        study.update(adaptive_bank_root=str(args.bank_root.resolve()),
                     noisy_checkpoint=str(args.noisy_checkpoint.resolve()),
                     limitation="one noise seed; repeatedly consulted validation; pilot only",
                     declaration="paired error ratio <=1; capped neural weight .10/.25; target100",
                     policy="150-mm limit; window200; acceleration20; 3 matured; transitions80")
        write_json(args.output / "adaptive_study.json", study)
    print("Frozen same-target baseline pilot: 6 candidates x 4 profiles x 3 clips = 72 runs",
          flush=True)
    if not args.prepare_only:
        run_study(args.output, study)


if __name__ == "__main__":
    main()
