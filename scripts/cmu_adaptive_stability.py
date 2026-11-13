"""Predeclare M8 noise-tolerant dynamics/model-history/transition ablations."""

import argparse
import json
from pathlib import Path

from cmu_adaptive_validation import candidates as initial_candidates
from cmu_robust_selection import prepare_panel, run_study, write_json

from poseahead.artifacts import sha256_file


def candidates(bank_root):
    old = initial_candidates(bank_root)
    base = old["adaptive_e60_horizon_blend"]
    def variant(**options):
        return {"method": "adaptive", "parameters": {**base["parameters"], **options}}
    return {
        "kalman_cv": old["kalman_cv"], "raw": old["raw"],
        "adaptive_v1_e60_blend": base,
        "adaptive_window200": variant(dynamics_window_ms=200),
        "adaptive_window200_history10": variant(dynamics_window_ms=200, history_prefilter_ms=10),
        "adaptive_window200_history10_transition80": variant(
            dynamics_window_ms=200, history_prefilter_ms=10, transition_time_constant_ms=80),
    }


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
        if (sha256_file(args.dataset) != study["dataset_index_sha256"]
                or str(args.bank_root.resolve()) != study["adaptive_bank_root"]):
            raise ValueError("prepared study input identity differs")
    else:
        study = prepare_panel(
            args.dataset, candidates(args.bank_root), args.output,
            protocol_id="cmu-adaptive-stability-ablation-v1", seeds=(42,),
            checkpoint_selection="unchanged clean seed-42 bank; no training or test scoring",
        )
        study.update(adaptive_bank_root=str(args.bank_root.resolve()),
                     limitation="one noise seed; exploratory repeatedly consulted validation",
                     declaration="200-ms trailing fit; 10-ms history EMA; 80-ms weight transitions",
                     policy="same 60-mm/20-m_s2/3-matured/100-ms budget as previous pilot")
        write_json(args.output / "adaptive_study.json", study)
    print("Frozen stability ablation: 6 candidates x 4 profiles x 3 validation clips = 72 runs",
          flush=True)
    if not args.prepare_only:
        run_study(args.output, study)


if __name__ == "__main__":
    main()
