import json
from dataclasses import replace

import pytest

from poseahead.cli import main
from poseahead.core import PoseSequence
from poseahead.evaluation.batch import run_batch
from poseahead.evaluation.comparison import compare_batch, freeze_comparator
from poseahead.evaluation.synthetic import create_synthetic_experiment
from poseahead.recording import load_recording, save_recording


def suite(tmp_path, split):
    root = tmp_path / split
    create_synthetic_experiment(root)
    raw = load_recording(root / "truth.npz")
    subject = "S8" if split == "validation" else "S9"
    entries = []
    for sequence_number in (1, 2):
        frames = tuple(replace(f, metadata={**f.metadata, "dataset": "Human3.6M",
                                            "subject": subject, "split": split})
                       for f in raw.frames)
        path = root / f"sequence{sequence_number}.npz"
        save_recording(path, PoseSequence(
            f"{subject}/clip{sequence_number}", raw.clock_domain, frames,
        ))
        for method in ("raw", "ema"):
            config = json.loads((root / f"{method}.json").read_text())
            config.update(observations=path.name, reference=path.name,
                          run_id=f"{method}-{sequence_number}")
            name = f"{method}-{sequence_number}.json"
            (root / name).write_text(json.dumps(config), encoding="utf-8")
            entries.append(name)
    manifest = root / "suite.json"
    manifest.write_text(json.dumps({"version": 1, "batch_id": split, "split": split,
                                    "experiments": entries}), encoding="utf-8")
    out = root / "results"
    run_batch(manifest, out)
    return out


def test_validation_freeze_then_paired_test_comparison(tmp_path):
    validation = suite(tmp_path, "validation")
    frozen = tmp_path / "frozen.json"
    freeze_comparator(validation, "raw-1", frozen)
    with pytest.raises(FileExistsError):
        freeze_comparator(validation, "raw-1", frozen)
    test = suite(tmp_path, "test")
    with pytest.raises(ValueError, match="frozen"):
        compare_batch(test, "raw-1")
    with pytest.raises(ValueError, match="only.*validation"):
        freeze_comparator(test, "raw-1", tmp_path / "bad.json")
    report = compare_batch(test, "raw-1", frozen_comparator=frozen, repetitions=100)
    comparison = report["comparisons"][0]
    assert comparison["eligible_pairs"] == 2
    assert comparison["complete_matched_panel"]
    assert comparison["eligible_for_frozen_rule_claim"]
    assert comparison["metrics"]["latency_ms"]["delta_mean"] > 0
    assert not comparison["passes_rule_on_eligible_pairs"]
    assert report["pareto"]["nondominated"] == ["baseline"]
    assert compare_batch(test, "raw-1", frozen_comparator=frozen, repetitions=100) == report
    with pytest.raises(ValueError, match="does not match"):
        compare_batch(test, "ema-1", frozen_comparator=frozen)


def test_unlabelled_single_sequence_cannot_claim_significance(tmp_path, capsys):
    create_synthetic_experiment(tmp_path / "input")
    root = tmp_path / "input"
    run_batch(root / "batch.json", root / "results")
    report = compare_batch(root / "results", "synthetic-raw-delay100", repetitions=100)
    assert all(not c["eligible_for_frozen_rule_claim"] for c in report["comparisons"])
    assert all(c["rules"]["rule1"] is None for c in report["comparisons"])
    output = root / "comparison.json"
    command = ["compare-batch", str(root / "results"), "synthetic-raw-delay100",
               "--repetitions", "100", "--output", str(output)]
    assert main(command) == 0
    capsys.readouterr()
    assert json.loads(output.read_text())["comparisons"] == report["comparisons"]
    before = output.read_bytes()
    assert main(command) == 2
    assert output.read_bytes() == before
