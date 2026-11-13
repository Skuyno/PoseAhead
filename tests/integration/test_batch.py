import json
from dataclasses import replace

import pytest

from poseahead.cli import main
from poseahead.core import PoseSequence
from poseahead.evaluation.batch import _comparison_key, _summary, run_batch, verify_batch
from poseahead.evaluation.experiment import load_experiment_config
from poseahead.evaluation.synthetic import create_synthetic_experiment
from poseahead.recording import load_recording, save_recording


def fixture(tmp_path, *, entries=None, split="unlabelled"):
    inputs = tmp_path / "inputs"
    create_synthetic_experiment(inputs)
    manifest = inputs / "batch.json"
    manifest.write_text(json.dumps({"version": 1, "batch_id": "sanity", "split": split,
                                    "experiments": entries or ["raw.json", "ema.json"]}),
                        encoding="utf-8")
    return manifest


def test_suite_runs_and_verifies_without_reusing_output(tmp_path, capsys):
    manifest = fixture(tmp_path)
    output = tmp_path / "batch"
    assert main(["run-batch", str(manifest), str(output)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert len(summary["groups"]) == len(summary["runs"]) == 2
    assert summary["runs"][0]["latency_ms"] == pytest.approx(100, abs=.3)
    assert main(["verify-batch", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["verified_runs"] == 2
    with pytest.raises(FileExistsError):
        run_batch(manifest, output)
    metrics = output / "run-0000" / "metrics.json"
    metrics.write_bytes(metrics.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum"):
        verify_batch(output)


@pytest.mark.parametrize("kind", ["run_id", "sequence", "mixed_split", "bad_grid"])
def test_batch_preflight_creates_nothing_on_invalid_inputs(tmp_path, kind):
    manifest = fixture(tmp_path)
    config_path = manifest.parent / "ema.json"
    config = json.loads(config_path.read_text())
    if kind == "run_id":
        config["run_id"] = "synthetic-raw-delay100"
    elif kind == "sequence":
        config["processor"] = {"method": "raw", "parameters": {}}
    elif kind == "bad_grid":
        config["start_ns"] += 1
    else:
        value = json.loads(manifest.read_text())
        value["split"] = "test"
        manifest.write_text(json.dumps(value), encoding="utf-8")
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        run_batch(manifest, tmp_path / "batch")
    assert not (tmp_path / "batch").exists()


@pytest.mark.parametrize("subject, split, valid", [
    ("S8", "validation", True), ("S9", "validation", False), ("S1", "test", False),
])
def test_fixed_subject_split_verified_from_every_frame(tmp_path, subject, split, valid):
    manifest = fixture(tmp_path, entries=["raw.json"], split=split)
    reference = load_recording(manifest.parent / "truth.npz")
    labelled = PoseSequence(reference.sequence_id, reference.clock_domain, tuple(
        replace(f, metadata={"subject": subject, "split": split, "dataset": "Human3.6M"})
        for f in reference.frames
    ))
    save_recording(manifest.parent / "labelled.npz", labelled)
    path = manifest.parent / "raw.json"
    config = json.loads(path.read_text())
    config.update(reference="labelled.npz", observations="labelled.npz")
    path.write_text(json.dumps(config), encoding="utf-8")
    if valid:
        result = run_batch(manifest, tmp_path / "batch")
        assert result["runs"][0]["subject"] == "S8"
    else:
        with pytest.raises(ValueError, match="split"):
            run_batch(manifest, tmp_path / "batch")
        assert not (tmp_path / "batch").exists()


def test_summary_is_sequence_macro_not_pooled_frames_and_reports_nulls():
    rows = [{"condition": {"processor": "raw"}, "run_id": str(i),
             "display_mpjpe_mm": value, "root_error_mm": value,
             "jitter_m_s2": value, "latency_ms": None}
            for i, value in enumerate((1., 9., None))]
    report = _summary(rows)["groups"][0]["metrics"]
    assert report["display_mpjpe_mm"] == {
        "macro_mean": 5., "eligible_sequences": 2, "total_sequences": 3,
    }
    assert report["latency_ms"]["macro_mean"] is None


@pytest.mark.parametrize("stress", [
    {"outlier_probability": .01, "outlier_std_m": .2},
    {"gap_period_ms": 1000, "gap_duration_ms": 150}, {"observation_stride": 6},
])
def test_comparison_conditions_do_not_collapse_extended_corruption(tmp_path, stress):
    manifest = fixture(tmp_path)
    raw = load_experiment_config(manifest.parent / "raw.json")
    assert _comparison_key(raw) != _comparison_key({**raw, **stress})


def test_batch_manifest_cannot_reference_external_artifacts(tmp_path):
    manifest = fixture(tmp_path, entries=["raw.json"])
    output = tmp_path / "batch"
    run_batch(manifest, output)
    completion = output / "batch_completion.json"
    data = json.loads(completion.read_text())
    data["sha256"]["../outside.txt"] = "bad"
    completion.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="path"):
        verify_batch(output)


def test_two_sequences_share_condition_without_frame_weighting(tmp_path):
    manifest = fixture(tmp_path, entries=["raw.json", "second.json"])
    source = load_recording(manifest.parent / "truth.npz")
    second = PoseSequence("second-sequence", source.clock_domain,
                          tuple(replace(f, joint_positions=f.joint_positions * .5)
                                for f in source.frames))
    save_recording(manifest.parent / "second.npz", second)
    config = json.loads((manifest.parent / "raw.json").read_text())
    config.update(run_id="second-raw", observations="second.npz", reference="second.npz")
    (manifest.parent / "second.json").write_text(json.dumps(config), encoding="utf-8")
    report = run_batch(manifest, tmp_path / "batch")
    assert len(report["groups"]) == 1
    values = [row["display_mpjpe_mm"] for row in report["runs"]]
    assert values[1] == pytest.approx(values[0] * .5)
    metric = report["groups"][0]["metrics"]["display_mpjpe_mm"]
    assert metric["macro_mean"] == pytest.approx(sum(values) / 2)
    assert metric["eligible_sequences"] == 2


def test_missing_entry_never_starts_first_run(tmp_path):
    manifest = fixture(tmp_path, entries=["raw.json", "missing.json"])
    with pytest.raises(FileNotFoundError):
        run_batch(manifest, tmp_path / "batch")
    assert not (tmp_path / "batch").exists()
