import numpy as np
import pytest

from poseahead.evaluation.comparison import _rules, _same_coverage, paired_bootstrap, pareto_front


def test_constant_paired_difference_has_known_ci_and_sign():
    baseline = np.array([[100, 10, 5], [150, 20, 10], [75, 30, 8]])
    candidate = baseline + [-30, -1, -.5]
    report = paired_bootstrap(baseline, candidate, seed=42, repetitions=100)
    assert report["latency_ms"]["delta_ci95"] == [-30, -30]
    assert report["display_mpjpe_mm"]["delta_ci95"] == [-1, -1]
    assert _rules(report, 3)["rule1"] is True
    assert _rules(report, 3)["rule2"] is True


def test_tradeoff_boundary_and_zero_comparator_error():
    baseline = [[100, 10, 5], [100, 10, 5]]
    candidate = [[80, 11, 5.5], [80, 11, 5.5]]
    report = paired_bootstrap(baseline, candidate, repetitions=100)
    assert not _rules(report, 2)["rule1"]
    assert _rules(report, 2)["rule2"]
    report = paired_bootstrap([[100, 0, 0]] * 2, [[70, .001, 0]] * 2, repetitions=100)
    assert not _rules(report, 2)["rule2"]


def test_single_sequence_has_no_ci_or_rule_claim():
    report = paired_bootstrap([[100, 10, 5]], [[70, 5, 1]], repetitions=100)
    assert report["latency_ms"]["delta_ci95"] is None
    assert _rules(report, 1)["rule1"] is None
    assert _rules(None, 0)["rule2"] is None


def test_seeded_bootstrap_is_reproducible():
    baseline = np.arange(30).reshape(10, 3)
    candidate = baseline + np.sin(baseline)
    assert paired_bootstrap(baseline, candidate, seed=5) == paired_bootstrap(
        baseline, candidate, seed=5,
    )


def test_pareto_keeps_tradeoffs_and_ties_but_excludes_dominated_points():
    assert pareto_front([[100, 10, 5], [80, 11, 6], [110, 11, 6], [100, 10, 5]]) == [0, 1, 3]


@pytest.mark.parametrize("a,b", [([], []), ([[1, 2]], [[1, 2]]),
                                  ([[1, 2, 3]], [[1, 2, float("nan")]])])
def test_invalid_metrics(a, b):
    with pytest.raises(ValueError):
        paired_bootstrap(a, b)


def test_equal_coverage_fractions_with_different_masks_are_not_paired(tmp_path):
    import json

    for name, mask in (("a", [[True, False], [True, False]]),
                       ("b", [[False, True], [False, True]])):
        folder = tmp_path / name
        folder.mkdir()
        np.savez(folder / "display_samples.npz", validity=np.array(mask, bool))
        (folder / "metrics.json").write_text(json.dumps({"latency": {"windows": [
            {"joint_windows": [{"status": "ok"}, {"status": "invalid"}]}
        ]}}), encoding="utf-8")
    assert not _same_coverage(tmp_path, {"directory": "a"}, {"directory": "b"})
    np.savez(tmp_path / "b" / "display_samples.npz",
             validity=np.array([[True, False], [True, False]], bool))
    assert _same_coverage(tmp_path, {"directory": "a"}, {"directory": "b"})
    (tmp_path / "b" / "metrics.json").write_text(json.dumps({"latency": {"windows": [
        {"joint_windows": [{"status": "invalid"}, {"status": "ok"}]}
    ]}}), encoding="utf-8")
    assert not _same_coverage(tmp_path, {"directory": "a"}, {"directory": "b"})
