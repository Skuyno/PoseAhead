import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def analyzer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("analyze_adaptive_motion")


def test_window_error_jitter_and_coverage_use_ground_truth_and_valid_joints(analyzer):
    reference = np.zeros((40, 17, 3))
    positions = reference.copy()
    positions[:, 1:, 0] = .01
    validity = np.ones((40, 17), bool)
    times = tuple(i * 25_000_000 for i in range(40))
    report = analyzer.window_metrics(positions, validity, (reference, validity), times, "test")
    assert report["mpjpe_mm"] == pytest.approx(10 * 16 / 17)
    assert report["jitter_m_s2"] == 0
    assert report["pose_coverage"] == 1
    validity[:, 1] = False
    report = analyzer.window_metrics(positions, validity, (reference, np.ones_like(validity)),
                                     times, "test")
    assert report["mpjpe_mm"] == pytest.approx(10 * 15 / 16)
    assert report["pose_coverage"] == pytest.approx(16 / 17)


def test_incomplete_class_cell_is_not_treated_as_finite_macro(analyzer):
    assert analyzer.mean_complete([1., None]) is None
    assert analyzer.mean_complete([1., 3.]) == 2.
