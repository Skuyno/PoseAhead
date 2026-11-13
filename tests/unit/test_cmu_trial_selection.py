import importlib.util
from pathlib import Path

from poseahead.datasets.splits import CMU_SUBJECT_SPLITS


def test_expansion_changes_training_only():
    path = Path(__file__).resolve().parents[2] / "scripts" / "cmu_pilot_training.py"
    spec = importlib.util.spec_from_file_location("cmu_training_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original, expanded = module.selected_trials(), module.selected_trials(True)
    assert len(original) == 13
    assert len(expanded) == 22
    for split in ("validation", "test"):
        before = [pair for pair in original if CMU_SUBJECT_SPLITS[f"CMU:{pair[0]}"] == split]
        after = [pair for pair in expanded if CMU_SUBJECT_SPLITS[f"CMU:{pair[0]}"] == split]
        assert before == after
