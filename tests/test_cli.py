"""Tests for the project command-line entry point."""

from __future__ import annotations

import pytest

from poseahead import __version__
from poseahead.cli import main


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--version"])

    assert error.value.code == 0
    assert capsys.readouterr().out.strip() == f"poseahead {__version__}"


def test_doctor(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["doctor"]) == 0

    output = capsys.readouterr().out
    assert "poseahead:" in output
    assert "python:" in output
    assert "platform:" in output
    assert "gpu:" in output


@pytest.mark.parametrize(
    "command,label",
    [
        ("check-config", "Invalid config"),
        ("inspect-recording", "Recording error"),
        ("verify-dataset", "Dataset error"),
        ("verify-batch", "Batch error"),
        ("verify-experiment", "Experiment error"),
        ("verify-windows", "Training data error"),
    ],
)
def test_json_commands_keep_error_labels_and_exit_codes(tmp_path, capsys, command, label):
    assert main([command, str(tmp_path / "missing")]) == 2
    result = capsys.readouterr()
    assert result.out == ""
    assert result.err.startswith(label + ": ")


def test_all_json_commands_have_error_labels():
    import argparse

    from poseahead.cli import _ERROR_LABELS, _build_parser

    parser = _build_parser()
    commands = next(
        action.choices
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    assert set(_ERROR_LABELS) == set(commands) - {"doctor", "gui"}
