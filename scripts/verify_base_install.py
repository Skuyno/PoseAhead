"""Build and check a non-editable base wheel in a new isolated venv, without a camera."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new report directory; must not already exist")
    args = parser.parse_args(argv)
    repository = Path(__file__).resolve().parents[1]
    output = (
        args.output
        or repository
        / "outputs"
        / f"base_install_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    ).resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PIP_TARGET", "PIP_PREFIX", "PIP_USER"):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PIP_CONFIG_FILE"] = os.devnull
    evidence = {
        "status": "running",
        "scope": "fresh venv on the same host, not a clean OS",
        "output_directory": str(output),
        "commands": [],
        "camera_started": False,
    }

    def run(command, *, cwd=output, timeout=180):
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
        )
        evidence["commands"].append(
            {
                "command": [str(value) for value in command],
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
        completed.check_returncode()
        return completed.stdout

    try:
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--wheel-dir",
                str(output),
                str(repository),
            ]
        )
        wheels = list(output.glob("poseahead-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError("Expected exactly one PoseAhead wheel")
        venv = output / "environment"
        run([sys.executable, "-m", "venv", str(venv)])
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        entry = venv / ("Scripts/poseahead.exe" if os.name == "nt" else "bin/poseahead")
        run([str(python), "-I", "-m", "pip", "install", str(wheels[0])])
        run([str(python), "-I", "-m", "pip", "check"])
        probe = run([str(python), "-I", str(repository / "scripts/base_install_probe.py")])
        evidence["probe"] = json.loads(probe)
        run([str(entry), "--version"])
        run([str(entry), "--help"])
        run([str(entry), "gui", "--help"])
        run([str(python), "-I", "-m", "poseahead.runtime.webcam", "--help"])
        run([str(entry), "check-config", str(repository / "configs/live/default.json")])
        evidence["status"] = "passed"
    except Exception as error:
        evidence.update(status="failed", error=str(error))
        raise
    finally:
        with (output / "report.json").open("x", encoding="utf-8") as stream:
            json.dump(evidence, stream, ensure_ascii=False, indent=2, allow_nan=False)
        print("Base installation evidence:", output / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
