"""Read-only demo preflight: no sockets, cameras, CUDA warmup or test-split scoring."""

import argparse
import json
from pathlib import Path

from poseahead.artifacts import sha256_file
from poseahead.config import load_live_config
from poseahead.desktop.demo import DEMO_PRESETS, demo_options
from poseahead.processing.config import load_processor
from poseahead.recording import load_recording


def check_demo(repository, *, load_models=False):
    repository = Path(repository).resolve()
    checks = []
    for label, preset in DEMO_PRESETS.items():
        try:
            options = demo_options(repository, preset)
            details = {"preset": preset, "label": label, "status": "ready"}
            if options.recording:
                path = Path(options.recording)
                sequence = load_recording(path)
                details.update(
                    recording_sha256=sha256_file(path), frames=len(sequence.frames),
                    source_span_s=(sequence.frames[-1].source_timestamp_ns
                                   - sequence.frames[0].source_timestamp_ns) / 1e9,
                )
            if options.processor_config and load_models:
                processor = load_processor(options.processor_config)
                for frame in sequence.frames[:60]:
                    processor.process(frame)
                details["cpu_model_smoke"] = "passed_60_frames"
            checks.append(details)
        except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
            checks.append({"preset": preset, "label": label, "status": "unavailable",
                           "reason": str(error)})
    project = repository / "unity/PoseAheadAvatar"
    scene = project / "Assets/PoseAhead/Scenes/PoseAheadDiagnostic 1.unity"
    checks.append({"name": "unity_scene",
                   "status": "ready" if scene.is_file() else "unavailable"})
    version_path = project / "ProjectSettings/ProjectVersion.txt"
    try:
        version = next((line.split(":", 1)[1].strip()
                        for line in version_path.read_text(encoding="utf-8").splitlines()
                        if line.startswith("m_EditorVersion:")), None)
        if not version:
            raise ValueError("ProjectVersion.txt has no m_EditorVersion")
        if version != "6000.3.23f1":
            raise ValueError("Presentation requires pinned Unity 6000.3.23f1, got " + version)
        checks.append({"name": "unity_version", "status": "ready", "version": version})
    except (OSError, ValueError) as error:
        checks.append({"name": "unity_version", "status": "unavailable", "reason": str(error)})
    for name, filename in (("live_config", "default.json"),
                           ("fast_live_config", "fp16_conv.json")):
        try:
            config = load_live_config(repository / "configs/live" / filename)
            expected = "fp16_conv" if name == "fast_live_config" else "autocast"
            if config.crop_precision != expected:
                raise ValueError(f"Presentation profile {filename} must use {expected}")
            checks.append({"name": name, "status": "ready",
                           "crop_precision": config.crop_precision})
        except (OSError, ValueError, TypeError) as error:
            checks.append({"name": name, "status": "unavailable", "reason": str(error)})
    return {
        "status": "ready" if all(x["status"] == "ready" for x in checks) else "incomplete",
        "scope": "Local presentation files only; Unity reception/live performance not assessed.",
        "camera_started": False, "udp_sent": False, "models_loaded_on_cpu": load_models,
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--load-models", action="store_true", help="also run 60 CMU frames on CPU")
    args = parser.parse_args()
    report = check_demo(args.repository, load_models=args.load_models)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
