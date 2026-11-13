"""Private producer subprocess for the UI; stdout event protocol, stdin graceful stop."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from poseahead.desktop.commands import listen_for_stop
from poseahead.desktop.session import EVENT_PREFIX, SessionOptions
from poseahead.processing.config import load_processor, read_processor_config
from poseahead.recording import load_recording
from poseahead.runtime.run_report import assemble_run_report, read_unity_snapshot
from poseahead.runtime.unity_replay import diagnostic_sequence, stream_sequence
from poseahead.runtime.webcam import main as run_webcam_benchmark
from poseahead.transport import UdpPoseSender


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def run_session(options, repository, output, stop, emit):
    options.validate()
    started_unix_ms = time.time() * 1000
    write_json(output / "request.json", asdict(options))
    emit({"kind": "state", "state": "loading", "message": "Подготовка источника и процессора"})
    if options.method == "config":
        values = read_processor_config(options.processor_config)
        processor_path = output / "processor.json"
        write_json(processor_path, values)
    else:
        parameters = (
            {"target_offset_ms": options.horizon_ms} if options.method == "kalman_cv" else {}
        )
        processor_path = output / "processor.json"
        write_json(processor_path, {"method": options.method, "parameters": parameters})

    report = None
    snapshot_path = (
        repository / "unity/PoseAheadAvatar/Temp" / f"PoseAheadTelemetry-{options.udp_port}.json"
    )

    def unity_snapshot(session_id, sent_frames):
        return read_unity_snapshot(
            snapshot_path, session_id, not_before_unix_ms=started_unix_ms,
            expected_udp_port=options.udp_port, sent_frames=sent_frames,
        )

    def progress(event):
        nonlocal report
        if event["kind"] == "finished":
            report = event["report"]
        else:
            if event["kind"] == "progress":
                event = {**event, "unity": unity_snapshot(
                    event.get("sender_session_id"), event.get("sent_frames"),
                )}
            emit(event)

    if options.source == "camera":
        args = [
            "--config",
            options.live_config,
            "--camera",
            str(options.camera_index),
            "--udp-port",
            str(options.udp_port),
            "--duration",
            str(options.duration_seconds),
            "--send-udp",
            "--processor-config",
            str(processor_path),
        ]
        if options.record_raw:
            args += ["--record-output", str(output / "observations.npz")]
        run_webcam_benchmark(args, stop_requested=stop.is_set, on_progress=progress)
    else:
        processor = load_processor(processor_path)
        sequence = (
            diagnostic_sequence()
            if options.source == "diagnostic"
            else load_recording(options.recording)
        )
        emit({"kind": "state", "state": "running", "message": "Отправка движения в Unity"})
        with UdpPoseSender("127.0.0.1", options.udp_port) as sender:
            report = stream_sequence(
                sequence,
                processor,
                sender,
                duration_seconds=options.duration_seconds,
                stop_requested=stop.is_set,
                on_progress=lambda event: progress({
                    "kind": "progress", **event, "sender_session_id": str(sender.session_id),
                }),
            )
            report["sender_session_id"] = str(sender.session_id)
        report["processor_parameters"] = getattr(processor, "parameters", {})
    if report is None:
        raise RuntimeError("Источник завершился без итогового отчёта.")
    write_json(output / "report.json", report)
    # Allow the next local telemetry flush after the final datagram; no camera is restarted.
    deadline = time.monotonic() + 0.65
    unity = unity_snapshot(report.get("sender_session_id"), report.get("sent_frames"))
    while time.monotonic() < deadline and not stop.is_set():
        time.sleep(0.05)
        unity = unity_snapshot(report.get("sender_session_id"), report.get("sent_frames"))
    combined = assemble_run_report(
        report, unity, source=options.source, requested_duration_s=options.duration_seconds
    )
    write_json(output / "run_report.json", combined)
    emit(
        {
            "kind": "finished",
            "report": report,
            "run_report": combined,
            "output_directory": str(output),
        }
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options_input = parser.add_mutually_exclusive_group(required=True)
    options_input.add_argument("--options")
    options_input.add_argument("--options-file", type=Path)
    args = parser.parse_args(argv)
    stop = threading.Event()
    finished = threading.Event()

    def commands():
        try:
            listen_for_stop(sys.stdin, stop, finished)
        except (OSError, ValueError) as error:
            stop.set()
            emit({"kind": "log", "message": "Ошибка канала остановки: " + str(error)})

    def emit(event):
        print(EVENT_PREFIX + json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)

    command_thread = threading.Thread(
        target=commands, daemon=True, name="poseahead-ui-commands"
    )
    command_thread.start()
    try:
        raw_options = (
            args.options_file.read_text(encoding="utf-8-sig")
            if args.options_file is not None
            else args.options
        )
        options = SessionOptions(**json.loads(raw_options))
        run_session(options, args.repository.resolve(), args.output.resolve(), stop, emit)
    except Exception as error:
        traceback.print_exc()
        emit({"kind": "error", "message": str(error)})
        return 1
    finally:
        finished.set()
        command_thread.join(timeout=0.5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
