"""Package-owned live probe with validity, recording and capture-to-send evidence."""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import ExitStack
from pathlib import Path

from poseahead.config import LiveConfig, load_live_config
from poseahead.perception.metrabs import MetrabsBackend, MetrabsStartupCancelled
from poseahead.processing.config import METHODS, create_processor, load_processor
from poseahead.recording import save_recording
from poseahead.recording.capture import ObservationRecorder
from poseahead.runtime import LatestFrameCamera
from poseahead.runtime.benchmark import BenchmarkStats
from poseahead.runtime.memory import memory_summary, working_set_mib
from poseahead.runtime.pipeline_timing import PipelineTiming
from poseahead.transport import UdpPoseSender
from poseahead.transport.processor_packet import processor_packet_fields


def main(argv=None, *, stop_requested=None, on_progress=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--camera", type=int)
    parser.add_argument("--udp-port", type=int)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--send-udp", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--record-output", type=Path)
    parser.add_argument("--record-max-frames", type=int, default=100_000)
    methods = parser.add_mutually_exclusive_group()
    methods.add_argument("--processor", choices=METHODS, default="raw")
    methods.add_argument("--processor-config", type=Path)
    parser.add_argument("--time-constant-ms", type=float, default=50)
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration) or args.duration <= 0 or args.warmup < 0:
        parser.error("duration must be finite and positive and warmup non-negative")
    values = (load_live_config(args.config) if args.config else LiveConfig()).to_dict()
    if args.camera is not None:
        values["camera_index"] = args.camera
    if args.udp_port is not None:
        values["udp_port"] = args.udp_port
    config = LiveConfig(**values)
    processor = (
        load_processor(args.processor_config)
        if args.processor_config
        else create_processor(
            args.processor,
            {"time_constant_ms": args.time_constant_ms} if args.processor == "ema" else {},
        )
    )
    if processor.method in ("learned", "adaptive") and processor.history_sampling != "linear_past":
        parser.error("live learned prediction requires explicit linear_past history sampling")
    target_offset_ns = getattr(processor, "target_offset_ns", 0)
    if args.send_udp and target_offset_ns % 1_000_000:
        parser.error("UDP v2 requires a whole-millisecond prediction horizon")
    if args.record_output is not None and args.record_output.exists():
        parser.error("record-output already exists; choose a new recording path")
    recorder = (
        ObservationRecorder(
            sequence_id=f"webcam-{time.perf_counter_ns()}",
            clock_domain="producer_monotonic",
            max_frames=args.record_max_frames,
        )
        if args.record_output is not None
        else None
    )

    def stopped():
        return stop_requested is not None and stop_requested()

    def notify(kind, **fields):
        if on_progress is not None:
            on_progress({"kind": kind, **fields})

    def cancelled_before_measurement():
        notify(
            "finished",
            report={
                "status": "stopped_before_measurement",
                "raw_pose_recording": None,
                "acceptance_status": "not_assessed",
            },
        )
        return 0

    if stopped():
        return cancelled_before_measurement()
    notify("state", state="loading", message="Загрузка библиотеки PyTorch")
    import torch

    if stopped():
        return cancelled_before_measurement()
    try:
        backend = MetrabsBackend(
            confidence_threshold=config.confidence_threshold,
            num_aug=config.num_aug,
            internal_batch_size=config.internal_batch_size,
            detector_interval=config.detector_interval,
            max_input_side=config.max_input_side,
            default_fov_degrees=config.default_fov_degrees,
            camera_intrinsics=config.camera_intrinsics,
            crop_precision=config.crop_precision,
            on_startup=lambda message: notify("state", state="loading", message=message),
            stop_requested=stopped,
        )
    except MetrabsStartupCancelled:
        return cancelled_before_measurement()
    if stopped():
        return cancelled_before_measurement()
    stats = BenchmarkStats()
    pipeline_timing = PipelineTiming()
    prediction_status_counts = {}
    memory_samples = []
    host_samples = []
    with ExitStack() as stack:
        if args.send_udp:
            notify("state", state="loading", message="Подготовка отправки UDP в Unity")
        if stopped():
            return cancelled_before_measurement()
        sender = (
            stack.enter_context(UdpPoseSender(config.udp_host, config.udp_port))
            if args.send_udp
            else None
        )
        notify("state", state="loading", message=f"Открытие камеры {config.camera_index}")
        if stopped():
            return cancelled_before_measurement()
        camera = stack.enter_context(LatestFrameCamera(config.camera_index))
        for index in range(args.warmup):
            if stopped():
                return cancelled_before_measurement()
            warmup = f"Прогрев: кадр {index + 1}/{args.warmup}"
            notify("state", state="warmup", message=warmup + " — ожидание кадра камеры")
            if stopped():
                return cancelled_before_measurement()
            captured = camera.get(timeout=5.0)
            if stopped():
                return cancelled_before_measurement()
            if captured is None:
                raise RuntimeError("camera delivered no frame during warmup")
            notify("state", state="warmup", message=warmup + " — оценка позы")
            if stopped():
                return cancelled_before_measurement()
            backend.infer(
                captured.image,
                frame_id=captured.frame_id,
                source_timestamp_ns=captured.source_timestamp_ns,
            )
        if stopped():
            return cancelled_before_measurement()
        # Establish an accounting boundary under the capture lock and discard warm-up leftovers.
        start_counts = camera.discard_pending_and_snapshot()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()

        def sample_memory(force=False):
            seconds = time.perf_counter() - started
            if force or not memory_samples or seconds - memory_samples[-1][0] >= 1:
                memory_samples.append(
                    [
                        seconds,
                        torch.cuda.memory_allocated() / 1024**2,
                        torch.cuda.memory_reserved() / 1024**2,
                    ]
                )
                host_samples.append([seconds, working_set_mib()])

        sample_memory()
        last_progress = started - 1
        notify("state", state="running", message="Камера работает; UDP без подтверждения приёма")
        while time.perf_counter() - started < args.duration and not stopped():
            captured = camera.get(timeout=min(1.0, args.duration))
            sample_memory()
            if stopped():
                break
            if captured is None:
                continue
            inference_start_ns = time.perf_counter_ns()
            result = backend.infer(
                captured.image,
                frame_id=captured.frame_id,
                source_timestamp_ns=captured.source_timestamp_ns,
            )
            completed_ns = time.perf_counter_ns()
            inference_end_ns = completed_ns
            if recorder is not None:
                recorder.append(
                    result,
                    frame_id=captured.frame_id,
                    source_timestamp_ns=captured.source_timestamp_ns,
                    completed_timestamp_ns=completed_ns,
                )
            if result is None:
                processor.reset()
            else:
                result = processor.process(result)
                if processor.method in ("learned", "adaptive"):
                    status = result.metadata["prediction_status"]
                    prediction_status_counts[status] = prediction_status_counts.get(status, 0) + 1
            completed_ns = time.perf_counter_ns()
            valid = stats.observe(result, (completed_ns - captured.source_timestamp_ns) / 1e6)
            send_start_ns = send_end_ns = None
            if valid and sender is not None:
                send_start_ns = time.perf_counter_ns()
                sender.send(result, **processor_packet_fields(processor, result))
                send_end_ns = time.perf_counter_ns()
                stats.send_latency_ms.append(
                    (send_end_ns - captured.source_timestamp_ns) / 1e6
                )
            pipeline_timing.observe(
                source_ns=captured.source_timestamp_ns, inference_start_ns=inference_start_ns,
                inference_end_ns=inference_end_ns, processed_ns=completed_ns,
                send_start_ns=send_start_ns, send_end_ns=send_end_ns,
            )
            sample_memory()
            now = time.perf_counter()
            if on_progress is not None and now - last_progress >= 0.5:
                progress = stats.report(max(now - started, 1e-9))
                notify(
                    "progress",
                    **progress,
                    frame_id=captured.frame_id,
                    sent_frames=len(stats.send_latency_ms),
                    sender_session_id=str(sender.session_id) if sender is not None else None,
                    pipeline_latency_ms=pipeline_timing.report(),
                    udp_fps=len(stats.send_latency_ms) / max(now - started, 1e-9),
                )
                last_progress = now
        elapsed = time.perf_counter() - started
        sample_memory(force=True)
        end_counts = camera.snapshot()
    if recorder is not None and stats.latency_ms:
        save_recording(args.record_output, recorder.sequence())
    report = stats.report(elapsed)
    report.update(
        {
            "resolved_config": config.to_dict(),
            "warmup_frames": args.warmup,
            "captured_frames": end_counts[0] - start_counts[0],
            "overwritten_camera_frames": end_counts[1] - start_counts[1],
            "pending_camera_frames": end_counts[2],
            "udp_enabled": args.send_udp,
            "sender_session_id": str(sender.session_id) if sender is not None else None,
            "sent_frames": len(stats.send_latency_ms),
            "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "gpu_samples_seconds_allocated_mib_reserved_mib": memory_samples,
            "host_memory": memory_summary(host_samples),
            "gpu_reserved_growth_mib": memory_samples[-1][2] - memory_samples[0][2],
            "gpu_reserved_stability_status": (
                "insufficient_duration"
                if elapsed < 590
                else "passed"
                if memory_samples[-1][2] - memory_samples[0][2] <= 32
                else "failed"
            ),
            "memory_stability_status": "see_host_memory_and_gpu_reserved_stability_status",
            "latency_scope": "capture_available_to_pose_return_or_local_send_completion",
            "camera_images_stored": False,
            "processor": processor.method,
            "processor_parameters": getattr(processor, "parameters", {}),
            "prediction_status_counts": prediction_status_counts,
            "pipeline_latency_ms": pipeline_timing.report(),
            "raw_pose_recording": (
                str(args.record_output) if args.record_output and stats.latency_ms else None
            ),
            "status": "stopped" if stopped() else "completed",
        }
    )
    rendered = json.dumps(report, indent=2, allow_nan=False)
    if on_progress is None:
        print(rendered)
    else:
        notify("finished", report=report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
