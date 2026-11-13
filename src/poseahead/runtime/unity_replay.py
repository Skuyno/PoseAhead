"""Display-only UDP replay: preserve research frames, rebase wire clocks to this boot."""

import math
import time
from dataclasses import replace

import numpy as np

from poseahead.core import PoseFrame, PoseSequence
from poseahead.recording import replay
from poseahead.transport.processor_packet import processor_packet_fields


def diagnostic_sequence():
    """Eight one-second diagnostic holds, NOT a motion/accuracy dataset."""
    display = np.array([
        [0, 0, 0], [0, .18, 0], [0, .4, 0], [0, .56, 0], [0, .72, 0],
        [.22, .52, 0], [.52, .52, 0], [.78, .52, 0],
        [-.22, .52, 0], [-.52, .52, 0], [-.78, .52, 0],
        [.1, 0, 0], [.1, -.45, 0], [.1, -.9, 0],
        [-.1, 0, 0], [-.1, -.45, 0], [-.1, -.9, 0],
    ])
    frames = []
    for i in range(160):
        phase = i // 20
        points, root = display.copy(), np.array([0., 0., 3.])
        valid = np.ones(17, dtype=np.bool_)
        if phase in (1, 2):
            points[6] = [.22, .82, 0]
            points[7] = [.22, 1.08, 0] if phase == 1 else [.22, .82, .26]
        elif phase == 3:
            points[12], points[13] = [.1, 0, .45], [.1, -.45, .45]
        elif phase == 4:
            points = points @ np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        elif phase == 5:
            root += [.3, -.15, -.2]
        elif phase == 6:
            valid[7] = False
        frames.append(PoseFrame(
            frame_id=i, source_timestamp_ns=i * 50_000_000,
            produced_timestamp_ns=i * 50_000_000, root_position=root,
            joint_positions=points * [1, -1, 1], joint_confidence=np.ones(17),
            joint_validity=valid, metadata={"origin": "synthetic_unity_diagnostic", "phase": phase},
        ))
    return PoseSequence("unity-diagnostic", "synthetic_20hz", tuple(frames))


def stream_sequence(sequence, processor, sender, *, duration_seconds=None,
                    clock_ns=time.perf_counter_ns, sleep=time.sleep,
                    stop_requested=None, on_progress=None):
    """Pace at recorded availability, process original times, rebase only outgoing copies.

    Unity receives display packets in this sender's clock domain. These rebased
    copies are not saved/evaluated as research recordings. No accelerated replay,
    loop, noise injection, VPN/interface change or model fitting occurs here.
    """
    if duration_seconds is not None and (
        type(duration_seconds) not in (int, float) or not math.isfinite(duration_seconds)
        or duration_seconds <= 0
    ):
        raise ValueError("duration must be finite and positive")
    origin = sequence.frames[0].produced_timestamp_ns
    if duration_seconds is not None:
        # Bound scheduled waits as well as work, even if the next source frame has a long gap.
        frames = tuple(f for f in sequence.frames
                       if f.produced_timestamp_ns - origin < duration_seconds * 1e9)
        sequence = replace(sequence, frames=frames)
    start = clock_ns()
    shift = start - origin
    if sequence.frames[0].source_timestamp_ns + shift < 0:
        raise ValueError("recorded production delay cannot be rebased to this monotonic clock")
    processor.reset()
    count = 0
    last_progress_ns = start - 500_000_000

    class Stopped(Exception):
        pass

    def interrupted_sleep(seconds):
        if stop_requested is not None and stop_requested():
            raise Stopped
        sleep(seconds)  # replay already limits each wait to 50 ms, including long recording gaps.

    try:
        for observation in replay(sequence, speed=1, clock_ns=clock_ns, sleep=interrupted_sleep):
            if stop_requested is not None and stop_requested():
                break
            if duration_seconds is not None and clock_ns() - start >= duration_seconds * 1e9:
                break
            processed = processor.process(observation)
            fields = processor_packet_fields(processor, processed)
            metadata = {**processed.metadata, "transport_clock_rebased": True,
                        "original_clock_domain": sequence.clock_domain,
                        "original_source_timestamp_ns": processed.source_timestamp_ns,
                        "original_produced_timestamp_ns": processed.produced_timestamp_ns}
            if "target_timestamp_ns" in metadata:
                metadata["target_timestamp_ns"] += shift
            wire = replace(processed, source_timestamp_ns=processed.source_timestamp_ns + shift,
                           produced_timestamp_ns=processed.produced_timestamp_ns + shift,
                           metadata=metadata)
            sender.send(wire, send_timestamp_ns=clock_ns(), **fields)
            count += 1
            now = clock_ns()
            if on_progress is not None and now - last_progress_ns >= 500_000_000:
                elapsed = (now - start) / 1e9
                on_progress({"sent_frames": count, "frame_id": observation.frame_id,
                             "elapsed_s": elapsed,
                             "udp_fps": count / elapsed if elapsed > 0 else None})
                last_progress_ns = now
    except Stopped:
        pass
    return {"sequence_id": sequence.sequence_id, "sent_frames": count,
            "processor": processor.method, "wire_clock": "current_sender_monotonic_rebased",
            "status": "stopped" if stop_requested is not None and stop_requested() else "completed",
            "scope": "display replay, not measured camera latency"}
