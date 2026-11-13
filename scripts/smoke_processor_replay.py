"""Offline detector-recording compatibility: no camera, sockets, Unity or ground-truth claims."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from poseahead.artifacts import sha256_file
from poseahead.core import PoseSequence
from poseahead.processing.config import create_processor, read_processor_config
from poseahead.recording import load_recording, save_recording
from poseahead.transport.processor_packet import processor_packet_fields
from poseahead.transport.udp_protocol import decode_udp_packet, encode_udp_packet


def smoke_replay(recording, processor_config, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("offline smoke output already exists")
    input_hash = sha256_file(recording)
    sequence = load_recording(recording)
    config = read_processor_config(processor_config)
    processor = create_processor(config["method"], config["parameters"])
    processed, durations = [], []
    algorithms = {}
    for frame in sequence.frames:
        start = time.perf_counter_ns()
        result = processor.process(frame)
        durations.append((time.perf_counter_ns() - start) / 1e6)
        packet = decode_udp_packet(encode_udp_packet(
            result, **processor_packet_fields(processor, result),
            send_timestamp_ns=result.produced_timestamp_ns,
        ))
        if (packet.frame.source_timestamp_ns != frame.source_timestamp_ns
                or packet.target_offset_ms * 1_000_000 != result.metadata.get(
                    "target_offset_ns", 0)):
            raise ValueError("offline UDP event-clock/target mismatch")
        name = packet.algorithm.name
        algorithms[name] = algorithms.get(name, 0) + 1
        processed.append(result)
    if sha256_file(recording) != input_hash:
        raise ValueError("recording changed during offline smoke")
    report = {
        "recording_sha256": input_hash, "processor_config_sha256": sha256_file(processor_config),
        "processor": {"method": processor.method,
                      "parameters": getattr(processor, "parameters", config["parameters"])},
        "sequence_id": sequence.sequence_id, "clock_domain": sequence.clock_domain,
        "input_frames": len(sequence.frames), "input_valid_frames": sum(f.is_valid
                                                                        for f in sequence.frames),
        "output_valid_frames": sum(f.is_valid for f in processed), "udp_algorithms": algorithms,
        "processor_p50_ms": float(np.percentile(durations, 50)),
        "processor_p95_ms": float(np.percentile(durations, 95)),
        "scope": "offline_processor_only_not_detector_camera_render_or_end_to_end",
        "accuracy_assessment": "not_scored_no_verified_ground_truth",
        "network": "encode_decode_only_no_sockets_no_packets_sent",
    }
    output.mkdir(parents=True, exist_ok=False)
    save_recording(output / "processed.npz", PoseSequence(sequence.sequence_id + "/smoke",
                                                         sequence.clock_domain, tuple(processed)))
    report["processed_sha256"] = sha256_file(output / "processed.npz")
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("processor_config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    config = read_processor_config(args.processor_config)
    if config["method"] in ("adaptive", "learned"):
        import torch

        torch.set_num_threads(2)
    report = smoke_replay(args.recording, args.processor_config, args.output)
    print(json.dumps({k: report[k] for k in (
        "input_frames", "output_valid_frames", "udp_algorithms", "processor_p95_ms",
        "accuracy_assessment", "network",
    )}))


if __name__ == "__main__":
    main()
