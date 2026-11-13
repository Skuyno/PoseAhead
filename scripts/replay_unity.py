"""Send a diagnostic pose sequence or existing recording to localhost Unity."""

import argparse
import json
import math

from poseahead.processing.config import create_processor, load_processor
from poseahead.recording import load_recording
from poseahead.runtime.unity_replay import diagnostic_sequence, stream_sequence
from poseahead.transport import UdpPoseSender


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--diagnostic", action="store_true", help="8-second synthetic holds")
    source.add_argument("--recording", help="existing NPZ recording; read-only")
    parser.add_argument("--processor-config", help="same causal processor config as other runners")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--port", type=int, default=5005)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("duration must be finite and positive")
    processor = (load_processor(args.processor_config) if args.processor_config
                 else create_processor("raw"))
    sequence = diagnostic_sequence() if args.diagnostic else load_recording(args.recording)
    with UdpPoseSender("127.0.0.1", args.port) as sender:
        result = stream_sequence(sequence, processor, sender, duration_seconds=args.duration)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
