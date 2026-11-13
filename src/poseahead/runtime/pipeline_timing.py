"""Host-side stage timing, separate from effective latency or capture-to-screen."""

from dataclasses import dataclass, field

from poseahead.runtime.benchmark import latency_summary


@dataclass
class PipelineTiming:
    frame_age: list[float] = field(default_factory=list)
    backend: list[float] = field(default_factory=list)
    processing: list[float] = field(default_factory=list)
    bookkeeping: list[float] = field(default_factory=list)
    send: list[float] = field(default_factory=list)
    total_send: list[float] = field(default_factory=list)

    def observe(self, *, source_ns, inference_start_ns, inference_end_ns, processed_ns,
                send_start_ns=None, send_end_ns=None):
        if (send_start_ns is None) != (send_end_ns is None):
            raise ValueError("both send timestamps must be provided together")
        times = (source_ns, inference_start_ns, inference_end_ns, processed_ns)
        if send_start_ns is not None:
            times += (send_start_ns, send_end_ns)
        if (any(type(t) is not int or t < 0 for t in times)
                or any(a > b for a, b in zip(times, times[1:], strict=False))):
            raise ValueError("pipeline timestamps must be ordered nonnegative integer nanoseconds")
        self.frame_age.append((inference_start_ns - source_ns) / 1e6)
        self.backend.append((inference_end_ns - inference_start_ns) / 1e6)
        self.processing.append((processed_ns - inference_end_ns) / 1e6)
        if send_start_ns is not None:
            self.bookkeeping.append((send_start_ns - processed_ns) / 1e6)
            self.send.append((send_end_ns - send_start_ns) / 1e6)
            self.total_send.append((send_end_ns - source_ns) / 1e6)

    def report(self):
        return {
            "frame_age_before_inference_ms": latency_summary(self.frame_age),
            "backend_return_ms": latency_summary(self.backend),
            "recording_and_processing_ms": latency_summary(self.processing),
            "validity_bookkeeping_before_send_ms": latency_summary(self.bookkeeping),
            "packet_and_send_ms": latency_summary(self.send),
            "capture_available_to_send_complete_ms": latency_summary(self.total_send),
            "scope": "host wall time; frame age includes waiting before inference; "
                     "backend includes preprocessing/GPU/postprocessing; not capture-to-screen",
            "aggregation": "each stage has its own sample set; do not sum stage percentiles",
        }
