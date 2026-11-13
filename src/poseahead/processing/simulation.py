"""Evaluate causal baseline state at explicit times without reading future poses."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from poseahead.core import PoseFrame, PoseSequence
from poseahead.processing.filters import PoseProcessor
from poseahead.recording.delay import DelayedReplay


@dataclass(frozen=True, slots=True)
class ProcessingSnapshot:
    evaluation_timestamp_ns: int
    clock_domain: str
    pose: PoseFrame | None
    observations_consumed: int

    @property
    def observation_age_ns(self) -> int | None:
        if self.pose is None:
            return None
        return self.evaluation_timestamp_ns - self.pose.source_timestamp_ns

    @property
    def represented_timestamp_ns(self) -> int | None:
        if self.pose is None:
            return None
        return self.pose.metadata.get("target_timestamp_ns", self.pose.source_timestamp_ns)


def simulate_processing(
    sequence: PoseSequence, processor: PoseProcessor, evaluation_times_ns: Iterable[int],
    *, delay_ns: int = 0,
) -> Iterator[ProcessingSnapshot]:
    """Run a fresh processor over observations available at each evaluation time.

    Evaluation times use the sequence's event clock, never a host wall clock.
    Each observation is processed exactly once, even when queries repeat or skip
    multiple observations. Between arrivals the most recent output is held.
    Missing/invalid outputs remain missing/invalid: this is not a renderer's
    tracking-loss fallback. There is no automatic extrapolation to query time;
    configured prediction horizons retain their explicit target metadata.
    Compute duration is not simulated. Do not share the processor concurrently.
    """
    delayed = DelayedReplay(sequence, delay_ns=delay_ns)
    processor.reset()
    latest = None
    consumed = 0
    for evaluation in evaluation_times_ns:
        for observation in delayed.advance(evaluation):
            latest = processor.process(observation)
            consumed += 1
        yield ProcessingSnapshot(evaluation, sequence.clock_domain, latest, consumed)
