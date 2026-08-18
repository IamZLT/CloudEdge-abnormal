"""Discrete-event cloud-load simulator for load-dependent CRR benchmarking.

The cloud reviewer is modelled as a c-server queue (``c = max_inflight``) with a
per-request service time (the measured / configured cloud inference latency).
Requests arrive on a virtual timeline driven by edge processing; CRR reads the
instantaneous ``inflight + queue`` and charges ``-w_c * c_cloud`` in its utility,
so it sheds load as the cloud saturates.

Why this exists: a sequential for-loop makes ``inflight ≈ 0`` forever (each
upload resolves immediately), so the ``-w_c * c_cloud`` term in the cost–risk
router is never exercised. Driving ``CloudState`` from this simulator produces a
real congestion trajectory so load-awareness is actually measurable.

Timeline model: ``advance(now)`` / ``submit(now)`` are called with monotonically
non-decreasing ``now`` in small steps (edge inter-arrival time). Service
completions are tracked on a min-heap; a queued job that can be promoted starts
at the current clock (approximating ``max(arrival, server_free_time)``, which is
exact up to one inter-arrival step — negligible vs. service time).
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from src.collab_routing.base import CloudState


@dataclass
class CloudLoadSim:
    max_inflight: int = 2
    service_ms: float = 3000.0

    def __post_init__(self) -> None:
        self.clock: float = 0.0
        self._busy_until: list[float] = []  # min-heap of server busy-until times
        self._waiting: list[float] = []     # FIFO arrival times of queued jobs
        self.n_submitted: int = 0
        self.n_completed: int = 0
        self.n_queued: int = 0
        self.max_queue_seen: int = 0
        self.total_wait_ms: float = 0.0

    def advance(self, now: float) -> None:
        """Advance the virtual clock to ``now`` (retire completions, promote queue)."""
        if now <= self.clock:
            return
        self.clock = now
        # retire servers finished by now
        while self._busy_until and self._busy_until[0] <= now:
            heapq.heappop(self._busy_until)
            self.n_completed += 1
        # promote waiting jobs onto freed slots
        while self._waiting and len(self._busy_until) < self.max_inflight:
            self._waiting.pop(0)
            heapq.heappush(self._busy_until, now + self.service_ms)
        self.max_queue_seen = max(self.max_queue_seen, len(self._waiting))

    @property
    def inflight(self) -> int:
        return len(self._busy_until)

    @property
    def queue(self) -> int:
        return len(self._waiting)

    @property
    def load(self) -> int:
        return self.inflight + self.queue

    def submit(self, now: float) -> float:
        """A cloud-review request arrives at ``now``. Returns estimated queue wait (ms)."""
        self.advance(now)
        self.n_submitted += 1
        if len(self._busy_until) < self.max_inflight:
            heapq.heappush(self._busy_until, now + self.service_ms)
            return 0.0
        self._waiting.append(now)
        self.n_queued += 1
        wait = (len(self._waiting) / max(self.max_inflight, 1)) * self.service_ms
        self.total_wait_ms += wait
        self.max_queue_seen = max(self.max_queue_seen, len(self._waiting))
        return wait

    def state(self) -> CloudState:
        return CloudState(inflight=self.inflight, queue=self.queue, max_inflight=self.max_inflight)

    def summary(self) -> dict:
        return {
            "n_submitted": self.n_submitted,
            "n_completed": self.n_completed,
            "n_queued": self.n_queued,
            "max_queue_seen": self.max_queue_seen,
            "total_wait_ms": self.total_wait_ms,
            "final_clock_ms": self.clock,
        }
