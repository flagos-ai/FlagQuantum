"""Phase-aware no-progress diagnosis independent of process launchers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProgressSnapshot:
    timestamp: float
    phase: str
    operation: str
    completed_units: int
    memory_bytes: int
    collective: str = "none"
    rank: int = 0
    useful_work_launched: bool = True


@dataclass(frozen=True)
class StallDiagnosis:
    cause: str
    phase: str
    last_operation: str
    collective: str
    memory_trend_bytes: int
    rank: int
    cleanup_required: bool = True


class PhaseAwareWatchdog:
    """Classify genuine stalls while allowing bounded compile/checkpoint phases."""

    def __init__(
        self,
        *,
        stall_seconds: float,
        bounded_phase_seconds: dict[str, float] | None = None,
    ) -> None:
        if stall_seconds <= 0:
            raise ValueError("stall_seconds must be positive")
        self.stall_seconds = stall_seconds
        self.bounded = dict(bounded_phase_seconds or {})
        self._first: ProgressSnapshot | None = None
        self._last: ProgressSnapshot | None = None

    def observe(self, snapshot: ProgressSnapshot) -> StallDiagnosis | None:
        last = self._last
        if last is None or snapshot.completed_units > last.completed_units:
            self._first = snapshot
            self._last = snapshot
            return None
        self._last = snapshot
        first = self._first or last
        elapsed = snapshot.timestamp - first.timestamp
        allowed = max(self.stall_seconds, self.bounded.get(snapshot.phase, 0.0))
        if elapsed <= allowed:
            return None
        memory_trend = snapshot.memory_bytes - first.memory_bytes
        if not snapshot.useful_work_launched and snapshot.memory_bytes > 0:
            cause = "memory_owner_without_useful_work"
        elif snapshot.collective != "none":
            cause = "collective_participant_or_transport_stall"
        elif memory_trend > 0:
            cause = "memory_growth_without_progress"
        elif snapshot.phase == "input":
            cause = "stalled_input"
        else:
            cause = "rank_desynchronization_or_no_progress"
        return StallDiagnosis(
            cause=cause,
            phase=snapshot.phase,
            last_operation=snapshot.operation,
            collective=snapshot.collective,
            memory_trend_bytes=memory_trend,
            rank=snapshot.rank,
        )
