"""Provider-neutral quantum-error-correction result records."""

from __future__ import annotations

from dataclasses import dataclass


def _binary_tuple(values: tuple[int, ...], *, owner: str) -> tuple[int, ...]:
    normalized = tuple(int(value) for value in values)
    if not normalized or any(value not in {0, 1} for value in normalized):
        raise ValueError(f"{owner} must contain binary values")
    return normalized


@dataclass(frozen=True)
class DetectionEvent:
    """One changed stabilizer-check outcome at a round boundary."""

    round_index: int
    check_index: int

    def __post_init__(self) -> None:
        if self.round_index < 0 or self.check_index < 0:
            raise ValueError("detection-event indices must be non-negative")


@dataclass(frozen=True)
class SyndromeRound:
    """Syndrome bits and derived detection events for one QEC round."""

    round_index: int
    bits: tuple[int, ...]
    detection_events: tuple[DetectionEvent, ...] = ()

    def __post_init__(self) -> None:
        if self.round_index < 0:
            raise ValueError("syndrome round index must be non-negative")
        object.__setattr__(
            self, "bits", _binary_tuple(self.bits, owner="syndrome bits")
        )
        if any(
            event.round_index != self.round_index for event in self.detection_events
        ):
            raise ValueError("detection events must belong to their syndrome round")
        check_indices = tuple(event.check_index for event in self.detection_events)
        if any(index >= len(self.bits) for index in check_indices):
            raise ValueError("detection-event check index exceeds syndrome width")
        if len(set(check_indices)) != len(check_indices):
            raise ValueError("a syndrome check may produce at most one detection event")


@dataclass(frozen=True)
class Correction:
    """One decoder decision for a repetition-code syndrome round."""

    round_index: int
    wire: int | None
    pauli: str = "x"

    def __post_init__(self) -> None:
        if self.round_index < 0:
            raise ValueError("correction round index must be non-negative")
        if self.wire is not None and self.wire < 0:
            raise ValueError("correction wire must be non-negative or None")
        if self.pauli != "x":
            raise ValueError("the repetition reference profile supports Pauli X only")

    @property
    def applied(self) -> bool:
        return self.wire is not None


@dataclass(frozen=True)
class RepetitionMemoryShot:
    """Decoded evidence for one repetition-memory trajectory."""

    shot_index: int
    syndrome_rounds: tuple[SyndromeRound, ...]
    corrections: tuple[Correction, ...]
    final_data_bits: tuple[int, int, int]
    logical_bit: int
    logical_failure: bool

    def __post_init__(self) -> None:
        if self.shot_index < 0:
            raise ValueError("shot index must be non-negative")
        final_data_bits = _binary_tuple(self.final_data_bits, owner="final data bits")
        if len(final_data_bits) != 3:
            raise ValueError("repetition memory requires three final data bits")
        object.__setattr__(self, "final_data_bits", final_data_bits)
        if self.logical_bit not in {0, 1}:
            raise ValueError("logical bit must be binary")
        if self.logical_bit != int(sum(final_data_bits) >= 2):
            raise ValueError("logical bit must be the majority of final data bits")
        if self.logical_failure is not bool(self.logical_bit):
            raise ValueError("reference memory starts in logical zero")
        if len(self.syndrome_rounds) != len(self.corrections):
            raise ValueError("every syndrome round requires one decoder decision")
        expected_rounds = tuple(range(len(self.syndrome_rounds)))
        if tuple(item.round_index for item in self.syndrome_rounds) != expected_rounds:
            raise ValueError("syndrome rounds must be dense and ordered from zero")
        if tuple(item.round_index for item in self.corrections) != expected_rounds:
            raise ValueError("correction rounds must align with syndrome rounds")


@dataclass(frozen=True)
class RepetitionMemoryResult:
    """Shot-resolved result of one bounded repetition-code memory experiment."""

    rounds: int
    injected_error_wire: int | None
    shot_records: tuple[RepetitionMemoryShot, ...]
    execution_semantics: str

    def __post_init__(self) -> None:
        if self.rounds <= 0:
            raise ValueError("rounds must be positive")
        if self.injected_error_wire not in {None, 0, 1, 2}:
            raise ValueError("injected error wire must be one data wire or None")
        if not self.shot_records:
            raise ValueError("memory result requires at least one shot")
        if any(len(shot.syndrome_rounds) != self.rounds for shot in self.shot_records):
            raise ValueError("shot syndrome history must match configured rounds")
        if tuple(shot.shot_index for shot in self.shot_records) != tuple(
            range(len(self.shot_records))
        ):
            raise ValueError("shot records must be dense and ordered from zero")
        if not self.execution_semantics:
            raise ValueError("execution semantics must be recorded")

    @property
    def shot_count(self) -> int:
        return len(self.shot_records)

    @property
    def logical_failures(self) -> int:
        return sum(shot.logical_failure for shot in self.shot_records)

    @property
    def logical_error_rate(self) -> float:
        return self.logical_failures / self.shot_count


__all__ = (
    "Correction",
    "DetectionEvent",
    "RepetitionMemoryResult",
    "RepetitionMemoryShot",
    "SyndromeRound",
)
