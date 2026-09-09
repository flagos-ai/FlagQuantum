"""Provider-neutral quantum-error-correction records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


def _binary_tuple(values: Sequence[int], *, owner: str) -> tuple[int, ...]:
    normalized = tuple(int(value) for value in values)
    if not normalized or any(value not in {0, 1} for value in normalized):
        raise ValueError(f"{owner} must contain binary values")
    return normalized


@dataclass(frozen=True, order=True)
class ErrorEvent:
    """One deterministic Pauli error injected before checks in a QEC round."""

    round_index: int
    wire: int
    pauli: str = "x"

    def __post_init__(self) -> None:
        if self.round_index < 0:
            raise ValueError("error-event round index must be non-negative")
        if self.wire not in {0, 1, 2}:
            raise ValueError("repetition error wire must be 0, 1, or 2")
        if self.pauli != "x":
            raise ValueError("the repetition reference profile supports Pauli X only")


@dataclass(frozen=True)
class ErrorSchedule:
    """Canonical bounded collection of deterministic error events."""

    events: tuple[ErrorEvent, ...] = ()

    def __post_init__(self) -> None:
        if any(not isinstance(event, ErrorEvent) for event in self.events):
            raise TypeError("error schedule entries must be ErrorEvent records")
        events = tuple(sorted(self.events))
        if len(events) > 64:
            raise ValueError("error schedule exceeds the 64-event reference limit")
        identities = tuple((event.round_index, event.wire) for event in events)
        if len(set(identities)) != len(identities):
            raise ValueError("an error schedule cannot repeat a round/wire event")
        object.__setattr__(self, "events", events)

    def for_round(self, round_index: int) -> tuple[ErrorEvent, ...]:
        return tuple(event for event in self.events if event.round_index == round_index)


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
    """One decoder or compiled-feedback correction decision."""

    round_index: int
    wire: int | None
    pauli: str = "x"

    def __post_init__(self) -> None:
        if self.round_index < 0:
            raise ValueError("correction round index must be non-negative")
        if self.wire is not None and self.wire not in {0, 1, 2}:
            raise ValueError("repetition correction wire must be 0, 1, 2, or None")
        if self.pauli != "x":
            raise ValueError("the repetition reference profile supports Pauli X only")

    @property
    def applied(self) -> bool:
        return self.wire is not None


@dataclass(frozen=True)
class PauliFrame:
    """Parity-reduced X corrections recommended at a readout boundary."""

    x_wires: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.x_wires))) != self.x_wires:
            raise ValueError("Pauli-frame X wires must be unique and ordered")
        if any(wire not in {0, 1, 2} for wire in self.x_wires):
            raise ValueError("Pauli-frame wire must be 0, 1, or 2")

    @classmethod
    def from_corrections(cls, corrections: Sequence[Correction]) -> "PauliFrame":
        parity: set[int] = set()
        for correction in corrections:
            if correction.wire is None:
                continue
            if correction.wire in parity:
                parity.remove(correction.wire)
            else:
                parity.add(correction.wire)
        return cls(tuple(sorted(parity)))

    def apply(self, bits: Sequence[int]) -> tuple[int, int, int]:
        normalized = _binary_tuple(bits, owner="Pauli-frame input bits")
        if len(normalized) != 3:
            raise ValueError("repetition Pauli frame requires three data bits")
        result = list(normalized)
        for wire in self.x_wires:
            result[wire] ^= 1
        return result[0], result[1], result[2]


@dataclass(frozen=True)
class DecodeResult:
    """A decoder's history-bounded correction and Pauli-frame recommendation."""

    consumed_rounds: int
    corrections: tuple[Correction, ...]
    pauli_frame: PauliFrame

    def __post_init__(self) -> None:
        if self.consumed_rounds <= 0:
            raise ValueError("decode result must consume at least one round")
        if any(
            correction.round_index >= self.consumed_rounds
            for correction in self.corrections
        ):
            raise ValueError("decoder correction is outside the consumed history")
        if PauliFrame.from_corrections(self.corrections) != self.pauli_frame:
            raise ValueError("decoder corrections and Pauli frame are inconsistent")


@dataclass(frozen=True)
class RepetitionMemoryShot:
    """Decoded evidence for one repetition-memory trajectory."""

    shot_index: int
    syndrome_rounds: tuple[SyndromeRound, ...]
    executed_feedback: tuple[Correction, ...]
    decode_result: DecodeResult
    raw_final_data_bits: tuple[int, int, int]
    decoded_data_bits: tuple[int, int, int]
    logical_bit: int
    logical_failure: bool

    def __post_init__(self) -> None:
        if self.shot_index < 0:
            raise ValueError("shot index must be non-negative")
        raw = _binary_tuple(self.raw_final_data_bits, owner="raw final data bits")
        decoded = _binary_tuple(self.decoded_data_bits, owner="decoded data bits")
        if len(raw) != 3 or len(decoded) != 3:
            raise ValueError("repetition memory requires three final data bits")
        object.__setattr__(self, "raw_final_data_bits", raw)
        object.__setattr__(self, "decoded_data_bits", decoded)
        if self.logical_bit not in {0, 1}:
            raise ValueError("logical bit must be binary")
        if self.logical_bit != int(sum(decoded) >= 2):
            raise ValueError("logical bit must be the majority of decoded data bits")
        if self.logical_failure is not bool(self.logical_bit):
            raise ValueError("reference memory starts in logical zero")
        if len(self.syndrome_rounds) != len(self.executed_feedback):
            raise ValueError("every syndrome round requires one feedback record")
        expected_rounds = tuple(range(len(self.syndrome_rounds)))
        if tuple(item.round_index for item in self.syndrome_rounds) != expected_rounds:
            raise ValueError("syndrome rounds must be dense and ordered from zero")
        if (
            tuple(item.round_index for item in self.executed_feedback)
            != expected_rounds
        ):
            raise ValueError("feedback rounds must align with syndrome rounds")
        if self.decode_result.consumed_rounds != len(self.syndrome_rounds):
            raise ValueError("decoder must consume the complete syndrome history")


@dataclass(frozen=True)
class RepetitionMemoryResult:
    """Shot-resolved result of one bounded repetition-code memory experiment."""

    rounds: int
    error_schedule: ErrorSchedule
    feedback_mode: str
    shot_records: tuple[RepetitionMemoryShot, ...]
    execution_semantics: str

    def __post_init__(self) -> None:
        if self.rounds <= 0:
            raise ValueError("rounds must be positive")
        if self.feedback_mode not in {"compiled_lookup", "offline_pauli_frame"}:
            raise ValueError("unsupported repetition feedback mode")
        if any(
            event.round_index >= self.rounds for event in self.error_schedule.events
        ):
            raise ValueError("error event is outside the configured rounds")
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
        if self.feedback_mode == "compiled_lookup" and any(
            shot.decoded_data_bits != shot.raw_final_data_bits
            for shot in self.shot_records
        ):
            raise ValueError("compiled feedback cannot apply an offline readout frame")
        if self.feedback_mode == "offline_pauli_frame" and any(
            any(item.applied for item in shot.executed_feedback)
            or shot.decoded_data_bits
            != shot.decode_result.pauli_frame.apply(shot.raw_final_data_bits)
            for shot in self.shot_records
        ):
            raise ValueError("offline mode must apply only the decoder readout frame")

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
    "DecodeResult",
    "DetectionEvent",
    "ErrorEvent",
    "ErrorSchedule",
    "PauliFrame",
    "RepetitionMemoryResult",
    "RepetitionMemoryShot",
    "SyndromeRound",
)
