"""Decoder contracts and the bounded repetition-code reference decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from .types import Correction, DecodeResult, PauliFrame, SyndromeRound


def _repetition_wire(bits: tuple[int, ...]) -> int | None:
    if len(bits) != 2:
        raise ValueError("repetition syndrome must contain two binary values")
    return {
        (0, 0): None,
        (1, 0): 0,
        (1, 1): 1,
        (0, 1): 2,
    }[bits]


def _validate_repetition_history(
    syndrome_history: Sequence[SyndromeRound],
) -> tuple[SyndromeRound, ...]:
    history = tuple(syndrome_history)
    if not history:
        raise ValueError("repetition decoder requires syndrome history")
    if tuple(item.round_index for item in history) != tuple(range(len(history))):
        raise ValueError("syndrome history must be dense and ordered from zero")
    if any(len(item.bits) != 2 for item in history):
        raise ValueError("repetition syndrome must contain two binary values")
    return history


@runtime_checkable
class Decoder(Protocol):
    """Translate a complete syndrome history into a readout correction."""

    def decode(self, syndrome_history: Sequence[SyndromeRound]) -> DecodeResult: ...


@runtime_checkable
class StreamingDecoder(Protocol):
    """Select one bounded correction from the history available this round."""

    def decode_round(self, syndrome_history: Sequence[SyndromeRound]) -> Correction: ...


@dataclass(frozen=True)
class RepetitionLookupDecoder:
    """Decode the terminal adjacent-check syndrome of three data qubits."""

    def decode(self, syndrome_history: Sequence[SyndromeRound]) -> DecodeResult:
        history = _validate_repetition_history(syndrome_history)
        wire = _repetition_wire(history[-1].bits)
        correction = Correction(round_index=history[-1].round_index, wire=wire)
        corrections = (correction,)
        return DecodeResult(
            consumed_rounds=len(history),
            corrections=corrections,
            pauli_frame=PauliFrame.from_corrections(corrections),
        )


@dataclass(frozen=True)
class RepetitionStreamingLookupDecoder:
    """Select immediate repetition-code feedback from the latest syndrome."""

    def decode_round(self, syndrome_history: Sequence[SyndromeRound]) -> Correction:
        history = _validate_repetition_history(syndrome_history)
        wire = _repetition_wire(history[-1].bits)
        return Correction(round_index=history[-1].round_index, wire=wire)


@dataclass(frozen=True)
class RepetitionTemporalDecoder:
    """Confirm a persistent syndrome before selecting bounded feedback.

    One isolated readout fault produces paired detection events and is ignored.
    A data error is acted on after the same non-zero syndrome is observed for
    two consecutive rounds. Errors first seen in the terminal round therefore
    remain deliberately unconfirmed.
    """

    def decode_round(self, syndrome_history: Sequence[SyndromeRound]) -> Correction:
        history = _validate_repetition_history(syndrome_history)
        previous_bits = (0, 0)
        for record in history:
            expected_events = tuple(
                index
                for index, (before, after) in enumerate(zip(previous_bits, record.bits))
                if before != after
            )
            actual_events = tuple(
                event.check_index for event in record.detection_events
            )
            if actual_events != expected_events:
                raise ValueError(
                    "detection events are inconsistent with syndrome history"
                )
            previous_bits = record.bits
        latest = history[-1]
        wire = None
        if (
            len(history) >= 2
            and latest.bits != (0, 0)
            and latest.bits == history[-2].bits
            and not latest.detection_events
        ):
            wire = _repetition_wire(latest.bits)
        return Correction(round_index=latest.round_index, wire=wire)


__all__ = (
    "Decoder",
    "RepetitionLookupDecoder",
    "RepetitionStreamingLookupDecoder",
    "RepetitionTemporalDecoder",
    "StreamingDecoder",
)
