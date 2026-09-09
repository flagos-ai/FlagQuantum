"""Decoder contracts and the bounded repetition-code reference decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from .types import Correction, DecodeResult, PauliFrame, SyndromeRound


@runtime_checkable
class Decoder(Protocol):
    """Translate a complete syndrome history into a readout correction."""

    def decode(self, syndrome_history: Sequence[SyndromeRound]) -> DecodeResult: ...


@dataclass(frozen=True)
class RepetitionLookupDecoder:
    """Decode the terminal adjacent-check syndrome of three data qubits."""

    def decode(self, syndrome_history: Sequence[SyndromeRound]) -> DecodeResult:
        history = tuple(syndrome_history)
        if not history:
            raise ValueError("repetition decoder requires syndrome history")
        if tuple(item.round_index for item in history) != tuple(range(len(history))):
            raise ValueError("syndrome history must be dense and ordered from zero")
        bits = history[-1].bits
        if len(bits) != 2:
            raise ValueError("repetition syndrome must contain two binary values")
        wire = {
            (0, 0): None,
            (1, 0): 0,
            (1, 1): 1,
            (0, 1): 2,
        }[bits]
        correction = Correction(round_index=history[-1].round_index, wire=wire)
        corrections = (correction,)
        return DecodeResult(
            consumed_rounds=len(history),
            corrections=corrections,
            pauli_frame=PauliFrame.from_corrections(corrections),
        )


__all__ = ("Decoder", "RepetitionLookupDecoder")
