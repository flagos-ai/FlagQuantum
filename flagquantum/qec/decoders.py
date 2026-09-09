"""Decoder contracts and the bounded repetition-code reference decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from .types import Correction


@runtime_checkable
class Decoder(Protocol):
    """Translate one syndrome into a correction decision."""

    def decode(self, syndrome: Sequence[int], *, round_index: int) -> Correction: ...


@dataclass(frozen=True)
class RepetitionLookupDecoder:
    """Exact lookup decoder for adjacent checks of three data qubits."""

    def decode(self, syndrome: Sequence[int], *, round_index: int) -> Correction:
        bits = tuple(int(value) for value in syndrome)
        if len(bits) != 2 or any(value not in {0, 1} for value in bits):
            raise ValueError("repetition syndrome must contain two binary values")
        wire = {
            (0, 0): None,
            (1, 0): 0,
            (1, 1): 1,
            (0, 1): 2,
        }[bits]
        return Correction(round_index=round_index, wire=wire)


__all__ = ("Decoder", "RepetitionLookupDecoder")
