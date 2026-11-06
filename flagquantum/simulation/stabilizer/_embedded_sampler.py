"""Private adapter for the wheel's embedded Stim measurement engine."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .engine import _EngineOperation


class _EmbeddedSampler:
    def __init__(
        self,
        operations: tuple[_EngineOperation, ...],
        seed: int | None,
        columns: int,
    ) -> None:
        self._native = import_module(f"{__package__}._embedded_stim")
        lines = []
        for gate, wires, arguments in operations:
            if arguments is not None:
                raise ValueError("embedded measurement operations take no parameters")
            targets = " ".join(f"rec[{wire}]" if wire < 0 else str(wire) for wire in wires)
            lines.append(f"{gate} {targets}")
        self._handle = self._native.prepare("\n".join(lines), seed)
        self._columns = columns

    def sample(self, shots: int, *, bit_packed: bool) -> torch.Tensor:
        width = (self._columns + 7) // 8 if bit_packed else self._columns
        buffer = self._native.sample(self._handle, shots, bit_packed)
        dtype = torch.uint8 if bit_packed else torch.bool
        return torch.frombuffer(buffer, dtype=dtype).reshape(shots, width)

    def sample_write(self, shots: int, *, filepath: str, format: str) -> None:
        if format != "b8":
            raise ValueError("embedded measurement sampling writes only b8")
        self._native.write_b8(self._handle, shots, filepath)
