"""Private adapter for the wheel's embedded Stim measurement engine."""

from __future__ import annotations

import ctypes
from importlib import import_module
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .engine import _EngineOperation


def _operations_text(operations: tuple[_EngineOperation, ...]) -> str:
    lines = []
    for gate, wires, arguments in operations:
        if arguments is not None:
            raise ValueError("embedded measurement operations take no parameters")
        targets = " ".join(f"rec[{wire}]" if wire < 0 else str(wire) for wire in wires)
        lines.append(f"{gate} {targets}")
    return "\n".join(lines)


class _EmbeddedSampler:
    def __init__(
        self,
        operations: tuple[_EngineOperation, ...],
        seed: int | None,
        columns: int,
    ) -> None:
        self._native = import_module(f"{__package__}._embedded_stim")
        self._handle = self._native.prepare(_operations_text(operations), seed)
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


class _EmbeddedRecordConverter:
    def __init__(self, text: str, detectors: int, observables: int) -> None:
        self._native = import_module(f"{__package__}._embedded_stim")
        self._handle = self._native.prepare_converter(text)
        self._widths = (detectors, observables)

    def convert(self, records: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        contiguous = records.contiguous()
        raw = ctypes.string_at(contiguous.data_ptr(), contiguous.numel())
        buffers = self._native.convert_records(self._handle, raw, len(records))
        if not isinstance(buffers, tuple) or len(buffers) != 2:
            raise ValueError("embedded conversion returned an invalid output pair")
        result = []
        for buffer, width in zip(buffers, self._widths, strict=True):
            if not isinstance(buffer, bytearray) or len(buffer) != len(records) * width:
                raise ValueError(
                    "embedded conversion returned an invalid output buffer"
                )
            if width:
                result.append(
                    torch.frombuffer(buffer, dtype=torch.bool).reshape(
                        len(records), width
                    )
                )
            else:
                result.append(torch.empty((len(records), 0), dtype=torch.bool))
        return result[0], result[1]


class _EmbeddedDetectorSampler:
    def __init__(
        self, text: str, seed: int | None, detectors: int, observables: int
    ) -> None:
        self._native = import_module(f"{__package__}._embedded_stim")
        self._handle = self._native.prepare_detector(text, seed)
        self._widths = (detectors, observables)

    def sample(self, shots: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self._native.sample_detectors(self._handle, shots)
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("embedded detector sampling returned an invalid pair")
        results = []
        for buffer, width in zip(pair, self._widths, strict=True):
            if not isinstance(buffer, bytearray) or len(buffer) != shots * width:
                raise ValueError("embedded detector sampling returned invalid bytes")
            results.append(
                torch.frombuffer(buffer, dtype=torch.bool).reshape(shots, width)
                if width
                else torch.empty((shots, 0), dtype=torch.bool)
            )
        return results[0], results[1]
