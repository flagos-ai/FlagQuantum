"""Private adapter for the wheel's embedded Stim measurement engine."""

from __future__ import annotations

import ctypes
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

import torch

from ...errors import CapabilityError
from ._validation import _validate_seed, _validate_stim_experiment_source

if TYPE_CHECKING:
    from .engine import _EngineOperation


def _operations_text(operations: tuple[_EngineOperation, ...]) -> str:
    lines = []
    for gate, targets, arguments in operations:
        if arguments is not None:
            values = arguments if isinstance(arguments, tuple) else (arguments,)
            gate += f"({','.join(map(repr, values))})"
        encoded = []
        for target in targets:
            if isinstance(target, tuple):
                wire, pauli = target
                encoded.append(f"{pauli}{wire}")
            else:
                encoded.append(f"rec[{target}]" if target < 0 else str(target))
        lines.append(f"{gate} {' '.join(encoded)}")
    return "\n".join(lines)


def _annotated_operations_text(
    operations: tuple[_EngineOperation, ...],
    terminal_qubits: tuple[int, ...],
    detector_columns: tuple[tuple[int, ...], ...],
    observable_columns: tuple[tuple[int, ...], ...],
) -> str:
    if terminal_qubits:
        operations += (("M", terminal_qubits, None),)
    recorded = sum(len(targets) for name, targets, _ in operations if name == "M")
    lines = [_operations_text(operations)]
    for kind, columns in (
        ("DETECTOR", detector_columns),
        ("OBSERVABLE_INCLUDE", observable_columns),
    ):
        for index, parity in enumerate(columns):
            if any(column < 0 or column >= recorded for column in parity):
                raise ValueError("parity column outside recorded measurements")
            name = kind if kind == "DETECTOR" else f"{kind}({index})"
            targets = " ".join(f"rec[{column - recorded}]" for column in parity)
            lines.append(f"{name} {targets}".rstrip())
    return "\n".join(lines)


def _validate_text_experiment(text: str) -> None:
    _validate_stim_experiment_source(text)
    try:
        _text_experiment_engine().validate_text_experiment(text)
    except OverflowError as error:
        raise CapabilityError(str(error)) from error


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

    def sample(self, shots: int, *, bit_packed: bool = False) -> torch.Tensor:
        width = (self._columns + 7) // 8 if bit_packed else self._columns
        buffer = self._native.sample(self._handle, shots, bit_packed)
        dtype = torch.uint8 if bit_packed else torch.bool
        return torch.frombuffer(buffer, dtype=dtype).reshape(shots, width)

    def sample_write(self, shots: int, *, filepath: str, format: str) -> None:
        if format == "b8":
            self._native.write_b8(self._handle, shots, filepath)
        else:
            self._native.write_records(self._handle, shots, filepath, format)


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

    def convert_file(
        self,
        source: str,
        input_format: str,
        target: str,
        output_format: str,
        append: bool,
        observable: str | None,
        obs_format: str,
    ) -> None:
        self._native.convert_file(
            self._handle,
            source,
            input_format,
            target,
            output_format,
            append,
            observable,
            obs_format,
        )


class _EmbeddedDetectorSampler:
    def __init__(
        self, text: str, seed: int | None, detectors: int, observables: int
    ) -> None:
        self._native = import_module(f"{__package__}._embedded_stim")
        self._handle = self._native.prepare_detector(text, seed)
        self._widths = (detectors, observables)

    def sample(self, shots: int) -> tuple[torch.Tensor, torch.Tensor]:
        return _sample_detector_pair(self._native, self._handle, shots, self._widths)

    def sample_write(
        self,
        shots: int,
        *,
        filepath: str,
        format: str,
        prepend_observables: bool,
        append_observables: bool,
        obs_out_filepath: str | None,
        obs_out_format: str,
    ) -> None:
        self._native.write_detectors(
            self._handle,
            shots,
            filepath,
            format,
            prepend_observables,
            append_observables,
            obs_out_filepath,
            obs_out_format,
        )


def _sample_detector_pair(
    native: ModuleType, handle: object, shots: int, widths: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    pair = native.sample_detectors(handle, shots)
    if not isinstance(pair, tuple) or len(pair) != 2:
        raise ValueError("embedded detector sampling returned an invalid pair")
    results = []
    for buffer, width in zip(pair, widths, strict=True):
        if not isinstance(buffer, bytearray) or len(buffer) != shots * width:
            raise ValueError("embedded detector sampling returned invalid bytes")
        results.append(
            torch.frombuffer(buffer, dtype=torch.bool).reshape(shots, width)
            if width
            else torch.empty((shots, 0), dtype=torch.bool)
        )
    return results[0], results[1]


def _text_experiment_engine() -> ModuleType:
    try:
        return import_module(f"{__package__}._embedded_stim")
    except (ImportError, OSError) as error:
        from .engine import StabilizerDependencyError

        raise StabilizerDependencyError(
            "the embedded stabilizer engine could not be loaded; "
            "use a compatible FlagQuantum embedded build"
        ) from error


class _EmbeddedTextExperiment:
    def __init__(
        self, native: ModuleType, handle: object, widths: tuple[int, int]
    ) -> None:
        self._native = native
        self._handle = handle
        self._widths = widths

    def sample(self, shots: int) -> torch.Tensor:
        return torch.cat(
            _sample_detector_pair(self._native, self._handle, shots, self._widths),
            dim=1,
        )


def _prepare_text_experiment(
    text: str, seed: int | None
) -> tuple[_EmbeddedTextExperiment, int, int]:
    _validate_stim_experiment_source(text)
    native = _text_experiment_engine()
    try:
        handle, detectors, observables = native.prepare_text_experiment(
            text, _validate_seed(seed)
        )
    except OverflowError as error:
        raise CapabilityError(str(error)) from error
    except RuntimeError as error:
        raise CapabilityError(
            "Stim experiment has nondeterministic declared parities"
        ) from error
    return (
        _EmbeddedTextExperiment(native, handle, (detectors, observables)),
        detectors,
        observables,
    )


def _text_experiment_dem(
    text: str, decompose_errors: bool, normalize_uniform_pauli: bool
) -> str:
    _validate_stim_experiment_source(text)
    native = _text_experiment_engine()
    try:
        result: object = native.text_experiment_dem(
            text, decompose_errors, normalize_uniform_pauli
        )
    except OverflowError as error:
        raise CapabilityError(str(error)) from error
    except RuntimeError as error:
        raise CapabilityError(
            "Stim experiment has no supported exact DEM conversion"
        ) from error
    if not isinstance(result, str):
        raise CapabilityError("embedded DEM returned an invalid text result")
    return result
