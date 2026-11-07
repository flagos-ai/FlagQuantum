"""Private Stim preparation and conversion of externally supplied records."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from ...core.classical_control import is_classical_control_instruction
from ...core.ir import CircuitIR, Instruction
from ...errors import CapabilityError
from .engine import (
    CLIFFORD_GATE_NAMES,
    _annotated_stim_source,
    _parse_stim_experiment,
    _validate_stim_reference,
)


def _terminal_clifford_view(program: CircuitIR) -> CircuitIR:
    instructions: list[Instruction] = []
    for index, gate in enumerate(program.instructions):
        if gate.name in {"measure", "reset"} or is_classical_control_instruction(gate):
            raise CapabilityError("terminal conversion requires a static circuit")
        if gate.matrix is not None or gate.metadata.get("is_channel"):
            raise CapabilityError(
                f"instruction {index} is outside the exact Clifford subset"
            )
        if gate.name in CLIFFORD_GATE_NAMES and not gate.params:
            instructions.append(gate)
            continue
        theta = gate.params.get("theta")
        if (
            gate.name == "rx"
            and gate.params.keys() == {"theta"}
            and type(theta) is float
            and theta == math.pi
        ):
            # RX(pi) differs from X by a global phase, which cannot affect Z records.
            instructions.append(Instruction("x", gate.wires))
            continue
        raise CapabilityError(
            f"instruction {index} {gate.name!r} is outside the bound exact Clifford subset"
        )
    return replace(program, instructions=tuple(instructions))


def _stim_record_array(
    record: torch.Tensor, width: int, packed: bool, name: str
) -> Any:
    expected = (width + 7) // 8 if packed else width
    if record.shape[1] != expected:
        raise ValueError(f"{name} has the wrong declared width")
    if packed and width % 8 and bool((record[:, -1] >> (width % 8)).any()):
        raise ValueError(f"{name} has nonzero packed padding")
    return record.to(dtype=torch.uint8 if packed else torch.bool).contiguous().numpy()


def _owned_stim_conversion_bits(buffer: Any, shape: tuple[int, int]) -> torch.Tensor:
    try:
        bits = torch.from_numpy(buffer)
    except (TypeError, ValueError, RuntimeError) as error:
        raise CapabilityError("Stim returned an invalid conversion buffer") from error
    if bits.dtype != torch.bool or bits.shape != shape:
        raise CapabilityError("Stim returned an invalid conversion dtype or shape")
    return bits.clone()


def _compile_stim_converter(circuit: Any) -> Any:
    try:
        return circuit.compile_m2d_converter(skip_reference_sample=False)
    except (ValueError, RuntimeError) as error:
        raise CapabilityError("Stim measurement conversion failed") from error


def _convert_stim_arrays(
    converter: Any,
    raw: Any,
    sweeps: Any,
    shapes: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    if shapes[0][0] == 0:
        return torch.empty(shapes[0], dtype=torch.bool), torch.empty(
            shapes[1], dtype=torch.bool
        )
    try:
        result = converter.convert(
            measurements=raw,
            sweep_bits=sweeps,
            separate_observables=True,
            bit_packed=False,
        )
    except (ValueError, RuntimeError) as error:
        raise CapabilityError("Stim measurement conversion failed") from error
    if not isinstance(result, tuple) or len(result) != 2:
        raise CapabilityError("Stim returned an invalid conversion pair")
    return (
        _owned_stim_conversion_bits(result[0], shapes[0]),
        _owned_stim_conversion_bits(result[1], shapes[1]),
    )


def _convert_stim_measurements(
    circuit_text: str,
    measurements: torch.Tensor,
    *,
    sweep_bits: torch.Tensor | None,
    bit_packed_measurements: bool,
    bit_packed_sweep_bits: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve one-shot validation order and zero-shot behavior."""
    circuit = _parse_stim_experiment(circuit_text)
    raw = _stim_record_array(
        measurements, circuit.num_measurements, bit_packed_measurements, "measurements"
    )
    sweeps = None
    if sweep_bits is not None:
        sweeps = _stim_record_array(
            sweep_bits, circuit.num_sweep_bits, bit_packed_sweep_bits, "sweep_bits"
        )
    _validate_stim_reference(circuit)
    shots = measurements.shape[0]
    converter = _compile_stim_converter(circuit) if shots else None
    return _convert_stim_arrays(
        converter,
        raw,
        sweeps,
        ((shots, circuit.num_detectors), (shots, circuit.num_observables)),
    )


def _prepare_terminal_record_conversion(
    program: CircuitIR,
    *,
    terminal_qubits: tuple[int, ...],
    detector_columns: tuple[tuple[int, ...], ...],
    observable_columns: tuple[tuple[int, ...], ...],
) -> Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    text = _annotated_stim_source(
        _terminal_clifford_view(program),
        terminal_qubits=terminal_qubits,
        detector_columns=detector_columns,
        observable_columns=observable_columns,
    )
    circuit = _parse_stim_experiment(text)
    _validate_stim_reference(circuit)
    converter = _compile_stim_converter(circuit)
    width, detectors, observables = (
        circuit.num_measurements,
        circuit.num_detectors,
        circuit.num_observables,
    )

    def convert(measurements: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = _stim_record_array(measurements, width, False, "measurements")
        shots = measurements.shape[0]
        return _convert_stim_arrays(
            converter, raw, None, ((shots, detectors), (shots, observables))
        )

    return convert


_MAX_EXPLICIT_CONVERSION_BYTES = 64 * 1024**2


def _prepare_explicit_backend(
    text: str, detectors: int, observables: int
) -> Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    if Path(__file__).with_name("_embedded_stim_profile.txt").is_file():
        from ._embedded_sampler import _EmbeddedRecordConverter

        try:
            native = _EmbeddedRecordConverter(text, detectors, observables)
        except (ValueError, RuntimeError) as error:
            raise CapabilityError(
                "explicit conversion has nondeterministic declared parities"
            ) from error

        def embedded_convert(
            records: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            try:
                return native.convert(records)
            except (TypeError, ValueError, RuntimeError) as error:
                raise CapabilityError(
                    "embedded measurement conversion failed"
                ) from error

        return embedded_convert

    from .engine import _stim

    if _stim().__version__ != "1.16.0":
        raise CapabilityError("explicit measurement conversion requires Stim 1.16.0")
    circuit = _parse_stim_experiment(text)
    _validate_stim_reference(circuit)
    converter = _compile_stim_converter(circuit)

    def sdk_convert(records: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _convert_stim_arrays(
            converter,
            records.contiguous().numpy(),
            None,
            ((len(records), detectors), (len(records), observables)),
        )

    return sdk_convert


def _prepare_explicit_record_conversion(
    program: CircuitIR,
    *,
    detector_columns: tuple[tuple[int, ...], ...],
    observable_columns: tuple[tuple[int, ...], ...],
) -> tuple[
    tuple[int, ...], Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
]:
    from ._embedded_sampler import _operations_text
    from .measurement_sampler import _lower_program

    operations, width = _lower_program(program)
    qubits = tuple(
        gate.wires[0] for gate in program.instructions if gate.name == "measure"
    )
    lines = [_operations_text(operations)]
    for row in detector_columns:
        targets = " ".join(f"rec[{column - width}]" for column in row)
        lines.append(f"DETECTOR {targets}")
    for index, row in enumerate(observable_columns):
        targets = " ".join(f"rec[{column - width}]" for column in row)
        lines.append(f"OBSERVABLE_INCLUDE({index}) {targets}")
    backend = _prepare_explicit_backend(
        "\n".join(lines), len(detector_columns), len(observable_columns)
    )
    outputs = len(detector_columns) + len(observable_columns)

    def convert(records: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shots = len(records)
        copied_input = shots * width * (1 + int(not records.is_contiguous()))
        output_bytes = shots * outputs
        aligned_shots = ((shots + 1023) // 1024) * 1024

        # Round both table dimensions conservatively above the pinned SIMD width.
        # Include frame tables, growing measurement storage (with the old
        # allocation live), observable storage, zero-width tables and scratch.
        def padded(rows: int) -> int:
            return max(1024, ((rows + 1023) // 1024) * 1024)

        table_rows = (
            padded(width)
            + padded(outputs)
            + 2 * padded(program.n_wires)
            + 2 * padded(2 * width)
            + padded(len(observable_columns))
            + 4 * padded(0)
            + 8
        )
        principal_bytes = table_rows * aligned_shots // 8
        for label, needed in (
            ("input copies", copied_input),
            ("output payload", output_bytes),
            ("principal tables", principal_bytes),
        ):
            if needed > _MAX_EXPLICIT_CONVERSION_BYTES:
                raise CapabilityError(
                    f"explicit conversion {label} exceeds the byte limit"
                )
        if not shots:
            return (
                torch.empty((0, len(detector_columns)), dtype=torch.bool),
                torch.empty((0, len(observable_columns)), dtype=torch.bool),
            )
        return backend(records)

    return qubits, convert
