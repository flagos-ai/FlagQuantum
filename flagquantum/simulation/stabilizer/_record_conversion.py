"""Private Stim preparation and conversion of externally supplied records."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
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
