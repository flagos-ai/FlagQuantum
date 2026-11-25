"""Serialize positioned memory circuits at the stabilizer engine boundary."""

from __future__ import annotations

from pathlib import Path

from ...core.ir import CircuitIR


def _annotated_stim_text(
    program: CircuitIR,
    *,
    terminal_qubits: tuple[int, ...],
    detector_columns: tuple[tuple[int, ...], ...],
    observable_columns: tuple[tuple[int, ...], ...],
) -> str:
    """Serialize and validate a positioned program with QEC-owned parity columns."""
    from . import engine

    text = _annotated_stim_source(
        program,
        terminal_qubits=terminal_qubits,
        detector_columns=detector_columns,
        observable_columns=observable_columns,
    )
    if Path(engine.__file__).with_name("_embedded_stim_profile.txt").is_file():
        from ._embedded_sampler import _validate_text_experiment

        _validate_text_experiment(text)
    else:
        engine._parse_stim_experiment(text)
    return text


def _annotated_stim_source(
    program: CircuitIR,
    *,
    terminal_qubits: tuple[int, ...],
    detector_columns: tuple[tuple[int, ...], ...],
    observable_columns: tuple[tuple[int, ...], ...],
) -> str:
    from . import engine

    ir = engine._measurement_record_ir(program)
    if Path(engine.__file__).with_name("_embedded_stim_profile.txt").is_file():
        from ._embedded_sampler import _annotated_operations_text

        return _annotated_operations_text(
            engine._noisy_operations(ir),
            terminal_qubits,
            detector_columns,
            observable_columns,
        )
    circuit = engine._noisy_engine_circuit(
        engine._noisy_operations(ir), terminal_qubits
    )
    recorded = circuit.num_measurements
    stim = engine._stim()
    for kind, columns in (
        ("DETECTOR", detector_columns),
        ("OBSERVABLE_INCLUDE", observable_columns),
    ):
        for index, parity in enumerate(columns):
            if any(column < 0 or column >= recorded for column in parity):
                raise ValueError("parity column outside recorded measurements")
            targets = [stim.target_rec(column - recorded) for column in parity]
            if kind == "DETECTOR":
                circuit.append(kind, targets)
            else:
                circuit.append(kind, targets, index)
    lines = []
    for instruction in circuit:
        line = str(instruction)
        if arguments := instruction.gate_args_copy():
            # The SDK circuit printer rounds arguments; preserve their binary64 values.
            prefix, suffix = line.split("(", 1)
            values = ", ".join(map(repr, arguments))
            line = f"{prefix}({values}){suffix.split(')', 1)[1]}"
        lines.append(line)
    return "\n".join(lines)
