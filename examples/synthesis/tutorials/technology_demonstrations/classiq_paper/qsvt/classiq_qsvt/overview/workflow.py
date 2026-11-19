"""Observe native QSVT synthesis, exact target conversion and local verification."""

import time
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler.basis_conversion import convert_basis
from flagquantum.compiler.resource_estimation import (
    ResourceEstimate,
    estimate_resources,
)
from flagquantum.core.ir import CircuitIR

from .comparisons.circuit import reference_circuit
from .exact_basis import exact_rules
from .program import PHASES, build_program, main


@dataclass(frozen=True)
class Workflow:
    """Actually compiled artifacts, target costs and measured local durations."""

    program: synth.Program
    executable: CircuitIR
    resources: ResourceEstimate
    synthesis_seconds: float
    conversion_seconds: float
    elapsed_seconds: float


def logical_state(executable: CircuitIR, program: synth.Program) -> torch.Tensor:
    """Extract named logical amplitudes only after checking hidden work is zero."""
    logical = tuple(
        w for name in ("block", "data", "qsvt_aux") for w in program.registers[name]
    )
    width = executable.n_wires
    hidden = tuple(w for w in range(width) if w not in logical)
    vector = fq.run(executable).statevector().flatten()
    ordered = (
        vector.reshape((2,) * width).permute((*logical, *hidden)).reshape(1024, -1)
    )
    if float(ordered[:, 1:].abs().square().sum()) > 1e-20:
        raise ValueError("QSVT returned nonzero compiler workspace.")
    result = ordered[:, 0]
    if not isinstance(result, torch.Tensor):
        raise TypeError("Expected a statevector tensor.")
    return result


def run_workflow(
    definition: object = main,
    *,
    phases: tuple[float, ...] = PHASES,
    max_added_operations: int = 200_000,
) -> Workflow:
    """Retain source width eleven and verify native and CX/U3 absolute states."""
    started = time.perf_counter()
    program = build_program(definition, phases=phases)
    synthesis_seconds = time.perf_counter() - started
    begin = time.perf_counter()
    executable = convert_basis(
        program.circuit,
        gates=("cx", "u3"),
        max_added_operations=max_added_operations,
        rules=exact_rules(),
    ).program
    conversion_seconds = time.perf_counter() - begin
    resources = estimate_resources(executable, gates=("cx", "u3"))
    if resources.n_qubits > 11:
        raise ValueError("The source width ceiling is eleven physical qubits.")
    expected = fq.run(reference_circuit(phases)).statevector().flatten()
    for ir in (program.circuit, executable):
        torch.testing.assert_close(
            logical_state(ir, program), expected, atol=1e-10, rtol=0
        )
    return Workflow(
        program,
        executable,
        resources,
        synthesis_seconds,
        conversion_seconds,
        time.perf_counter() - started,
    )
