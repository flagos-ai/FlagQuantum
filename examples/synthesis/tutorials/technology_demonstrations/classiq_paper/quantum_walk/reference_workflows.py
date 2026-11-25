"""Run native counterparts of three originally commented walk references."""

import time
from dataclasses import dataclass
from typing import Literal

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler.basis_conversion import convert_basis
from flagquantum.compiler.resource_estimation import (
    ResourceEstimate,
    estimate_resources,
)
from flagquantum.core.ir import CircuitIR

from .classiq_discrete_quantum_walk.overview.exact_basis import exact_rules
from .reference_circuit import reference_circuit
from .reference_programs import build_program
from .reference_rotation_basis import rotation_basis

Source = Literal["qiskit", "tket", "catalyst"]


@dataclass(frozen=True)
class Workflow:
    """Actual native artifact, executable named-basis IR and observed durations."""

    program: synth.Program
    executable: CircuitIR
    resources: ResourceEstimate
    synthesis_seconds: float
    lowering_seconds: float
    elapsed_seconds: float


def logical_state(executable: CircuitIR, program: synth.Program) -> torch.Tensor:
    """Verify clean work before extracting the seven logical bits in named order."""
    logical = tuple(wire for name in ("x", "coin") for wire in program.registers[name])
    width = estimate_resources(executable).n_qubits
    hidden = tuple(wire for wire in range(width) if wire not in logical)
    vector = fq.run(executable).statevector().flatten()
    ordered = vector.reshape((2,) * width).permute((*logical, *hidden)).reshape(128, -1)
    if float(ordered[:, 1:].abs().square().sum()) > 1e-24:
        raise ValueError("Reference walk returned nonzero workspace.")
    logical_vector = ordered[:, 0]
    if not isinstance(logical_vector, torch.Tensor):
        raise TypeError("Expected a statevector tensor.")
    return logical_vector


def run_workflow(source: Source, *, max_added_operations: int = 20_000) -> Workflow:
    """Compile the fixed source and verify both IRs against an independent Circuit."""
    if source not in ("qiskit", "tket", "catalyst"):
        raise ValueError("Unknown reference walk.")
    started = time.perf_counter()
    program = build_program(catalyst=source == "catalyst")
    synthesis_seconds = time.perf_counter() - started
    lowering_started = time.perf_counter()
    gates: tuple[str, ...]
    if source == "catalyst":
        executable = rotation_basis(program, max_added_operations=max_added_operations)
        gates = ("cx", "rz", "ry", "rx")
    else:
        executable = convert_basis(
            program.circuit,
            gates=("cx", "u3"),
            max_added_operations=max_added_operations,
            rules=exact_rules(),
        ).program
        gates = ("cx", "u3")
    lowering_seconds = time.perf_counter() - lowering_started
    resources = estimate_resources(executable, gates=gates)
    expected = (
        fq.run(reference_circuit(catalyst=source == "catalyst")).statevector().flatten()
    )
    for ir in (program.circuit, executable):
        torch.testing.assert_close(
            logical_state(ir, program), expected, atol=1e-11, rtol=0
        )
    return Workflow(
        program,
        executable,
        resources,
        synthesis_seconds,
        lowering_seconds,
        time.perf_counter() - started,
    )


def run_example(source: Source) -> None:
    """Print actual resources and named results; never decode physical bitstrings."""
    workflow = run_workflow(source)
    result = fq.run(workflow.program)
    print(dict(workflow.program.probabilities(result, registers=("x", "coin"))))
    print(workflow.resources)
    print("synthesis_seconds", workflow.synthesis_seconds)
    print("lowering_seconds", workflow.lowering_seconds)
    print("local_workflow_seconds", workflow.elapsed_seconds)
