"""Synthesize and verify one 551-term first-order water evolution."""

import math
from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.pauli_evolution import reference_circuit, reference_sequence
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate, estimate_resources
from flagquantum.compiler.basis_conversion import convert_basis
from flagquantum.synth import Output, QArray, QBit, allocate, qfunc, suzuki_trotter

from .model import WIDTH, Model, build_model


@dataclass(frozen=True)
class Result:
    model: Model
    program: synth.Program
    target_resources: ResourceEstimate
    absolute_error: float
    target_error: float
    distribution: tuple[tuple[int, float], ...]


def build_program(model: Model, time: float = 1.0) -> synth.Program:
    if not math.isfinite(time):
        raise ValueError("Evolution time must be finite")
    hamiltonian = model.hamiltonian

    @qfunc
    def main(time: float, state: Output[QArray[QBit]]) -> None:
        allocate(12, state)
        suzuki_trotter(hamiltonian, time, 1, 1, state)

    return synth.synthesize(
        main,
        classical_inputs={"time": time},
        max_qubits=WIDTH,
        max_workspace=0,
        max_operations=50_000,
        memory_limit_bytes=64 * 1024**2,
    )


def run_workflow() -> Result:
    model = build_model()
    program = build_program(model)
    execution = fq.run(program)
    actual = execution.statevector().flatten()
    distribution = tuple(
        (word, probability)
        for (word,), probability in program.probabilities(
            execution, registers=("state",), threshold=0
        ).items()
    )
    reference = reference_circuit(
        WIDTH, model.terms, reference_sequence(model.terms, 1.0, None)
    )
    expected = fq.run(reference).statevector().flatten()
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)

    target = convert_basis(
        program.circuit, gates=("cx", "u"), max_added_operations=50_000
    )
    converted = fq.run(target.program).statevector().flatten()
    aligned = converted * torch.vdot(converted, actual)
    torch.testing.assert_close(aligned, actual, atol=1e-11, rtol=0)
    return Result(
        model,
        program,
        estimate_resources(target.program),
        float((actual - expected).abs().max()),
        float((aligned - actual).abs().max()),
        distribution,
    )
