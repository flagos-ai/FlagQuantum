"""Optimize the original one-qubit RY ansatz against Z + 0.1 X."""

import math

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.algorithms import NelderMeadOptimizer, NelderMeadResult
from flagquantum.synth import Output, QBit, R, allocate, qfunc

HAMILTONIAN = torch.tensor(((1.0, 0.1), (0.1, -1.0)), dtype=torch.complex128)


@qfunc
def main(angle: float, q: Output[QBit]) -> None:
    allocate(q)
    R(angle, math.pi / 2, q)


def build_program(angle: float) -> synth.Program:
    return synth.synthesize(
        main,
        classical_inputs={"angle": angle},
        max_qubits=1,
        max_workspace=0,
        memory_limit_bytes=1024**2,
    )


def energy(parameters: torch.Tensor) -> torch.Tensor:
    program = build_program(float(parameters[0]))
    state = fq.run(program).statevector().flatten()
    return torch.vdot(state, HAMILTONIAN @ state).real


def run_workflow() -> NelderMeadResult:
    # Retain zero initial angle and maximum 1000 iterations; choose an explicit
    # existing local optimizer rather than assuming the cloud optimizer method.
    optimizer = NelderMeadOptimizer(maxiter=1000)
    result = optimizer.minimize(energy, torch.zeros(1, dtype=torch.float64))
    assert result.converged and abs(result.value + math.sqrt(1.01)) < 1e-10
    print("Minimal energy:", result.value)
    print("Optimal parameters:", result.parameters.tolist())
    print("Iterations:", result.iterations, "evaluations:", result.evaluations)
    return result


def run_example() -> None:
    run_workflow()


if __name__ == "__main__":
    run_example()
