"""Reference exact controlled RY and an explicitly enumerated mixer diagonal."""

import cmath

import torch

import flagquantum as fq
from examples.synthesis.algorithms.search_and_optimization.grover_mixer_qaoa.gm_qaoa.main import (
    PARAMETERS,
    build_program,
)
from examples.synthesis.compare_states import compare_states


def initial_circuit() -> fq.Circuit:
    circuit = fq.Circuit(4, dtype=torch.complex128).x(3)
    for low, high, angle in (
        (3, 2, 2.0943951023932),
        (2, 1, 1.91063323624902),
        (1, 0, 1.5707963267949),
    ):
        circuit.cx(high, low).cry(low, high, theta=angle).cx(high, low)
    return circuit


def reference_circuit(params: tuple[float, ...] = PARAMETERS) -> fq.Circuit:
    preparation = initial_circuit()
    circuit = fq.Circuit(4, dtype=torch.complex128).compose(preparation)
    for gamma, beta in zip(params[:3], params[3:], strict=True):
        for wire, coefficient in ((3, 1), (2, 2), (1, -1), (0, 12)):
            circuit.phase(wire, theta=gamma * coefficient)
        circuit.compose(preparation.adjoint()).x(0)
        values = [
            (
                cmath.exp(-0.318309886183791j * beta * ((raw >> 3) - 0.5))
                if raw & 7 == 0
                else 1.0
            )
            for raw in range(16)
        ]
        circuit.any(
            *range(4), matrix=torch.diag(torch.tensor(values, dtype=torch.complex128))
        )
        circuit.x(0).compose(preparation)
    return circuit


def run_example() -> None:
    compare_states(build_program(), reference_circuit(), ("x",))


if __name__ == "__main__":
    run_example()
