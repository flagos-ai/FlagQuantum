"""Reference QSVT with an explicit comparison permutation and projected phases."""

import cmath

import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.qsvt.main import PHASES, build_program


def reference_circuit(phases: tuple[float, ...] = PHASES) -> fq.Circuit:
    circuit = fq.Circuit(10, dtype=torch.complex128)
    comparator = torch.zeros((512, 512), dtype=torch.complex128)
    for state in range(16):
        for reference in range(16):
            for indicator in range(2):
                column = 32 * state + 2 * reference + indicator
                comparator[column ^ int(state <= reference), column] = 1

    def block(inverse: bool = False) -> None:
        if inverse:
            circuit.any(*range(9), matrix=comparator)
        for wire in range(4, 8):
            circuit.h(wire)
        if not inverse:
            circuit.any(*range(9), matrix=comparator)

    def projected(angle: float, wires: tuple[int, ...]) -> None:
        values = [
            cmath.exp(1j * angle * ((marker ^ int(word == 0)) - 0.5))
            for word in range(1 << len(wires))
            for marker in range(2)
        ]
        circuit.any(
            *wires, 9, matrix=torch.diag(torch.tensor(values, dtype=torch.complex128))
        )

    for wire in range(4):
        circuit.h(wire)
    circuit.h(9)
    projected(phases[0], (4, 5, 6, 7))
    for index in range(2):
        block()
        projected(phases[2 * index + 1], (8,))
        block(inverse=True)
        projected(phases[2 * index + 2], (4, 5, 6, 7))
    block()
    projected(phases[5], (8,))
    return circuit.h(9)


def run_example() -> None:
    compare_states(
        build_program(), reference_circuit(), ("state", "ref", "ind", "qsvt_aux")
    )


if __name__ == "__main__":
    run_example()
