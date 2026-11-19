"""Independent dense component operators for literal half-turn and CircBox semantics.

The largest matrix is 256 by 256 complex128 (one MiB). No foreign SDK runs.
"""

import cmath
import math

import torch

import flagquantum as fq

from ....classiq_qsvt.overview.program import PHASES

DTYPE = torch.complex128


def source_fourier_adder(constant: int) -> torch.Tensor:
    """Multiply literal source QFT and half-turn RZ matrices on seven MSB wires."""
    qft = torch.eye(128, dtype=DTYPE)
    for i in range(7):
        mask = 1 << (6 - i)
        hadamard = torch.zeros_like(qft)
        for word in range(128):
            hadamard[word & ~mask, word] = 2**-0.5
            hadamard[word | mask, word] = (-1 if word & mask else 1) * 2**-0.5
        qft = hadamard @ qft
        for j in range(i + 1, 7):
            other = 1 << (6 - j)
            angle = math.pi / 2 ** (j - i)
            diagonal = torch.tensor(
                [
                    cmath.exp(1j * angle) if word & mask and word & other else 1
                    for word in range(128)
                ],
                dtype=DTYPE,
            )
            qft = diagonal[:, None] * qft
    rotations = torch.ones(128, dtype=DTYPE)
    for j in range(7):
        angle = (constant % 2 ** (j + 1)) * math.pi**2 / 2**j
        rotations *= torch.tensor(
            [
                cmath.exp(0.5j * angle * (1 if word & (1 << (6 - j)) else -1))
                for word in range(128)
            ],
            dtype=DTYPE,
        )
    return qft.mH @ (rotations[:, None] * qft)


def source_reflection() -> torch.Tensor:
    """Literal X0 H0 zero-MCX(target5) H0 X0, without a diagonal assumption."""
    h = torch.tensor([[1, 1], [1, -1]], dtype=DTYPE) * 2**-0.5
    x = torch.tensor([[0, 1], [1, 0]], dtype=DTYPE)
    h0 = torch.kron(h, torch.eye(32, dtype=DTYPE))
    x0 = torch.kron(x, torch.eye(32, dtype=DTYPE))
    mcx = torch.eye(64, dtype=DTYPE)
    mcx[:, 0] = 0
    mcx[:, 1] = 0
    mcx[1, 0] = 1
    mcx[0, 1] = 1
    return x0 @ h0 @ mcx @ h0 @ x0


def zero_controlled(operator: torch.Tensor) -> torch.Tensor:
    """Place an operator in the selector-zero block of an independent matrix."""
    size = operator.shape[0]
    matrix = torch.eye(2 * size, dtype=DTYPE)
    matrix[:size, :size] = operator
    return matrix


def reference_circuit(phases: tuple[float, ...] = PHASES) -> fq.Circuit:
    """Reconstruct three literal boxes on canonical [block,data,aux] MSB wires."""
    circuit = fq.Circuit(10, dtype=DTYPE)
    # Internal [block,data] is embedded positionally on outer [data,block].
    select, deletion, projection, aux = 8, 7, 6, 9
    data = tuple(range(5, -1, -1))
    add = zero_controlled(source_fourier_adder(2))
    subtract = source_fourier_adder(127)
    reflect = zero_controlled(source_reflection())

    def block(inverse: bool = False) -> None:
        circuit.h(select).h(projection)
        if inverse:
            circuit.any(projection, *data, matrix=reflect.mH.contiguous())
            circuit.any(*data, deletion, matrix=subtract.mH.contiguous())
            circuit.any(select, *data, deletion, matrix=add.mH.contiguous())
        else:
            circuit.any(select, *data, deletion, matrix=add)
            circuit.any(*data, deletion, matrix=subtract)
            circuit.any(projection, *data, matrix=reflect)
        circuit.h(select).h(projection)

    def project(half_turn: float) -> None:
        angle = math.pi * half_turn
        diagonal = torch.tensor(
            [
                cmath.exp(0.5j * angle * (1 if (word & 1) ^ (word < 2) else -1))
                for word in range(16)
            ],
            dtype=DTYPE,
        )
        circuit.any(0, 1, 2, 9, matrix=torch.diag(diagonal))

    circuit.h(aux)
    project(phases[0])
    block()
    project(phases[1])
    block(True)
    project(phases[2])
    block()
    project(phases[3])
    circuit.h(aux)
    return circuit


def main() -> None:
    """Run the independent reference and show named auxiliary probabilities."""
    probabilities = (
        fq.run(reference_circuit())
        .statevector()
        .reshape(512, 2)
        .abs()
        .square()
        .sum(dim=0)
    )
    print({(bit,): float(probabilities[bit]) for bit in range(2)})


if __name__ == "__main__":
    main()
