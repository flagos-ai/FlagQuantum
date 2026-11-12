"""Independent operator for the commented Catalyst source, corrected to ten wires."""

import cmath

import torch

import flagquantum as fq

from ....classiq_qsvt.overview.program import PHASES

DTYPE = torch.complex128


def source_modular_adder(constant: int) -> torch.Tensor:
    matrix = torch.zeros((128, 128), dtype=DTYPE)
    for word in range(128):
        matrix[(word + constant) % 128, word] = 1
    return matrix


def reference_circuit(phases: tuple[float, ...] = PHASES) -> fq.Circuit:
    c = fq.Circuit(10, dtype=DTYPE)
    select, deletion, projection, aux = 2, 1, 0, 9
    data = tuple(range(8, 2, -1))
    controlled_add = torch.eye(256, dtype=DTYPE)
    controlled_add[128:, 128:] = source_modular_adder(2)
    subtract = source_modular_adder(127)
    reflect = torch.eye(128, dtype=DTYPE)
    reflect[64, 64] = -1

    def block(inverse: bool = False) -> None:
        c.h(select).h(projection)
        if inverse:
            c.any(projection, *data, matrix=reflect.mH.contiguous())
            c.any(*data, deletion, matrix=subtract.mH.contiguous())
            c.any(select, *data, deletion, matrix=controlled_add.mH.contiguous())
        else:
            c.any(select, *data, deletion, matrix=controlled_add)
            c.any(*data, deletion, matrix=subtract)
            c.any(projection, *data, matrix=reflect)
        c.h(select).h(projection)

    def project(angle: float) -> None:
        diagonal = torch.tensor(
            [
                cmath.exp(0.5j * angle * (1 if (word & 1) ^ (word < 2) else -1))
                for word in range(16)
            ],
            dtype=DTYPE,
        )
        c.any(0, 1, 2, 9, matrix=torch.diag(diagonal))

    c.h(aux)
    project(phases[0])
    block()
    project(phases[1])
    block(True)
    project(phases[2])
    block()
    project(phases[3])
    c.h(aux)
    return c


def main() -> None:
    """Run independent native Circuit inspection of the logical auxiliary."""
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
