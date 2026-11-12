"""Retain every diagonal cost coefficient and QAOA layer in native Python."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    RX,
    Output,
    QArray,
    QBit,
    allocate,
    hadamard_transform,
    qfunc,
    suzuki_trotter,
)

WIDTH = 7
PARAMETER_COUNT = 20
HAMILTONIAN = (
    30.5 * fq.I()
    + -6.5 * (fq.Z(6))
    + -6.5 * (fq.Z(1))
    + -8.5 * (fq.Z(5))
    + -8.5 * (fq.Z(3))
    + -17.5 * (fq.Z(4))
    + -23.5 * (fq.Z(2))
    + -23.5 * (fq.Z(0))
    + 1.0 * (fq.Z(5) @ fq.Z(6))
    + 1.0 * (fq.Z(3) @ fq.Z(6))
    + 1.0 * (fq.Z(3) @ fq.Z(5))
    + 1.0 * (fq.Z(1) @ fq.Z(5))
    + 1.0 * (fq.Z(1) @ fq.Z(3))
    + 2.0 * (fq.Z(4) @ fq.Z(6))
    + 2.0 * (fq.Z(1) @ fq.Z(4))
    + 4.0 * (fq.Z(0) @ fq.Z(5))
    + 4.0 * (fq.Z(2) @ fq.Z(3))
    + 16.0 * (fq.Z(0) @ fq.Z(2))
    + 3.0 * (fq.Z(2) @ fq.Z(6))
    + 3.0 * (fq.Z(0) @ fq.Z(6))
    + 3.0 * (fq.Z(4) @ fq.Z(5))
    + 3.0 * (fq.Z(3) @ fq.Z(4))
    + 3.0 * (fq.Z(1) @ fq.Z(2))
    + 3.0 * (fq.Z(0) @ fq.Z(1))
    + 7.0 * (fq.Z(1) @ fq.Z(6))
    + 9.0 * (fq.Z(2) @ fq.Z(5))
    + 9.0 * (fq.Z(0) @ fq.Z(3))
    + 13.0 * (fq.Z(2) @ fq.Z(4))
    + 13.0 * (fq.Z(0) @ fq.Z(4))
    + -1.0 * (fq.Z(2) @ fq.Z(5) @ fq.Z(6))
    + -1.0 * (fq.Z(1) @ fq.Z(5) @ fq.Z(6))
    + -1.0 * (fq.Z(2) @ fq.Z(4) @ fq.Z(6))
    + -1.0 * (fq.Z(0) @ fq.Z(4) @ fq.Z(6))
    + -1.0 * (fq.Z(1) @ fq.Z(3) @ fq.Z(6))
    + -1.0 * (fq.Z(0) @ fq.Z(3) @ fq.Z(6))
    + -1.0 * (fq.Z(0) @ fq.Z(2) @ fq.Z(6))
    + -1.0 * (fq.Z(0) @ fq.Z(4) @ fq.Z(5))
    + -1.0 * (fq.Z(2) @ fq.Z(3) @ fq.Z(5))
    + -1.0 * (fq.Z(0) @ fq.Z(3) @ fq.Z(5))
    + -1.0 * (fq.Z(1) @ fq.Z(2) @ fq.Z(5))
    + -1.0 * (fq.Z(2) @ fq.Z(3) @ fq.Z(4))
    + -1.0 * (fq.Z(1) @ fq.Z(2) @ fq.Z(4))
    + -1.0 * (fq.Z(0) @ fq.Z(1) @ fq.Z(4))
    + -1.0 * (fq.Z(0) @ fq.Z(1) @ fq.Z(3))
    + -1.0 * (fq.Z(0) @ fq.Z(1) @ fq.Z(2))
    + -2.0 * (fq.Z(1) @ fq.Z(4) @ fq.Z(6))
    + -4.0 * (fq.Z(0) @ fq.Z(2) @ fq.Z(5))
    + -4.0 * (fq.Z(0) @ fq.Z(2) @ fq.Z(3))
    + -8.0 * (fq.Z(0) @ fq.Z(2) @ fq.Z(4))
    + -3.0 * (fq.Z(1) @ fq.Z(2) @ fq.Z(6))
    + -3.0 * (fq.Z(0) @ fq.Z(1) @ fq.Z(6))
    + -3.0 * (fq.Z(2) @ fq.Z(4) @ fq.Z(5))
    + -3.0 * (fq.Z(0) @ fq.Z(3) @ fq.Z(4))
    + 1.0 * (fq.Z(1) @ fq.Z(2) @ fq.Z(5) @ fq.Z(6))
    + 1.0 * (fq.Z(1) @ fq.Z(2) @ fq.Z(4) @ fq.Z(6))
    + 1.0 * (fq.Z(0) @ fq.Z(1) @ fq.Z(4) @ fq.Z(6))
    + 1.0 * (fq.Z(0) @ fq.Z(1) @ fq.Z(3) @ fq.Z(6))
    + 1.0 * (fq.Z(0) @ fq.Z(1) @ fq.Z(2) @ fq.Z(6))
    + 1.0 * (fq.Z(0) @ fq.Z(2) @ fq.Z(4) @ fq.Z(5))
    + 1.0 * (fq.Z(0) @ fq.Z(2) @ fq.Z(3) @ fq.Z(5))
    + 1.0 * (fq.Z(0) @ fq.Z(2) @ fq.Z(3) @ fq.Z(4))
)
DEFAULT_PARAMS = (
    0.013,
    0.026,
    0.039,
    0.052,
    0.065,
    0.078,
    0.091,
    0.104,
    0.117,
    0.13,
    0.143,
    0.156,
    0.169,
    0.182,
    0.195,
    0.208,
    0.221,
    0.234,
    0.247,
    0.26,
)


@qfunc
def cost_layer(gamma: float, target: QArray[QBit]) -> None:
    suzuki_trotter(HAMILTONIAN, gamma, 1, 1, target)


@qfunc
def main(params_list: list[float], target: Output[QArray[QBit]]) -> None:
    allocate(7, target)
    hadamard_transform(target)
    for layer in range(10):
        cost_layer(params_list[2 * layer], target)
        for bit in range(7):
            RX(params_list[2 * layer + 1], target[bit])


def build_program(params: tuple[float, ...] = DEFAULT_PARAMS) -> synth.Program:
    if len(params) != PARAMETER_COUNT:
        raise ValueError("This retained QAOA program requires 20 parameters")
    return synth.synthesize(
        main, classical_inputs={"params_list": params}, max_workspace=0
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(program.report["resources"])
    print("State norm:", result.statevector().norm().item())
