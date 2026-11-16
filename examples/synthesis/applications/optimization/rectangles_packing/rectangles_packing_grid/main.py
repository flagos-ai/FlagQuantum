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

WIDTH = 17
PARAMETER_COUNT = 20
HAMILTONIAN = (
    2441.5 * fq.I()
    + -449.5 * (fq.Z(7))
    + -349.5 * (fq.Z(8))
    + -349.5 * (fq.Z(6))
    + -324.5 * (fq.Z(3))
    + -324.5 * (fq.Z(2))
    + -274.5 * (fq.Z(14))
    + -274.5 * (fq.Z(13))
    + -274.5 * (fq.Z(12))
    + -274.5 * (fq.Z(11))
    + -274.5 * (fq.Z(5))
    + -274.5 * (fq.Z(4))
    + -274.5 * (fq.Z(1))
    + -274.5 * (fq.Z(0))
    + -224.5 * (fq.Z(16))
    + -224.5 * (fq.Z(15))
    + -224.5 * (fq.Z(10))
    + -224.5 * (fq.Z(9))
    + 25.0 * (fq.Z(15) @ fq.Z(16))
    + 25.0 * (fq.Z(14) @ fq.Z(16))
    + 25.0 * (fq.Z(13) @ fq.Z(16))
    + 25.0 * (fq.Z(12) @ fq.Z(16))
    + 25.0 * (fq.Z(11) @ fq.Z(16))
    + 25.0 * (fq.Z(10) @ fq.Z(16))
    + 25.0 * (fq.Z(9) @ fq.Z(16))
    + 25.0 * (fq.Z(8) @ fq.Z(16))
    + 25.0 * (fq.Z(5) @ fq.Z(16))
    + 25.0 * (fq.Z(14) @ fq.Z(15))
    + 25.0 * (fq.Z(13) @ fq.Z(15))
    + 25.0 * (fq.Z(12) @ fq.Z(15))
    + 25.0 * (fq.Z(11) @ fq.Z(15))
    + 25.0 * (fq.Z(10) @ fq.Z(15))
    + 25.0 * (fq.Z(9) @ fq.Z(15))
    + 25.0 * (fq.Z(8) @ fq.Z(15))
    + 25.0 * (fq.Z(4) @ fq.Z(15))
    + 25.0 * (fq.Z(13) @ fq.Z(14))
    + 25.0 * (fq.Z(12) @ fq.Z(14))
    + 25.0 * (fq.Z(11) @ fq.Z(14))
    + 25.0 * (fq.Z(10) @ fq.Z(14))
    + 25.0 * (fq.Z(9) @ fq.Z(14))
    + 25.0 * (fq.Z(8) @ fq.Z(14))
    + 25.0 * (fq.Z(7) @ fq.Z(14))
    + 25.0 * (fq.Z(5) @ fq.Z(14))
    + 25.0 * (fq.Z(3) @ fq.Z(14))
    + 25.0 * (fq.Z(12) @ fq.Z(13))
    + 25.0 * (fq.Z(11) @ fq.Z(13))
    + 25.0 * (fq.Z(10) @ fq.Z(13))
    + 25.0 * (fq.Z(9) @ fq.Z(13))
    + 25.0 * (fq.Z(8) @ fq.Z(13))
    + 25.0 * (fq.Z(7) @ fq.Z(13))
    + 25.0 * (fq.Z(4) @ fq.Z(13))
    + 25.0 * (fq.Z(2) @ fq.Z(13))
    + 25.0 * (fq.Z(11) @ fq.Z(12))
    + 25.0 * (fq.Z(10) @ fq.Z(12))
    + 25.0 * (fq.Z(9) @ fq.Z(12))
    + 25.0 * (fq.Z(7) @ fq.Z(12))
    + 25.0 * (fq.Z(6) @ fq.Z(12))
    + 25.0 * (fq.Z(3) @ fq.Z(12))
    + 25.0 * (fq.Z(1) @ fq.Z(12))
    + 25.0 * (fq.Z(10) @ fq.Z(11))
    + 25.0 * (fq.Z(9) @ fq.Z(11))
    + 25.0 * (fq.Z(7) @ fq.Z(11))
    + 25.0 * (fq.Z(6) @ fq.Z(11))
    + 25.0 * (fq.Z(2) @ fq.Z(11))
    + 25.0 * (fq.Z(0) @ fq.Z(11))
    + 25.0 * (fq.Z(9) @ fq.Z(10))
    + 25.0 * (fq.Z(6) @ fq.Z(10))
    + 25.0 * (fq.Z(1) @ fq.Z(10))
    + 25.0 * (fq.Z(6) @ fq.Z(9))
    + 25.0 * (fq.Z(0) @ fq.Z(9))
    + 25.0 * (fq.Z(6) @ fq.Z(8))
    + 25.0 * (fq.Z(3) @ fq.Z(8))
    + 25.0 * (fq.Z(2) @ fq.Z(8))
    + 25.0 * (fq.Z(5) @ fq.Z(7))
    + 25.0 * (fq.Z(4) @ fq.Z(7))
    + 25.0 * (fq.Z(1) @ fq.Z(7))
    + 25.0 * (fq.Z(0) @ fq.Z(7))
    + 25.0 * (fq.Z(3) @ fq.Z(6))
    + 25.0 * (fq.Z(2) @ fq.Z(6))
    + 25.0 * (fq.Z(4) @ fq.Z(5))
    + 25.0 * (fq.Z(2) @ fq.Z(5))
    + 25.0 * (fq.Z(1) @ fq.Z(5))
    + 25.0 * (fq.Z(0) @ fq.Z(5))
    + 25.0 * (fq.Z(3) @ fq.Z(4))
    + 25.0 * (fq.Z(1) @ fq.Z(4))
    + 25.0 * (fq.Z(0) @ fq.Z(4))
    + 25.0 * (fq.Z(2) @ fq.Z(3))
    + 25.0 * (fq.Z(0) @ fq.Z(3))
    + 25.0 * (fq.Z(1) @ fq.Z(2))
    + 25.0 * (fq.Z(0) @ fq.Z(1))
    + 50.0 * (fq.Z(5) @ fq.Z(8))
    + 50.0 * (fq.Z(4) @ fq.Z(8))
    + 50.0 * (fq.Z(3) @ fq.Z(7))
    + 50.0 * (fq.Z(2) @ fq.Z(7))
    + 50.0 * (fq.Z(1) @ fq.Z(6))
    + 50.0 * (fq.Z(0) @ fq.Z(6))
    + 50.0 * (fq.Z(3) @ fq.Z(5))
    + 50.0 * (fq.Z(2) @ fq.Z(4))
    + 50.0 * (fq.Z(1) @ fq.Z(3))
    + 50.0 * (fq.Z(0) @ fq.Z(2))
    + 75.0 * (fq.Z(7) @ fq.Z(8))
    + 75.0 * (fq.Z(6) @ fq.Z(7))
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
    allocate(17, target)
    hadamard_transform(target)
    for layer in range(10):
        cost_layer(params_list[2 * layer], target)
        for bit in range(17):
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
