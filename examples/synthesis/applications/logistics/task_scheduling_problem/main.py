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

WIDTH = 24
PARAMETER_COUNT = 16
HAMILTONIAN = (
    851.0 * fq.I()
    + -10.0 * (fq.Z(23))
    + -20.0 * (fq.Z(2))
    + -40.0 * (fq.Z(1))
    + -80.0 * (fq.Z(12))
    + -30.0 * (fq.Z(11))
    + -60.0 * (fq.Z(5))
    + -60.0 * (fq.Z(3))
    + -120.0 * (fq.Z(4))
    + -42.0 * (fq.Z(10))
    + -12.5 * (fq.Z(9))
    + -50.0 * (fq.Z(7))
    + -100.0 * (fq.Z(6))
    + -100.0 * (fq.Z(18))
    + -200.0 * (fq.Z(17))
    + -110.0 * (fq.Z(20))
    + -110.0 * (fq.Z(19))
    + -130.0 * (fq.Z(16))
    + -130.0 * (fq.Z(15))
    + -75.0 * (fq.Z(22))
    + -75.0 * (fq.Z(14))
    + -170.0 * (fq.Z(21))
    + -91.5 * (fq.Z(13))
    + 5.0 * (fq.Z(18) @ fq.Z(21))
    + 5.0 * (fq.Z(18) @ fq.Z(20))
    + 5.0 * (fq.Z(18) @ fq.Z(19))
    + 5.0 * (fq.Z(12) @ fq.Z(13))
    + 5.0 * (fq.Z(11) @ fq.Z(13))
    + 10.0 * (fq.Z(0) @ fq.Z(9))
    + 10.0 * (fq.Z(8) @ fq.Z(23))
    + 10.0 * (fq.Z(7) @ fq.Z(22))
    + 10.0 * (fq.Z(3) @ fq.Z(5))
    + 10.0 * (fq.Z(5) @ fq.Z(18))
    + 10.0 * (fq.Z(5) @ fq.Z(14))
    + 10.0 * (fq.Z(5) @ fq.Z(13))
    + 10.0 * (fq.Z(3) @ fq.Z(18))
    + 10.0 * (fq.Z(3) @ fq.Z(14))
    + 10.0 * (fq.Z(3) @ fq.Z(13))
    + 10.0 * (fq.Z(2) @ fq.Z(11))
    + 10.0 * (fq.Z(2) @ fq.Z(10))
    + 10.0 * (fq.Z(22) @ fq.Z(23))
    + 10.0 * (fq.Z(18) @ fq.Z(23))
    + 10.0 * (fq.Z(18) @ fq.Z(22))
    + 10.0 * (fq.Z(17) @ fq.Z(21))
    + 10.0 * (fq.Z(12) @ fq.Z(21))
    + 10.0 * (fq.Z(16) @ fq.Z(20))
    + 10.0 * (fq.Z(15) @ fq.Z(19))
    + 10.0 * (fq.Z(14) @ fq.Z(18))
    + 10.0 * (fq.Z(13) @ fq.Z(18))
    + 10.0 * (fq.Z(12) @ fq.Z(17))
    + 10.0 * (fq.Z(11) @ fq.Z(14))
    + 10.0 * (fq.Z(10) @ fq.Z(13))
    + 10.0 * (fq.Z(9) @ fq.Z(13))
    + 10.0 * (fq.Z(9) @ fq.Z(10))
    + 20.0 * (fq.Z(6) @ fq.Z(7))
    + 20.0 * (fq.Z(7) @ fq.Z(20))
    + 20.0 * (fq.Z(7) @ fq.Z(19))
    + 20.0 * (fq.Z(6) @ fq.Z(22))
    + 20.0 * (fq.Z(4) @ fq.Z(5))
    + 20.0 * (fq.Z(5) @ fq.Z(16))
    + 20.0 * (fq.Z(5) @ fq.Z(15))
    + 20.0 * (fq.Z(3) @ fq.Z(4))
    + 20.0 * (fq.Z(4) @ fq.Z(18))
    + 20.0 * (fq.Z(4) @ fq.Z(14))
    + 20.0 * (fq.Z(4) @ fq.Z(13))
    + 20.0 * (fq.Z(3) @ fq.Z(16))
    + 20.0 * (fq.Z(3) @ fq.Z(15))
    + 20.0 * (fq.Z(1) @ fq.Z(2))
    + 20.0 * (fq.Z(1) @ fq.Z(11))
    + 20.0 * (fq.Z(1) @ fq.Z(10))
    + 20.0 * (fq.Z(13) @ fq.Z(16))
    + 20.0 * (fq.Z(13) @ fq.Z(15))
    + 40.0 * (fq.Z(6) @ fq.Z(20))
    + 40.0 * (fq.Z(6) @ fq.Z(19))
    + 40.0 * (fq.Z(4) @ fq.Z(16))
    + 40.0 * (fq.Z(4) @ fq.Z(15))
    + 40.0 * (fq.Z(19) @ fq.Z(20))
    + 40.0 * (fq.Z(15) @ fq.Z(16))
    + 15.0 * (fq.Z(13) @ fq.Z(14))
    + 15.0 * (fq.Z(10) @ fq.Z(11))
    + 30.0 * (fq.Z(7) @ fq.Z(21))
    + 30.0 * (fq.Z(5) @ fq.Z(17))
    + 30.0 * (fq.Z(3) @ fq.Z(17))
    + 30.0 * (fq.Z(2) @ fq.Z(12))
    + 30.0 * (fq.Z(14) @ fq.Z(17))
    + 30.0 * (fq.Z(11) @ fq.Z(12))
    + 60.0 * (fq.Z(6) @ fq.Z(21))
    + 60.0 * (fq.Z(4) @ fq.Z(17))
    + 60.0 * (fq.Z(1) @ fq.Z(12))
    + 60.0 * (fq.Z(20) @ fq.Z(21))
    + 60.0 * (fq.Z(19) @ fq.Z(21))
    + 60.0 * (fq.Z(16) @ fq.Z(17))
    + 60.0 * (fq.Z(15) @ fq.Z(17))
    + 25.0 * (fq.Z(20) @ fq.Z(22))
    + 25.0 * (fq.Z(19) @ fq.Z(22))
    + 25.0 * (fq.Z(16) @ fq.Z(18))
    + 25.0 * (fq.Z(15) @ fq.Z(18))
    + 25.0 * (fq.Z(14) @ fq.Z(16))
    + 25.0 * (fq.Z(14) @ fq.Z(15))
    + 35.0 * (fq.Z(21) @ fq.Z(22))
    + 35.0 * (fq.Z(17) @ fq.Z(18))
    + 35.0 * (fq.Z(13) @ fq.Z(17))
    + 35.0 * (fq.Z(10) @ fq.Z(12))
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
)


@qfunc
def cost_layer(gamma: float, target: QArray[QBit]) -> None:
    suzuki_trotter(HAMILTONIAN, gamma, 1, 1, target)


@qfunc
def main(params_list: list[float], target: Output[QArray[QBit]]) -> None:
    allocate(24, target)
    hadamard_transform(target)
    for layer in range(8):
        cost_layer(params_list[2 * layer], target)
        for bit in range(24):
            RX(params_list[2 * layer + 1], target[bit])


def build_program(params: tuple[float, ...] = DEFAULT_PARAMS) -> synth.Program:
    if len(params) != PARAMETER_COUNT:
        raise ValueError("This retained QAOA program requires 16 parameters")
    return synth.synthesize(
        main, classical_inputs={"params_list": params}, max_workspace=0
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(program.report["resources"])
    print("State norm:", result.statevector().norm().item())
