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

WIDTH = 6
PARAMETER_COUNT = 4
HAMILTONIAN = (
    -5.5 * fq.I()
    + 0.5 * (fq.Z(2) @ fq.Z(3))
    + 0.5 * (fq.Z(0) @ fq.Z(3))
    + 0.5 * (fq.Z(3) @ fq.Z(5))
    + 0.5 * (fq.Z(3) @ fq.Z(4))
    + 0.5 * (fq.Z(1) @ fq.Z(2))
    + 0.5 * (fq.Z(2) @ fq.Z(4))
    + 0.5 * (fq.Z(0) @ fq.Z(1))
    + 0.5 * (fq.Z(1) @ fq.Z(5))
    + 0.5 * (fq.Z(1) @ fq.Z(4))
    + 0.5 * (fq.Z(0) @ fq.Z(5))
    + 0.5 * (fq.Z(0) @ fq.Z(4))
)
DEFAULT_PARAMS = (0.013, 0.026, 0.039, 0.052)


@qfunc
def cost_layer(gamma: float, target: QArray[QBit]) -> None:
    suzuki_trotter(HAMILTONIAN, gamma, 1, 1, target)


@qfunc
def main(params_list: list[float], target: Output[QArray[QBit]]) -> None:
    allocate(6, target)
    hadamard_transform(target)
    for layer in range(2):
        cost_layer(params_list[2 * layer], target)
        for bit in range(6):
            RX(params_list[2 * layer + 1], target[bit])


def build_program(params: tuple[float, ...] = DEFAULT_PARAMS) -> synth.Program:
    if len(params) != PARAMETER_COUNT:
        raise ValueError("This retained QAOA program requires 4 parameters")
    return synth.synthesize(
        main, classical_inputs={"params_list": params}, max_workspace=0
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(program.report["resources"])
    print("State norm:", result.statevector().norm().item())
