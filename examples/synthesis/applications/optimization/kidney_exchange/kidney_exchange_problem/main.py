"""Retain the original assignment costs and every QAOA layer in native qfunc."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    RX,
    Output,
    QArray,
    QBit,
    allocate,
    hadamard_transform,
    phase,
    qfunc,
)


@qfunc
def cost(gamma: float, weights: list[float], matches: QArray[QBit]) -> None:
    for donor in range(3):
        for patient in range(3):
            phase(weights[3 * donor + patient] * matches[3 * donor + patient], gamma)
        for first in range(3):
            for second in range(first + 1, 3):
                phase(
                    -2 * matches[3 * donor + first] * matches[3 * donor + second], gamma
                )
                phase(
                    -2 * matches[3 * first + donor] * matches[3 * second + donor], gamma
                )


@qfunc
def main(params: list[float], matches: Output[QArray[QBit]]) -> None:
    allocate(9, matches)
    hadamard_transform(matches)
    for layer in range(5):
        cost(params[layer], [0.9, 0.7, 0.6, 0.8, 0.75, 0.65, 0.85, 0.8, 0.7], matches)
        for bit in range(9):
            RX(params[5 + layer], matches[bit])


DEFAULT_PARAMS = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1)


def build_program(params: tuple[float, ...] = DEFAULT_PARAMS) -> synth.Program:
    if len(params) != 10:
        raise ValueError("Use 10 real parameters for all 5 retained layers")
    return synth.synthesize(main, classical_inputs={"params": params}, max_workspace=0)


def run_example() -> None:
    program = build_program()
    print(program.report["resources"])
    print("State norm:", fq.run(program).statevector().norm().item())


if __name__ == "__main__":
    run_example()
