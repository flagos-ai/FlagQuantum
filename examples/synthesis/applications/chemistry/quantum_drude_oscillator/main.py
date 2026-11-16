"""Build the retained six-qubit Drude-oscillator variational quantum program."""

import math
from collections.abc import Sequence

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import CX, RZ, Output, QArray, QBit, R, allocate, qfunc


@qfunc
def ansatz_layer(params: list[float], state: QArray[QBit]) -> None:
    RZ(1.570796326794897, state[0])
    RZ(1.570796326794897, state[1])
    R(1.570796326794897, math.pi / 2, state[1])
    CX(state[1], state[0])
    R(params[0], math.pi / 2, state[0])
    RZ(params[1], state[0])
    R(params[2], math.pi / 2, state[1])
    RZ(params[3], state[1])
    CX(state[1], state[0])
    R(-1.570796326794897, math.pi / 2, state[1])
    RZ(-1.570796326794897, state[0])
    RZ(-1.570796326794897, state[1])


@qfunc
def main(params: list[float], state: Output[QArray[QBit]]) -> None:
    allocate(6, state)
    ansatz_layer([params[0], params[1], params[2], params[3]], state[0:2])
    ansatz_layer([params[4], params[5], params[6], params[7]], state[2:4])
    ansatz_layer([params[8], params[9], params[10], params[11]], state[4:6])
    ansatz_layer([params[12], params[13], params[14], params[15]], state[1:3])
    ansatz_layer([params[16], params[17], params[18], params[19]], state[3:5])
    ansatz_layer([params[20], params[21], params[22], params[23]], state[0:2])
    ansatz_layer([params[24], params[25], params[26], params[27]], state[2:4])
    ansatz_layer([params[28], params[29], params[30], params[31]], state[4:6])
    ansatz_layer([params[32], params[33], params[34], params[35]], state[1:3])
    ansatz_layer([params[36], params[37], params[38], params[39]], state[3:5])


def build_program(params: Sequence[float]) -> synth.Program:
    """Preserve the source's forty parameters and ten ordered pair layers."""
    if len(params) != 40:
        raise ValueError("the source ansatz requires exactly forty parameters")
    return synth.synthesize(main, classical_inputs={"params": list(params)})


if __name__ == "__main__":
    program = build_program([0.01 * (index + 1) for index in range(40)])
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("state",))))
