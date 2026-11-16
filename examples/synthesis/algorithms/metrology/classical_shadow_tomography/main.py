"""Native high-level counterpart of the exact classical_shadow_tomography.qmod original."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import CX, H, Output, QArray, QBit, S, prepare_bell_state, qfunc


@qfunc
def clifford_application(state: QArray[QBit]) -> None:
    S(state[0])
    H(state[0])
    S(state[1])
    S(state[1])
    S(state[0])
    H(state[1])
    CX(state[0], state[1])
    CX(state[0], state[1])
    S(state[0])
    H(state[1])
    S(state[0])
    H(state[0])
    S(state[0])
    S(state[0])
    CX(state[1], state[0])
    H(state[1])
    H(state[1])
    S(state[0])
    S(state[0])
    S(state[0])


@qfunc
def main(qarr: Output[QArray[QBit]]) -> None:
    prepare_bell_state(1, qarr)
    clifford_application(qarr)


if __name__ == "__main__":
    program = synth.synthesize(main)
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("qarr",))))
