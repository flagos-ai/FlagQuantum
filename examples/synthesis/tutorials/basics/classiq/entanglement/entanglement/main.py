"""Native high-level counterpart of the exact entanglement.qmod original."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import CX, H, Output, QArray, QBit, allocate, qfunc


@qfunc
def main(reg: Output[QArray[QBit]]) -> None:
    allocate(6, reg)
    H(reg[0])
    for index in range(len(reg) - 1):
        CX(reg[index], reg[index + 1])


if __name__ == "__main__":
    program = synth.synthesize(main)
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("reg",))))
