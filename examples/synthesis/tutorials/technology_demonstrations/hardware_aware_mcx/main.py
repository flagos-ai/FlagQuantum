"""Express all-ones control without designing the multi-control decomposition."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QArray, QBit, X, allocate, control, qfunc


@qfunc
def main(cntrl: Output[QArray[QBit]], target: Output[QBit]) -> None:
    allocate(15, cntrl)
    allocate(target)
    control(cntrl, lambda: X(target))


def build_program() -> synth.Program:
    return synth.synthesize(main, memory_limit_bytes=512 * 1024**2)


if __name__ == "__main__":
    program = build_program()
    print(dict(program.probabilities(fq.run(program), registers=("target",))))
