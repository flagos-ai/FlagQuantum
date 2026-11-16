"""Keep the fifty-control intent and let the checked planner choose exact MPS."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QArray, QBit, X, allocate, control, qfunc


@qfunc
def my_mcx(cntrl: QArray[QBit], target: QBit) -> None:
    control(cntrl, lambda: X(target))


@qfunc
def main(cntrl: Output[QArray[QBit]], target: Output[QBit]) -> None:
    allocate(50, cntrl)
    allocate(target)
    my_mcx(cntrl, target)


def build_program() -> synth.Program:
    return synth.synthesize(main, memory_limit_bytes=2048, max_workspace=13)


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(program.report["execution_profile"])
    print(dict(program.probabilities(result, registers=("target",))))
