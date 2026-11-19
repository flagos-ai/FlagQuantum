"""Run retained and coherent seven-control X programs using native qfunc."""

import flagquantum as fq
from examples.synthesis.functions.mcx.mcx_example.main import build_program
from flagquantum import synth
from flagquantum.synth import (
    RZ,
    H,
    Output,
    QArray,
    QBit,
    S,
    X,
    allocate,
    control,
    hadamard_transform,
    qfunc,
)


@qfunc
def coherent(cntrl: Output[QArray[QBit]], target: Output[QBit]) -> None:
    allocate(7, cntrl)
    allocate(target)
    hadamard_transform(cntrl)
    S(cntrl[0])
    H(target)
    RZ(0.73, target)
    control(cntrl, lambda: X(target))


def run_example() -> None:
    for program in (build_program(), synth.synthesize(coherent)):
        result = fq.run(program)
        print(
            "Probabilities:",
            dict(program.probabilities(result, registers=("cntrl", "target"))),
        )
        print("Resources:", program.report["resources"])


if __name__ == "__main__":
    run_example()
