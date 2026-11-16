"""Compose Simon's fixed two-to-one oracle from high-level arithmetic."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Const,
    Output,
    QNum,
    allocate,
    hadamard_transform,
    min,
    qfunc,
)


@qfunc
def representative(x: Const[QNum], res: Output[QNum]) -> None:
    res |= min(x, x ^ 6)


@qfunc
def main(x: Output[QNum], res: Output[QNum]) -> None:
    allocate(5, x)
    hadamard_transform(x)
    representative(x, res)
    hadamard_transform(x)


def build_program() -> synth.Program:
    return synth.synthesize(main, max_workspace=0)


def run_example() -> None:
    program = build_program()
    print(dict(program.probabilities(fq.run(program), registers=("x", "res"))))
    print(program.report["resources"])


if __name__ == "__main__":
    run_example()
