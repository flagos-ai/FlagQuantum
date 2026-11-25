"""Compose exact signed/scaled quantum arithmetic without selecting gates."""

from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    Const,
    Output,
    QNum,
    allocate,
    hadamard_transform,
    phase,
    qfunc,
)


@qfunc
def twice(value: Const[QNum], result: Output[QNum]) -> None:
    result |= 2 * value


@qfunc
def offset(value: Const[QNum], result: Output[QNum]) -> None:
    result |= value + 0.5


@qfunc
def main(
    x: Output[Annotated[QNum, (1, True, 1)]],
    doubled: Output[QNum],
    result: Output[QNum],
) -> None:
    allocate(x)
    hadamard_transform(x)
    phase(x + 0.125, 0.37)
    twice(x, doubled)
    offset(doubled, result)


def build_program() -> synth.Program:
    """Synthesize an exact two-branch numeric program with explicit budgets."""
    return synth.synthesize(main, max_qubits=24, memory_limit_bytes=64 * 1024**2)


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(
        dict(
            program.numeric_probabilities(result, registers=("x", "doubled", "result"))
        )
    )
    print(program.report["resources"])
