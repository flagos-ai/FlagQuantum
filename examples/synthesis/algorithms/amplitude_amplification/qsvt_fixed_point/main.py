"""Fixed-point amplification with exact projected phases and flattened fields."""

from typing import Annotated

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    H,
    Output,
    QBit,
    QNum,
    allocate,
    apply,
    hadamard_transform,
    phase,
    qfunc,
    within,
)

PHASES = (
    -1.433699528629295,
    3.083538811696024,
    3.212401795341816,
    3.055563303483763,
    3.246000344455865,
    3.014602346197096,
    3.29714073365046,
    2.948423145716331,
    3.387183429437787,
    2.816633064497373,
    3.603434404330445,
    2.38372208112054,
    4.896965176813679,
    4.89696517681368,
    2.38372208112054,
    3.603434404330444,
    2.816633064497374,
    3.387183429437787,
    2.948423145716331,
    3.29714073365046,
    3.014602346197096,
    3.246000344455865,
    3.055563303483763,
    3.212401795341816,
    3.083538811696024,
    -76.83192321478433,
)


@qfunc
def main(
    phases: list[float],
    a: Output[QNum],
    b: Output[QNum],
    c: Output[QNum],
    aux: Output[QBit],
) -> None:
    allocate(2, a)
    allocate(1, b)
    allocate(3, c)
    allocate(aux)
    hadamard_transform(a)
    hadamard_transform(b)
    hadamard_transform(c)
    H(aux)
    # The original identity block alternates uniform-state and target projectors.
    for step in range(13):
        hadamard_transform(a)
        hadamard_transform(b)
        hadamard_transform(c)
        with within():
            initial_mark: Annotated[QNum, 1] = (a < 1) & (b < 1) & (c < 1)
        with apply():
            phase(aux - 0.5, phases[2 * step])
            phase((1 - 2 * aux) * initial_mark, phases[2 * step])
        hadamard_transform(a)
        hadamard_transform(b)
        hadamard_transform(c)
        with within():
            # a-c == -2 becomes a+2 == c without unsigned underflow.
            target_mark: Annotated[QNum, 1] = (
                ((a + b) <= 3) & ((a + b) >= 3) & ((a + 2) == c)
            )
        with apply():
            phase(aux - 0.5, phases[2 * step + 1])
            phase((1 - 2 * aux) * target_mark, phases[2 * step + 1])
    H(aux)


def build_program(phases: tuple[float, ...] = PHASES) -> synth.Program:
    """Preserve the concrete source's 26 phases and six-bit OracleVars layout."""
    if len(phases) != 26:
        raise ValueError("this fixed-point program requires 26 phases")
    return synth.synthesize(
        main, classical_inputs={"phases": list(phases)}, max_operations=100_000
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("a", "b", "c", "aux"))))
    print(program.report["resources"])
