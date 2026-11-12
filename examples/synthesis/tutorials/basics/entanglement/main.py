"""Native high-level counterpart of the exact entanglement.qmod original."""

import flagquantum as fq
from flagquantum import synth
from flagquantum.drawer import draw
from flagquantum.synth import CX, H, Output, QArray, QBit, allocate, qfunc


@qfunc
def bell_pair(reg: QArray[QBit]) -> None:
    H(reg[0])
    CX(reg[0], reg[1])


@qfunc
def bell(registers: Output[QArray[QBit]]) -> None:
    allocate(2, registers)
    bell_pair(registers)


@qfunc
def exercise(reg: Output[QArray[QBit]]) -> None:
    """Keep the Notebook's blank six-qubit exercise available to modify."""
    allocate(6, reg)


@qfunc
def main(reg: Output[QArray[QBit]]) -> None:
    allocate(6, reg)
    H(reg[0])
    for index in range(len(reg) - 1):
        CX(reg[index], reg[index + 1])


def run_example() -> None:
    """Build and draw the Bell example, blank exercise and GHZ solution."""
    for label, function, register in (
        ("Bell pair", bell, "registers"),
        ("Blank six-qubit exercise", exercise, "reg"),
        ("Six-qubit GHZ solution", main, "reg"),
    ):
        program = synth.synthesize(function)
        print(label)
        print(draw(program.circuit))
        result = fq.run(program)
        print(dict(program.probabilities(result, registers=(register,))))


if __name__ == "__main__":
    run_example()
