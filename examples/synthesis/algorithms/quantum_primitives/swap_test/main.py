"""Overlap measurement with private quantum states and explicit environment drop."""

import math

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    CX,
    H,
    Output,
    QArray,
    QBit,
    allocate,
    drop,
    phase,
    prepare_amplitudes,
    qfunc,
)

RAW_FIRST = (0.41, -0.17, 0.30, 0.04, 0.55, -0.34, -0.33, 0.54)
RAW_SECOND = (-0.30, 0.53, 0.38, 0.06, -0.29, -0.19, 0.67, 0.14)


def normalize(values: tuple[float, ...]) -> tuple[float, ...]:
    norm = math.hypot(*values)
    if len(values) != 8 or not norm or not math.isfinite(norm):
        raise ValueError("use eight finite nonzero amplitudes")
    return tuple(v / norm for v in values)


@qfunc
def controlled_swap(test: QBit, a: QBit, b: QBit) -> None:
    CX(b, a)
    H(b)
    phase(test * a * b, math.pi)
    H(b)
    CX(b, a)


@qfunc
def overlap_test(first: QArray[QBit], second: QArray[QBit], test: QBit) -> None:
    H(test)
    for bit in range(3):
        controlled_swap(test, first[bit], second[bit])
    H(test)


@qfunc
def main(first: list[float], second: list[float], test: Output[QBit]) -> None:
    state1: QArray[QBit] = QArray()
    state2: QArray[QBit] = QArray()
    prepare_amplitudes(first, 0, state1)
    prepare_amplitudes(second, 0, state2)
    allocate(test)
    overlap_test(state1, state2, test)
    drop(state1)
    drop(state2)


def build_program(
    first: tuple[float, ...] = RAW_FIRST, second: tuple[float, ...] = RAW_SECOND
) -> synth.Program:
    """Explicitly normalize rounded source data, matching the approved revision."""
    return synth.synthesize(
        main,
        classical_inputs={
            "first": list(normalize(first)),
            "second": list(normalize(second)),
        },
        max_workspace=0,
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("test",))))
    print(program.report["resources"])
