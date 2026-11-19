"""Apply the fresh phases to the original six-bit arithmetic search."""

import math
from itertools import product
from typing import Annotated

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


def matching_words() -> tuple[tuple[int, int, int], ...]:
    """Enumerate the original equations on every fixed-width input word."""
    return tuple(
        (a, b, c)
        for a, b, c in product(range(4), range(2), range(8))
        if a + b == 3 and c - a == 2
    )


def build_program(phases: tuple[float, ...], *, max_qubits: int = 12) -> synth.Program:
    """Bind original numeric fields and exact disjoint target predicates."""
    if len(phases) != 26 or any(not math.isfinite(angle) for angle in phases):
        raise ValueError("Expected 26 finite real phases")
    (a0, b0, c0), (a1, b1, c1) = matching_words()

    @qfunc
    def amplify(
        phases: list[float],
        a0: int,
        b0: int,
        c0: int,
        a1: int,
        b1: int,
        c1: int,
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
        for step in range(13):
            hadamard_transform(a)
            hadamard_transform(b)
            hadamard_transform(c)
            with within():
                initial: Annotated[QNum, 1] = (a < 1) & (b < 1) & (c < 1)
            with apply():
                phase(aux - 0.5, phases[2 * step])
                phase((1 - 2 * aux) * initial, phases[2 * step])
            hadamard_transform(a)
            hadamard_transform(b)
            hadamard_transform(c)
            with within():
                match: Annotated[QNum, 1] = (
                    (a <= a0) & (a >= a0) & (b == b0) & (c == c0)
                ) | ((a <= a1) & (a >= a1) & (b == b1) & (c == c1))
            with apply():
                phase(aux - 0.5, phases[2 * step + 1])
                phase((1 - 2 * aux) * match, phases[2 * step + 1])
        H(aux)

    return synth.synthesize(
        amplify,
        classical_inputs={
            "phases": phases,
            "a0": a0,
            "b0": b0,
            "c0": c0,
            "a1": a1,
            "b1": b1,
            "c1": c1,
        },
        max_qubits=max_qubits,
        max_workspace=5,
        max_operations=1024,
        memory_limit_bytes=1024**2,
    )
