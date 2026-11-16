"""Native symmetric-weight state, GF(2) encoding and coherent DQI decoding."""

import math

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    CX,
    H,
    Output,
    QArray,
    QBit,
    X,
    allocate,
    control,
    inplace_prepare_amplitudes,
    phase,
    qfunc,
)

from .data import PARITY_COLUMNS, SIGNS, SYNDROME_TABLE, WEIGHTS


@qfunc
def gf2_product(matrix: list[int], y: QArray[QBit], solution: QArray[QBit]) -> None:
    for row in range(6):
        for column in range(6):
            if matrix[6 * row + column] == 1:
                CX(y[column], solution[row])


@qfunc
def controlled_flip(syndrome: QArray[QBit], target: QBit) -> None:
    control(syndrome, lambda: X(target))


@qfunc
def decode(
    syndrome_bits: list[int],
    mask_bits: list[int],
    y: QArray[QBit],
    solution: QArray[QBit],
) -> None:
    for entry in range(22):
        for bit in range(6):
            if syndrome_bits[6 * entry + bit] == 0:
                X(solution[bit])
        for bit in range(6):
            if mask_bits[6 * entry + bit] == 1:
                controlled_flip(solution, y[bit])
        for bit in range(6):
            if syndrome_bits[6 * entry + bit] == 0:
                X(solution[bit])


@qfunc
def main(
    amplitudes: list[float],
    matrix: list[int],
    signs: list[int],
    syndrome_bits: list[int],
    mask_bits: list[int],
    y: Output[QArray[QBit]],
    solution: Output[QArray[QBit]],
) -> None:
    allocate(6, y)
    allocate(6, solution)
    inplace_prepare_amplitudes(amplitudes, 0, y)
    for bit in range(6):
        phase(y[bit], math.pi * signs[bit])
    gf2_product(matrix, y, solution)
    decode(syndrome_bits, mask_bits, y, solution)
    for bit in range(6):
        H(solution[bit])


def symmetric_amplitudes(weights: tuple[float, ...]) -> list[float]:
    """Embed the retained weight-zero/one/two superposition into six bits."""
    if len(weights) != 4 or any(
        type(value) not in (int, float) or abs(value) > 1 or not math.isfinite(value)
        for value in weights
    ):
        raise ValueError("weights must be four finite real amplitudes in [-1, 1]")
    if weights[3] != 0:
        raise ValueError("the retained decoder supports weight zero through two only")
    if not math.isclose(
        math.fsum(value * value for value in weights), 1, rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError("weight amplitudes must be normalized")
    return [
        weights[k] / math.sqrt(math.comb(6, k)) if k <= 2 else 0.0
        for word in range(64)
        for k in (word.bit_count(),)
    ]


def build_program(weights: tuple[float, ...] = WEIGHTS) -> synth.Program:
    return synth.synthesize(
        main,
        classical_inputs={
            "amplitudes": symmetric_amplitudes(weights),
            "matrix": [
                (column >> row) & 1 for row in range(6) for column in PARITY_COLUMNS
            ],
            "signs": list(SIGNS),
            "syndrome_bits": [
                (word >> bit) & 1 for word, _ in SYNDROME_TABLE for bit in range(6)
            ],
            "mask_bits": [
                (mask >> bit) & 1 for _, mask in SYNDROME_TABLE for bit in range(6)
            ],
        },
        max_qubits=15,
        max_workspace=3,
        memory_limit_bytes=64 * 1024**2,
    )


if __name__ == "__main__":
    program = build_program()
    result = fq.run(program)
    print(dict(program.probabilities(result, registers=("y", "solution"))))
    print(program.report["resources"])
