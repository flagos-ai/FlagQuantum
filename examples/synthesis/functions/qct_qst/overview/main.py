"""Retain all four sine/cosine examples with source-seeded real amplitudes."""

from math import pi, sqrt
from typing import Literal

import numpy as np
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    H,
    Output,
    QBit,
    QNum,
    allocate,
    prepare_amplitudes,
    qct_qst_type1,
    qct_qst_type2,
    qct_type2,
    qfunc,
    qst_type2,
)


@qfunc
def type_one(data: list[float], x: Output[QNum]) -> None:
    prepare_amplitudes(data, 0, x)
    qct_qst_type1(x)


@qfunc
def type_two(data: list[float], x: Output[QNum], q: Output[QBit]) -> None:
    prepare_amplitudes(data, 0, x)
    allocate(q)
    H(q)
    qct_qst_type2(x, q)


@qfunc
def cosine(data: list[float], x: Output[QNum]) -> None:
    prepare_amplitudes(data, 0, x)
    qct_type2(x)


@qfunc
def sine(data: list[float], x: Output[QNum]) -> None:
    prepare_amplitudes(data, 0, x)
    qst_type2(x)


def original_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Preserve the source RNG and sequential draws of lengths 9, 7 and 8."""
    random = np.random.RandomState(123)
    vectors = []
    for length in (9, 7, 8):
        vector = random.rand(length)
        vector /= np.linalg.norm(vector)
        vectors.append(torch.tensor(vector, dtype=torch.float64))
    return vectors[0], vectors[1], vectors[2]


def dct(size: int, kind: Literal[1, 2]) -> torch.Tensor:
    """Classical orthonormal cosine matrix; Type II rows are frequency modes."""
    if (
        type(size) is not int
        or not 2 <= size <= 256
        or type(kind) is not int
        or kind not in (1, 2)
    ):
        raise ValueError(
            "This local reference supports sizes 2..256 and transform types 1 or 2."
        )
    j = torch.arange(size, dtype=torch.float64)
    if kind == 1:
        weights = torch.ones(size, dtype=torch.float64)
        weights[0] = weights[-1] = 1 / sqrt(2)
        return (
            sqrt(2 / (size - 1))
            * torch.cos(pi * torch.outer(j, j) / (size - 1))
            * torch.outer(weights, weights)
        )
    weights = torch.ones(size, dtype=torch.float64)
    weights[0] = 1 / sqrt(2)
    return (
        sqrt(2 / size)
        * torch.cos(pi * torch.outer(j, j + 0.5) / size)
        * weights[:, None]
    )


def dst(size: int, kind: Literal[1, 2]) -> torch.Tensor:
    """Classical orthonormal sine matrix; Type II includes the final-mode weight."""
    if (
        type(size) is not int
        or not 2 <= size <= 256
        or type(kind) is not int
        or kind not in (1, 2)
    ):
        raise ValueError(
            "This local reference supports sizes 2..256 and transform types 1 or 2."
        )
    j = torch.arange(size, dtype=torch.float64)
    if kind == 1:
        return sqrt(2 / (size + 1)) * torch.sin(
            pi * torch.outer(j + 1, j + 1) / (size + 1)
        )
    weights = torch.ones(size, dtype=torch.float64)
    weights[-1] = 1 / sqrt(2)
    return (
        sqrt(2 / size)
        * torch.sin(pi * torch.outer(j + 1, j + 0.5) / size)
        * weights[:, None]
    )


def run_workflow() -> dict[str, synth.Program]:
    cos_data, sin_data, data = original_data()
    combined = torch.cat((cos_data, sin_data)) / sqrt(2)
    programs = {}
    for name, entry, amplitudes in (
        ("type_one", type_one, combined),
        ("type_two", type_two, data),
        ("cosine", cosine, data),
        ("sine", sine, data),
    ):
        program = synth.synthesize(
            entry,
            classical_inputs={"data": amplitudes.tolist()},
            max_qubits=4,
            max_workspace=0,
            memory_limit_bytes=1024**2,
        )
        state = fq.run(program).statevector().flatten()
        if name == "type_one":
            expected = torch.cat(
                (dct(9, 1) @ cos_data, 1j * (dst(7, 1) @ sin_data))
            ) / sqrt(2)
        elif name == "type_two":
            expected = torch.stack(
                (dct(8, 2) @ data, -(dst(8, 2) @ data)), dim=1
            ).flatten() / sqrt(2)
        elif name == "cosine":
            expected = dct(8, 2) @ data
        else:
            expected = dst(8, 2) @ data
        torch.testing.assert_close(
            state, expected.to(torch.complex128), atol=1e-12, rtol=0
        )
        print(
            name,
            "max absolute error",
            float((state - expected).abs().max()),
            "resources",
            program.report["resources"],
            flush=True,
        )
        print("Measured absolute complex amplitudes:", state.tolist())
        print("Expected absolute complex amplitudes:", expected.tolist())
        programs[name] = program
    return programs


def run_example() -> None:
    run_workflow()


if __name__ == "__main__":
    run_example()
