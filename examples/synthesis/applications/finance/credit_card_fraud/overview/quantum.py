"""Reuse the original three-feature Pauli embedding for all selected rows."""

import math
from typing import Annotated

import torch

import flagquantum as fq
from examples.synthesis.applications.finance.credit_card_fraud.main import (
    main as overlap_root,
)
from examples.synthesis.applications.finance.credit_card_fraud.main import (
    pauli_kernel,
)
from flagquantum import synth
from flagquantum.synth import Output, QNum, allocate, qfunc


@qfunc
def encode(data: list[float], q: Output[Annotated[QNum, 3]]) -> None:
    allocate(q)
    pauli_kernel(data, q)


def checked(values: tuple[float, ...]) -> tuple[float, ...]:
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("Three finite PCA coordinates are required")
    return values


def overlap(left: tuple[float, ...], right: tuple[float, ...]) -> synth.Program:
    return synth.synthesize(
        overlap_root,
        classical_inputs={"data1": checked(left), "data2": checked(right)},
        max_qubits=3,
        max_workspace=0,
        max_operations=96,
        memory_limit_bytes=1024**2,
    )


def state(data: tuple[float, ...]) -> torch.Tensor:
    program = synth.synthesize(
        encode,
        classical_inputs={"data": checked(data)},
        max_qubits=3,
        max_workspace=0,
        max_operations=64,
        memory_limit_bytes=1024**2,
    )
    vector: torch.Tensor = fq.run(program).statevector().flatten()
    return vector
