"""Five-qubit postselection circuit for the original three-term VQLS LCU."""

import math

import torch

from examples.synthesis.algorithms.quantum_linear_solvers.vqls.vqls_with_lcu.main import (
    ansatz,
)
from flagquantum import synth
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    QNum,
    adjoint,
    allocate,
    control,
    hadamard_transform,
    inplace_prepare_amplitudes,
    phase,
    qfunc,
)

AMPLITUDES = (math.sqrt(0.55), math.sqrt(0.225), math.sqrt(0.225), 0.0)
MATRIX = torch.tensor(
    [
        0.55 + 0.225 * (1 - 2 * ((j >> 1) & 1)) + 0.225 * (1 - 2 * ((j >> 2) & 1))
        for j in range(8)
    ],
    dtype=torch.float64,
)
TARGET = torch.ones(8, dtype=torch.float64) / math.sqrt(8)


@qfunc
def lcu_prepare(weights: list[float], selector: QNum) -> None:
    inplace_prepare_amplitudes(weights, 0, selector)


@qfunc
def objective(
    params: list[float],
    weights: list[float],
    auxiliary: Output[QNum],
    system: Output[QArray[QBit]],
) -> None:
    allocate(2, auxiliary)
    allocate(3, system)
    ansatz(params, system)
    lcu_prepare(weights, auxiliary)
    control(auxiliary == 1, lambda: phase(system[1], math.pi))
    control(auxiliary == 2, lambda: phase(system[2], math.pi))
    with adjoint():
        lcu_prepare(weights, auxiliary)
    hadamard_transform(system)


def build_objective(params: tuple[float, ...]) -> synth.Program:
    if len(params) != 9 or any(
        type(value) not in (int, float) or not math.isfinite(value) for value in params
    ):
        raise ValueError("expected nine finite source angles")
    return synth.synthesize(
        objective,
        classical_inputs={"params": params, "weights": AMPLITUDES},
        max_qubits=5,
        max_workspace=0,
        max_operations=2048,
        memory_limit_bytes=1024**2,
    )
