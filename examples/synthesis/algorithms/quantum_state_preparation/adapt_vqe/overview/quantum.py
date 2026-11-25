"""Synthesize bounded source ansatz schedules with native Pauli evolution."""

import math
from collections.abc import Sequence

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Output, QArray, QBit, allocate, qfunc, suzuki_trotter

from .model import DTYPE, OUTER_BUDGET, POOL, observable


@qfunc
def _reference(v: Output[QArray[QBit]]) -> None:
    allocate(4, v)


def build_program(indices: tuple[int, ...], angles: Sequence[float]) -> synth.Program:
    """Bind up to ten source pool selections and the corresponding angles."""
    count = len(indices)
    if (
        count != len(angles)
        or count > OUTER_BUDGET
        or any(
            type(index) is not int or not 0 <= index < len(POOL) for index in indices
        )
        or not all(math.isfinite(angle) for angle in angles)
    ):
        raise ValueError(
            "Expected up to ten valid source pool selections and finite angles"
        )
    if not count:
        return synth.synthesize(_reference, max_qubits=4, max_workspace=0)
    generators = tuple(observable(POOL[index]) for index in indices)
    g0, g1, g2, g3, g4, g5, g6, g7, g8, g9 = generators + (generators[0],) * (
        OUTER_BUDGET - count
    )

    @qfunc
    def ansatz(params: list[float], v: Output[QArray[QBit]]) -> None:
        allocate(4, v)
        suzuki_trotter(g0, params[0], 1, 1, v)
        if count > 1:
            suzuki_trotter(g1, params[1], 1, 1, v)
        if count > 2:
            suzuki_trotter(g2, params[2], 1, 1, v)
        if count > 3:
            suzuki_trotter(g3, params[3], 1, 1, v)
        if count > 4:
            suzuki_trotter(g4, params[4], 1, 1, v)
        if count > 5:
            suzuki_trotter(g5, params[5], 1, 1, v)
        if count > 6:
            suzuki_trotter(g6, params[6], 1, 1, v)
        if count > 7:
            suzuki_trotter(g7, params[7], 1, 1, v)
        if count > 8:
            suzuki_trotter(g8, params[8], 1, 1, v)
        if count > 9:
            suzuki_trotter(g9, params[9], 1, 1, v)

    return synth.synthesize(
        ansatz,
        classical_inputs={"params": tuple(angles)},
        max_qubits=4,
        max_workspace=0,
        max_operations=140,
        memory_limit_bytes=1024**2,
    )


def state(program: synth.Program) -> torch.Tensor:
    """Read signed complex amplitudes in the source register's packed order."""
    values: torch.Tensor = fq.run(program).statevector().flatten()
    return values.to(DTYPE)
