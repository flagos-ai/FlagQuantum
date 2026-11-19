"""Encode the source signed-grid Hamiltonians with reusable native operations."""

import math
from typing import Annotated

import torch

import flagquantum as fq
from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.hamiltonian_simulation_gqsp.overview.phases import (
    Factorization,
)
from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.hamiltonian_simulation_qubitization.overview.quantum import (
    preparation,
)
from examples.synthesis.algorithms.quantum_primitives.gqsp.overview.phases import (
    rotation,
)
from flagquantum import synth
from flagquantum.algorithms.primitives.multiplexers.select_pauli_lcu import (
    SelectPauliLCU,
)
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    QNum,
    X,
    allocate,
    control,
    phase,
    qfunc,
    select_rotation,
)

from .model import DTYPE, GRID_BITS, GRID_WORDS, SHIFT, Grid

GRID = tuple(range(6))
MIX, LINEAR, PAULI = 6, 7, 8
DATA = (9, 10)
SIGNAL = 11
WIDTH = 12
PAULI_WORDS = ("II", "IZ", "XX", "ZZ")


@qfunc
def _linear_block(
    angles: list[float], selector: Output[QArray[QBit]], target: Output[QBit]
) -> None:
    allocate(8, selector)
    allocate(target)
    select_rotation("Y", angles, selector, target)
    X(selector[6])
    control(selector[6:8], lambda: X(target))
    X(selector[6])
    phase(selector[7] * (1 - selector[6]) * selector[5], math.pi)


@qfunc
def _kernel_phase(coefficient: float, j: Output[Annotated[QNum, (6, True, 0)]]) -> None:
    allocate(j)
    phase(j, coefficient)


def _linear_circuit(grid: Grid) -> fq.Circuit:
    active = tuple(
        2 * math.asin(abs(signed) / (GRID_WORDS // 2)) for signed in grid.signed
    )
    angles = (0.0,) * (2 * GRID_WORDS) + active + (0.0,) * GRID_WORDS
    program = synth.synthesize(
        _linear_block,
        classical_inputs={"angles": angles},
        max_qubits=9,
        max_workspace=0,
        max_operations=4096,
        memory_limit_bytes=1024**2,
    )
    return fq.Circuit.from_ir(program.circuit, dtype=DTYPE)


def controlled_walk(grid: Grid) -> tuple[fq.Circuit, fq.Circuit]:
    """Build one H/L SELECT and its reflection-conjugated inverse."""
    block = fq.Circuit(WIDTH, dtype=DTYPE)
    mix_angle = 2 * math.atan2(1, math.sqrt(grid.radius))
    block.ry(MIX, theta=mix_angle).h(PAULI)
    block.compose(_linear_circuit(grid), qubits=(SIGNAL, MIX, *GRID, LINEAR))
    block.append(
        SelectPauliLCU(2, 2, PAULI_WORDS, control_val=1).make_on(
            selection=(MIX, PAULI), target=DATA, control=(SIGNAL,)
        )
    )
    block.h(PAULI).ry(MIX, theta=-mix_angle)
    reflection = torch.diag(torch.tensor((1,) * 9 + (-1,) * 7, dtype=DTYPE))
    wires = (SIGNAL, MIX, LINEAR, PAULI)
    forward = (
        fq.Circuit(WIDTH, dtype=DTYPE).compose(block).any(*wires, matrix=reflection)
    )
    backward = (
        fq.Circuit(WIDTH, dtype=DTYPE).any(*wires, matrix=reflection).compose(block)
    )
    return forward, backward


def build_program(grid: Grid, state: torch.Tensor, factor: Factorization) -> fq.Circuit:
    """Prepare the grid kernel, apply signed GQSP selections and unprepare."""
    weights = preparation(tuple(map(float, grid.kernel_amplitudes)))
    initial = preparation(tuple(map(float, state)))
    forward, backward = controlled_walk(grid)
    circuit = fq.Circuit(WIDTH, dtype=DTYPE).compose(initial, qubits=DATA)
    circuit.compose(weights, qubits=GRID)
    circuit.any(SIGNAL, matrix=torch.from_numpy(rotation(factor.rows[0])))
    degree = (len(factor.rows) - 1) // 2
    for step, row in enumerate(factor.rows[1:], start=1):
        if step <= degree:
            circuit.compose(backward)
        else:
            circuit.x(SIGNAL).compose(forward).x(SIGNAL)
        circuit.any(SIGNAL, matrix=torch.from_numpy(rotation(row)))
    phase_program = synth.synthesize(
        _kernel_phase,
        classical_inputs={"coefficient": -grid.step * SHIFT},
        max_qubits=GRID_BITS,
        max_workspace=0,
    )
    circuit.compose(fq.Circuit.from_ir(phase_program.circuit, dtype=DTYPE), qubits=GRID)
    return circuit.compose(weights.adjoint(), qubits=GRID)
