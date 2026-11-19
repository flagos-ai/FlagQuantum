"""Run every original GQSP walk selection on the complete seventeen-bit state."""

import time
from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.hamiltonian_simulation_gqsp.overview.phases import (
    Factorization,
)
from examples.synthesis.algorithms.quantum_primitives.gqsp.overview.phases import (
    rotation,
)

from . import phases, physics, quantum


@dataclass(frozen=True)
class Result:
    initial: torch.Tensor
    state: torch.Tensor
    projected: torch.Tensor
    classical: torch.Tensor
    fidelity: float
    absolute_error: float
    projection_norm: float
    classical_defect: float
    operations_executed: int
    execute_seconds: float


def prepare_initial() -> fq.Circuit:
    spatial = physics.initial_state()[:1024].real
    difference = -spatial.clone()
    difference[0] += 1
    difference /= difference.norm()
    unitary = (
        torch.eye(1024, dtype=torch.complex128)
        - 2 * difference[:, None] * difference[None, :]
    )
    return fq.Circuit(17, dtype=torch.complex128).any(*range(7, 17), matrix=unitary)


def run(prepared: Factorization | None = None) -> Result:
    fitted = phases.fit() if prepared is None else prepared
    if len(fitted.rows) != 1373:
        raise ValueError("The original effective time requires all 1373 phase rows")
    walk = quantum.controlled_walk()
    inverse = walk.adjoint()
    forward = fq.Circuit(17, dtype=torch.complex128).x(0).compose(walk).x(0)
    circuit = prepare_initial()
    circuit.any(0, matrix=torch.from_numpy(rotation(fitted.rows[0])))
    started = time.monotonic()
    state = circuit.state().reshape(-1)
    operations = len(circuit.to_ir().instructions)
    for step, row in enumerate(fitted.rows[1:], 1):
        selected = inverse if step <= phases.DEGREE else forward
        circuit = fq.Circuit(17, dtype=torch.complex128, inputs=state).compose(selected)
        circuit.any(0, matrix=torch.from_numpy(rotation(row)))
        state = circuit.state().reshape(-1)
        operations += len(circuit.to_ir().instructions)
        if step % 16 == 0 or step == 1372:
            print(f"Executed {step}/1372 original walk selections", flush=True)
    duration = time.monotonic() - started
    projected = state.reshape(2, 16, 4096)[0, 0]
    classical, defect = physics.evolve()
    fidelity = float(
        torch.abs(torch.vdot(projected, classical)) ** 2
        / (projected.norm().square() * classical.norm().square())
    )
    error = float((projected - phases.SCALE * classical).abs().max())
    if 1 - fidelity >= 1e-4 or error >= 1e-6:
        raise ValueError(
            "The original full-grid Maxwell evolution missed its tolerance"
        )
    return Result(
        physics.initial_state(),
        state,
        projected,
        classical,
        fidelity,
        error,
        float(projected.norm()),
        defect,
        operations,
        duration,
    )
