"""Independent five-wire branch circuits and rank-one Grover reference."""

import math

import torch

import flagquantum as fq

from . import model


def branch_state(row: model.Path, transform: model.Encoding) -> torch.Tensor:
    """Realize the original conditional reference preparation and marker XOR."""
    circuit = fq.Circuit(transform.width + 1, dtype=torch.complex128)
    target = transform.width
    if row.payment is not model.Payment.PUT:
        circuit.ry(target, transform.rotation(row.discounted_payoff))
        rotated_state: torch.Tensor = fq.run(circuit).statevector().flatten()
        return rotated_state
    for bit in range(transform.width):
        circuit.ry(
            target - 1 - bit, 2 * math.atan(math.exp((1 << bit) / (2 * model.UNIT)))
        )
    dimension = 1 << (target + 1)
    order = list(range(dimension))
    threshold = row.log_returns[-1] + (1 << (target - 1))
    for word in range(threshold + 1):
        order[2 * word], order[2 * word + 1] = order[2 * word + 1], order[2 * word]
    circuit.any(
        *range(target + 1), matrix=torch.eye(dimension, dtype=torch.complex128)[order]
    )
    integrated_state: torch.Tensor = fq.run(circuit).statevector().flatten()
    return integrated_state


def prepared_state(
    grid: model.Grid, transform: model.Encoding, paths: tuple[model.Path, ...]
) -> torch.Tensor:
    """Use canonical indicator/grid/grid/grid/reference ordering."""
    branches = [
        math.sqrt(row.probability) * branch_state(row, transform) for row in paths
    ]
    grid_size = len(grid.points)
    return (
        torch.stack(branches)
        .reshape(grid_size, grid_size, grid_size, 1 << transform.width, 2)
        .permute(4, 0, 1, 2, 3)
        .flatten()
    )


def grover_once(prepared: torch.Tensor) -> torch.Tensor:
    marked = prepared.clone()
    marked[marked.numel() // 2 :] *= -1
    return 2 * torch.vdot(prepared, marked) * prepared - marked
