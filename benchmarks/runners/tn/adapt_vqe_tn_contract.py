"""Shared workload contract for the FlagQuantum/TC-NG ADAPT-VQE comparison."""

from __future__ import annotations

import math
import numpy as np


def grid_edges(rows: int, cols: int) -> tuple[tuple[int, int], ...]:
    edges = []
    for row in range(rows):
        for col in range(cols):
            wire = row * cols + col
            if col + 1 < cols:
                edges.append((wire, wire + 1))
            if row + 1 < rows:
                edges.append((wire, wire + cols))
    return tuple(edges)


def mean_field_angles(
    rows: int, cols: int, *, tolerance: float = 1e-14, max_iterations: int = 10_000
) -> tuple[float, ...]:
    """Find a deterministic low-energy real product state for this TFIM."""
    qubits = rows * cols
    neighbors = [[] for _ in range(qubits)]
    for left, right in grid_edges(rows, cols):
        neighbors[left].append(right)
        neighbors[right].append(left)
    angles = np.zeros(qubits, dtype=np.float64)
    for _ in range(max_iterations):
        z = np.cos(angles)
        updated = np.asarray([
            math.atan2(
                0.7,
                sum(z[other] for other in neighbors[wire])
                - 0.11 * (-1.0 if wire % 2 else 1.0),
            )
            for wire in range(qubits)
        ], dtype=np.float64)
        next_angles = 0.5 * angles + 0.5 * updated
        if float(np.max(np.abs(next_angles - angles))) <= tolerance:
            return tuple(map(float, next_angles))
        angles = next_angles
    raise RuntimeError("mean-field initial-state solver did not converge")


def product_state_energy(rows: int, cols: int, angles: tuple[float, ...]) -> float:
    values = np.asarray(angles, dtype=np.float64)
    z, x = np.cos(values), np.sin(values)
    energy = -sum(z[left] * z[right] for left, right in grid_edges(rows, cols))
    energy -= 0.7 * float(np.sum(x))
    energy += 0.11 * sum(
        (-1.0 if wire % 2 else 1.0) * z[wire] for wire in range(len(z))
    )
    return float(energy)


def initial_angles(rows: int, cols: int, initial_state: str) -> tuple[float, ...]:
    if initial_state == "mean_field":
        return mean_field_angles(rows, cols)
    if initial_state == "zero":
        return (0.0,) * (rows * cols)
    if initial_state == "legacy_h_ry":
        return tuple(0.07 * (wire + 1) for wire in range(rows * cols))
    raise ValueError(f"unsupported initial state: {initial_state}")
