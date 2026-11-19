"""Evaluate the four-slice product in independent tensor axes."""

from collections.abc import Sequence

import torch

from examples.synthesis.algorithms.quantum_differential_equations_solvers.time_marching.comparisons.circuit import (
    step_gates,
)

from .approximation import COMPLEX


def apply_gate(
    state: torch.Tensor, wires: Sequence[int], matrix: torch.Tensor
) -> torch.Tensor:
    """Apply one reference matrix to explicit axes of an eleven-qubit state."""
    dimensions = tuple(range(11))
    rest = tuple(wire for wire in dimensions if wire not in wires)
    order = (*wires, *rest)
    inverse = tuple(order.index(axis) for axis in dimensions)
    data = state.reshape((2,) * 11).permute(order).reshape(1 << len(wires), -1)
    transformed: torch.Tensor = matrix @ data
    return transformed.reshape((2,) * 11).permute(inverse).flatten()


def reference_state(
    even: tuple[float, ...],
    odd: tuple[float, ...],
    amp: tuple[float, ...] | None,
) -> torch.Tensor:
    """Follow real RY, signed projector phases and explicit counter ordering."""
    state = torch.zeros((4, 32, 2, 8), dtype=COMPLEX)
    state[:, 0, 0, 4] = 0.5
    state = state.flatten()
    h = torch.tensor([[1, 1], [1, -1]], dtype=COMPLEX) * 2**-0.5
    block_sign = -torch.ones(32, dtype=COMPLEX)
    block_sign[0] = 1

    def rotate_phase(vector: torch.Tensor, angle: float) -> torch.Tensor:
        factors = torch.stack(
            ((1j * angle * block_sign).exp(), (-1j * angle * block_sign).exp()), dim=1
        )
        return (vector.reshape(4, 32, 2, 8) * factors[None, :, :, None]).flatten()

    for i in range(4):
        gates = step_gates(
            i / 2,
            (i + 1) / 2,
            tuple(2 * angle for angle in even),
            tuple(2 * angle for angle in odd),
        )
        if amp is None:
            for wires, matrix in gates:
                state = apply_gate(state, wires, matrix)
        else:
            state = apply_gate(state, (7,), h)
            state = rotate_phase(state, amp[0])
            for position, angle in enumerate(amp[1:]):
                if position % 2:
                    for wires, matrix in reversed(gates):
                        state = apply_gate(state, wires, matrix.mH)
                else:
                    for wires, matrix in gates:
                        state = apply_gate(state, wires, matrix)
                state = rotate_phase(state, angle)
            state = apply_gate(state, (7,), h)
        encoded = state.reshape(4, 32, 2, 8).clone()
        if amp is None:
            encoded[:, 0] = encoded[:, 0].roll(-1, dims=-1)
        else:
            encoded[:, 0, 0] = encoded[:, 0, 0].roll(-1, dims=-1)
        state = encoded.flatten()
    return state
