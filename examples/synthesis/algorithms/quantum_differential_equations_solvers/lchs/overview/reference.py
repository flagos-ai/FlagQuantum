"""Build the complete small-block LCHS state with independent dense algebra."""

import cmath
import math

import torch

from examples.synthesis.algorithms.hamiltonian_simulation.hamiltonian_simulation_with_block_encoding.hamiltonian_simulation_gqsp.overview.phases import (
    Factorization,
)

from .model import DTYPE, SHIFT, Grid


def _ry(angle: float) -> torch.Tensor:
    cosine, sine = math.cos(angle / 2), math.sin(angle / 2)
    return torch.tensor(((cosine, -sine), (sine, cosine)), dtype=DTYPE)


def _unitary_preparation(amplitudes: torch.Tensor) -> torch.Tensor:
    size = amplitudes.numel()
    width = size.bit_length() - 1
    state = torch.eye(size, dtype=DTYPE)
    for level in range(width):
        suffix = 1 << (width - level - 1)
        updated = torch.zeros((size, size), dtype=DTYPE)
        for prefix in range(1 << level):
            start = 2 * prefix * suffix
            first = amplitudes[start : start + suffix].norm()
            second = amplitudes[start + suffix : start + 2 * suffix].norm()
            angle = 2 * math.atan2(float(second), float(first))
            group = slice(start, start + 2 * suffix)
            updated[group, group] = torch.kron(
                _ry(angle), torch.eye(suffix, dtype=DTYPE)
            )
        state = updated @ state
    return state


def _block_diag(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    boundary = first.shape[0]
    width = boundary + second.shape[0]
    result = torch.zeros((width, width), dtype=DTYPE)
    result[:boundary, :boundary] = first
    result[boundary:, boundary:] = second
    return result


def _dense_walks(grid: Grid) -> torch.Tensor:
    h = torch.tensor(((1, 1), (1, -1)), dtype=DTYPE) / math.sqrt(2)
    x = torch.tensor(((0, 1), (1, 0)), dtype=DTYPE)
    z = torch.diag(torch.tensor((1, -1), dtype=DTYPE))
    id2 = torch.eye(2, dtype=DTYPE)
    id4 = torch.eye(4, dtype=DTYPE)
    mix = _ry(2 * math.atan2(1, math.sqrt(grid.radius)))
    prep = torch.kron(torch.kron(torch.kron(mix, id2), h), id4)
    reflect = torch.diag(torch.tensor((1,) * 4 + (-1,) * 28, dtype=DTYPE))
    low = _block_diag(id4, torch.kron(id2, z))
    high = torch.kron(id2, _block_diag(torch.kron(x, x), torch.kron(z, z)))
    blocks = []
    for j in grid.signed:
        amplitude = j / 32
        off = math.copysign(math.sqrt(1 - amplitude**2), j)
        linear = torch.tensor(((amplitude, off), (off, -amplitude)), dtype=DTYPE)
        select = _block_diag(torch.kron(linear, low), high)
        blocks.append(reflect @ prep.mH @ select @ prep)
    return torch.stack(blocks)


def _rotation(row: tuple[float, float, float]) -> torch.Tensor:
    theta, phi, lam = row
    left = torch.diag(torch.tensor((cmath.exp(1j * phi), 1), dtype=DTYPE))
    right = torch.diag(torch.tensor((cmath.exp(1j * lam), -1), dtype=DTYPE))
    return left @ _ry(2 * theta) @ right


def dense_state(
    grid: Grid, input_state: torch.Tensor, fit: Factorization
) -> torch.Tensor:
    """Independently evolve every grid block and reverse the kernel preparation."""
    walks = _dense_walks(grid)
    state = torch.zeros((64, 32, 2), dtype=DTYPE)
    state[:, :4, 0] = grid.kernel_amplitudes[:, None] * input_state.to(DTYPE)[None]
    state = state @ _rotation(fit.rows[0]).T
    degree = (len(fit.rows) - 1) // 2
    for step, row in enumerate(fit.rows[1:], 1):
        branch = 1 if step <= degree else 0
        operation = walks.mH if branch else walks
        state[:, :, branch] = (operation @ state[:, :, branch, None]).squeeze(-1)
        state = state @ _rotation(row).T
    phases = torch.exp(-1j * SHIFT * torch.from_numpy(grid.points))
    state = state * phases[:, None, None]
    reverse = _unitary_preparation(grid.kernel_amplitudes).mH
    return (reverse @ state.reshape(64, -1)).flatten()
