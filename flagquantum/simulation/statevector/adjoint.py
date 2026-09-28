"""Pure local numerics for statevector adjoint differentiation."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ...core.ir import Instruction


def analytic_rotation_derivative(
    instruction: Instruction,
    matrix: torch.Tensor,
) -> torch.Tensor | None:
    """Return the matrix derivative for a supported Pauli rotation."""

    generators = {
        "rx": ((0, 1), (1, 0)),
        "ry": ((0, -1j), (1j, 0)),
        "rz": ((1, 0), (0, -1)),
        "rzz": (
            (1, 0, 0, 0),
            (0, -1, 0, 0),
            (0, 0, -1, 0),
            (0, 0, 0, 1),
        ),
    }
    values = generators.get(instruction.name)
    if values is None or tuple(instruction.params) != ("theta",):
        return None
    generator = torch.tensor(values, dtype=matrix.dtype, device=matrix.device)
    return (-0.5j) * (generator @ matrix)


def real_conjugate_inner_sum(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return the real part of the conjugate inner product without ``conj``."""

    return torch.sum(left.real * right.real + left.imag * right.imag)


def z_expectation_chunk(
    amplitudes: torch.Tensor,
    global_indices: torch.Tensor,
    *,
    n_wires: int,
    wire: int,
) -> torch.Tensor:
    """Return one amplitude chunk's contribution to a Z expectation."""

    bit = (global_indices >> (n_wires - wire - 1)) & 1
    signs = (1 - 2 * bit).to(dtype=amplitudes.real.dtype)
    return (amplitudes.abs().square() * signs.reshape(1, -1)).sum()


def z_expectation_adjoint_chunk(
    amplitudes: torch.Tensor,
    global_indices: torch.Tensor,
    *,
    n_wires: int,
    wire: int,
) -> torch.Tensor:
    """Return one amplitude chunk's derivative of a Z expectation."""

    bit = (global_indices >> (n_wires - wire - 1)) & 1
    signs = (1 - 2 * bit).to(dtype=amplitudes.real.dtype)
    return 2 * amplitudes * signs.reshape(1, -1)


def z_hamiltonian_chunk(
    amplitudes: torch.Tensor,
    global_indices: torch.Tensor,
    *,
    n_wires: int,
    terms: Sequence[tuple[float, tuple[int, ...]]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one chunk's weighted Z/ZZ expectation and adjoint seed."""

    weights = z_hamiltonian_weights(
        global_indices,
        n_wires=n_wires,
        terms=terms,
        dtype=amplitudes.real.dtype,
    )
    shaped = weights.reshape(1, -1)
    value = (amplitudes.abs().square() * shaped).sum()
    return value, 2 * amplitudes * shaped


def z_hamiltonian_weights(
    global_indices: torch.Tensor,
    *,
    n_wires: int,
    terms: Sequence[tuple[float, tuple[int, ...]]],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return the real diagonal weights for a Z/ZZ Hamiltonian chunk."""

    weights = torch.zeros(
        global_indices.shape,
        dtype=dtype,
        device=global_indices.device,
    )
    for coefficient, wires in terms:
        if len(wires) == 1:
            parity = global_indices >> (n_wires - wires[0] - 1)
        elif len(wires) == 2:
            first = global_indices >> (n_wires - wires[0] - 1)
            second = global_indices >> (n_wires - wires[1] - 1)
            parity = first ^ second
        else:
            parity = torch.zeros_like(global_indices)
            for wire in wires:
                parity = parity ^ (global_indices >> (n_wires - wire - 1))
        signs = (1 - 2 * (parity & 1)).to(dtype=weights.dtype)
        weights = weights + float(coefficient) * signs
    return weights


__all__ = (
    "analytic_rotation_derivative",
    "real_conjugate_inner_sum",
    "z_expectation_adjoint_chunk",
    "z_expectation_chunk",
    "z_hamiltonian_chunk",
    "z_hamiltonian_weights",
)
