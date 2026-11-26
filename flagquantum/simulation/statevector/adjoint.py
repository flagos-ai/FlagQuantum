"""Pure local numerics for statevector adjoint differentiation."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from flagquantum.core.operators.schema import canonical_opcode

from ...core.ir import Instruction
from ..native_cpu import fused_observable_adjoint_seed, fused_rotation_adjoint_
from .operations import (
    _DIAGONAL_STATEVECTOR_GATES,
    _apply_diagonal_matrix,
    _apply_matrix,
)


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
    n_qubits: int,
    qubit: int,
) -> torch.Tensor:
    """Return one amplitude chunk's contribution to a Z expectation."""

    bit = (global_indices >> (n_qubits - qubit - 1)) & 1
    signs = (1 - 2 * bit).to(dtype=amplitudes.real.dtype)
    return (amplitudes.abs().square() * signs.reshape(1, -1)).sum()


def z_expectation_adjoint_chunk(
    amplitudes: torch.Tensor,
    global_indices: torch.Tensor,
    *,
    n_qubits: int,
    qubit: int,
) -> torch.Tensor:
    """Return one amplitude chunk's derivative of a Z expectation."""

    bit = (global_indices >> (n_qubits - qubit - 1)) & 1
    signs = (1 - 2 * bit).to(dtype=amplitudes.real.dtype)
    return 2 * amplitudes * signs.reshape(1, -1)


def z_hamiltonian_chunk(
    amplitudes: torch.Tensor,
    global_indices: torch.Tensor,
    *,
    n_qubits: int,
    terms: Sequence[tuple[float, tuple[int, ...]]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one chunk's weighted Z/ZZ expectation and adjoint seed."""

    weights = z_hamiltonian_weights(
        global_indices,
        n_qubits=n_qubits,
        terms=terms,
        dtype=amplitudes.real.dtype,
    )
    shaped = weights.reshape(1, -1)
    value = (amplitudes.abs().square() * shaped).sum()
    return value, 2 * amplitudes * shaped


def z_hamiltonian_weights(
    global_indices: torch.Tensor,
    *,
    n_qubits: int,
    terms: Sequence[tuple[float, tuple[int, ...]]],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return the real diagonal weights for a Z/ZZ Hamiltonian chunk."""

    weights = torch.zeros(
        global_indices.shape,
        dtype=dtype,
        device=global_indices.device,
    )
    for coefficient, qubits in terms:
        if len(qubits) == 1:
            parity = global_indices >> (n_qubits - qubits[0] - 1)
        elif len(qubits) == 2:
            first = global_indices >> (n_qubits - qubits[0] - 1)
            second = global_indices >> (n_qubits - qubits[1] - 1)
            parity = first ^ second
        else:
            parity = torch.zeros_like(global_indices)
            for qubit in qubits:
                parity = parity ^ (global_indices >> (n_qubits - qubit - 1))
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


def _is_diagonal_instruction(instruction: Instruction) -> bool:
    """Whether an instruction provably applies a diagonal matrix.

    A carried matrix takes precedence over the operator the name denotes, so a
    name from the diagonal table says nothing about an instruction carrying one.
    """

    return (
        instruction.matrix is None
        and canonical_opcode(instruction.name) in _DIAGONAL_STATEVECTOR_GATES
    )


def _pair_components(
    state: torch.Tensor, *, wire: int, n_wires: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the zero/one views for one qubit of a contiguous state."""

    stride = 1 << (int(n_wires) - int(wire) - 1)
    paired = state.reshape(state.shape[0], -1, 2, stride)
    return paired[:, :, 0, :], paired[:, :, 1, :]


def _complex_cross(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Return Im(conj(left) * right) without materializing a complex product."""

    return left.real * right.imag - left.imag * right.real


def _single_process_cpu_pauli_rotation_gradient(
    ket: torch.Tensor,
    adjoint: torch.Tensor,
    instruction: Instruction,
    *,
    n_wires: int,
) -> torch.Tensor | None:
    """Evaluate a Pauli-rotation VJP directly from the post-gate ket and bra."""

    name = instruction.name
    wires = instruction.wires
    if name in {"rx", "ry", "rz"} and len(wires) == 1:
        ket_zero, ket_one = _pair_components(ket, wire=wires[0], n_wires=n_wires)
        adjoint_zero, adjoint_one = _pair_components(
            adjoint, wire=wires[0], n_wires=n_wires
        )
        if name == "rx":
            return 0.5 * (
                _complex_cross(adjoint_zero, ket_one).sum()
                + _complex_cross(adjoint_one, ket_zero).sum()
            )
        if name == "ry":
            zero = (
                adjoint_zero.real * ket_one.real + adjoint_zero.imag * ket_one.imag
            ).sum()
            one = (
                adjoint_one.real * ket_zero.real + adjoint_one.imag * ket_zero.imag
            ).sum()
            return 0.5 * (one - zero)
        return 0.5 * (
            _complex_cross(adjoint_zero, ket_zero).sum()
            - _complex_cross(adjoint_one, ket_one).sum()
        )
    if name == "rzz" and len(wires) == 2:
        cross = _complex_cross(adjoint, ket)
        ordered = tuple(int(wire) for wire in wires)
        rest = tuple(wire for wire in range(n_wires) if wire not in ordered)
        perm = (0,) + tuple(wire + 1 for wire in ordered + rest)
        basis = (
            cross.reshape((cross.shape[0],) + (2,) * n_wires)
            .permute(perm)
            .reshape(cross.shape[0], 4, -1)
        )
        return 0.5 * (
            basis[:, 0].sum()
            - basis[:, 1].sum()
            - basis[:, 2].sum()
            + basis[:, 3].sum()
        )
    return None


def _fused_single_process_cpu_reversible_vjp(
    ket: torch.Tensor,
    adjoint: torch.Tensor,
    matrix: torch.Tensor,
    instruction: Instruction,
    *,
    n_wires: int,
) -> torch.Tensor | None:
    """Reverse ket and bra while avoiding a derivative-state materialization."""

    if len(instruction.wires) == 1 or (
        instruction.name == "rzz" and len(instruction.wires) == 2
    ):
        native_gradient = fused_rotation_adjoint_(
            ket,
            adjoint,
            matrix,
            name=instruction.name,
            qubit=instruction.wires[0],
            second_qubit=(
                instruction.wires[1] if len(instruction.wires) == 2 else None
            ),
            n_qubits=n_wires,
        )
        if native_gradient is not None:
            return native_gradient

    gradient = _single_process_cpu_pauli_rotation_gradient(
        ket, adjoint, instruction, n_wires=n_wires
    )
    if gradient is None:
        return None
    inverse = matrix.mH
    apply = (
        _apply_diagonal_matrix
        if _is_diagonal_instruction(instruction)
        else _apply_matrix
    )
    previous_ket = apply(ket, inverse, instruction.wires, n_wires)
    previous_adjoint = apply(adjoint, inverse, instruction.wires, n_wires)
    ket.copy_(previous_ket)
    adjoint.copy_(previous_adjoint)
    return gradient


def seed_adjoint(amplitudes: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Seed the CPU adjoint natively when supported, with an eager fallback."""

    seed = fused_observable_adjoint_seed(amplitudes, weights)
    return seed if seed is not None else 2 * amplitudes * weights.reshape(1, -1)
