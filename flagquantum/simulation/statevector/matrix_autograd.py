"""Layout-controlled complex128 gradients for small statevector gate matrices."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch


class _Context(Protocol):
    qubit: int
    n_qubits: int

    @property
    def saved_tensors(self) -> tuple[torch.Tensor, ...]: ...

    @property
    def needs_input_grad(self) -> tuple[bool, ...]: ...

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...


class _GateMatrixBMM(torch.autograd.Function):
    """Preserve the forward operation and pack its long gradient contraction."""

    @staticmethod
    def forward(
        ctx: _Context, matrix: torch.Tensor, amplitudes: torch.Tensor
    ) -> torch.Tensor:
        ctx.save_for_backward(matrix, amplitudes)
        return torch.bmm(matrix, amplitudes)

    @staticmethod
    def backward(
        ctx: _Context, gradient: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        matrix, amplitudes = ctx.saved_tensors
        matrix_gradient = None
        amplitude_gradient = None
        if ctx.needs_input_grad[0]:
            # The reduction dimension must be contiguous in both operands.
            # Packing before conjugation retains the backend's conjugate view.
            matrix_gradient = torch.bmm(
                gradient.contiguous(),
                amplitudes.contiguous().transpose(1, 2).conj(),
            )
        if ctx.needs_input_grad[1]:
            amplitude_gradient = torch.bmm(matrix.transpose(1, 2).conj(), gradient)
        return matrix_gradient, amplitude_gradient


def apply_gate_matrix_bmm(
    matrix: torch.Tensor, amplitudes: torch.Tensor
) -> torch.Tensor:
    """Pack gradients for large flagos complex128 single-qubit gates."""

    if (
        amplitudes.device.type == "flagos"
        and amplitudes.dtype == torch.complex128
        and matrix.dtype == amplitudes.dtype
        and matrix.shape[-2:] == (2, 2)
        and amplitudes.shape[-1] >= 1 << 16
        and torch.is_grad_enabled()
        and matrix.requires_grad
    ):
        return _apply_gate_matrix_bmm(matrix, amplitudes)
    return torch.bmm(matrix, amplitudes)


class _SingleWireMatrix(torch.autograd.Function):
    """Apply one-qubit matrices without permuting the full state in forward."""

    @staticmethod
    def forward(
        ctx: _Context,
        state: torch.Tensor,
        matrix: torch.Tensor,
        qubit: int,
        n_qubits: int,
    ) -> torch.Tensor:
        from .operations import _apply_single_wire_matrix

        ctx.save_for_backward(state, matrix)
        ctx.qubit = qubit
        ctx.n_qubits = n_qubits
        return _apply_single_wire_matrix(state, matrix, qubit, n_qubits)

    @staticmethod
    def backward(
        ctx: _Context, gradient: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None]:
        from .operations import _apply_single_wire_matrix

        state, matrix = ctx.saved_tensors
        qubit, n_qubits = ctx.qubit, ctx.n_qubits
        state_gradient = None
        matrix_gradient = None
        if ctx.needs_input_grad[0]:
            state_gradient = _apply_single_wire_matrix(
                gradient, matrix.transpose(-1, -2).conj(), qubit, n_qubits
            )
        if ctx.needs_input_grad[1]:
            batch = state.shape[0]
            outer, inner = 1 << qubit, 1 << (n_qubits - qubit - 1)
            before = state.reshape(batch, outer, 2, inner)
            after = gradient.reshape(batch, outer, 2, inner)
            before = before.permute(0, 2, 1, 3).reshape(batch, 2, -1).contiguous()
            after = after.permute(0, 2, 1, 3).reshape(batch, 2, -1).contiguous()
            matrix_gradient = torch.bmm(after, before.transpose(1, 2).conj())
            if matrix.ndim == 2:
                matrix_gradient = matrix_gradient.sum(dim=0)
            elif matrix.shape[0] == 1 and batch != 1:
                matrix_gradient = matrix_gradient.sum(dim=0, keepdim=True)
        return state_gradient, matrix_gradient, None, None


def apply_single_qubit_matrix(
    state: torch.Tensor, matrix: torch.Tensor, qubit: int, n_qubits: int
) -> torch.Tensor:
    """Use layout-preserving forward and reverse on complex128 statevectors."""

    from .operations import _apply_single_wire_matrix

    matrix = matrix.to(device=state.device, dtype=state.dtype)
    if torch.is_grad_enabled() and (state.requires_grad or matrix.requires_grad):
        return _apply_single_qubit_autograd(state, matrix, qubit, n_qubits)
    return _apply_single_wire_matrix(state, matrix, qubit, n_qubits)


def _apply_gate_matrix_bmm(*args: object) -> torch.Tensor:
    apply: Callable[..., object] = _GateMatrixBMM.apply
    result = apply(*args)
    if not isinstance(result, torch.Tensor):
        raise TypeError("autograd action must return a tensor")
    return result


def _apply_single_qubit_autograd(*args: object) -> torch.Tensor:
    apply: Callable[..., object] = _SingleWireMatrix.apply
    result = apply(*args)
    if not isinstance(result, torch.Tensor):
        raise TypeError("autograd action must return a tensor")
    return result
