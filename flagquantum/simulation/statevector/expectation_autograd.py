"""Hermitian Pauli expectation derivatives without complex-product graphs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch


class _Context(Protocol):
    operators: tuple[tuple[int, str], ...]
    n_qubits: int

    @property
    def saved_tensors(self) -> tuple[torch.Tensor, ...]: ...

    @property
    def needs_input_grad(self) -> tuple[bool, ...]: ...

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...


def _transform(
    state: torch.Tensor,
    operators: tuple[tuple[int, str], ...],
    n_qubits: int,
) -> torch.Tensor:
    from ..matrices import X_MATRIX, Y_MATRIX, Z_MATRIX
    from .local import _apply_gate_matrix

    matrices = {"x": X_MATRIX, "y": Y_MATRIX, "z": Z_MATRIX}
    transformed = state
    for qubit, name in operators:
        matrix = matrices[name.lower()].to(device=state.device, dtype=state.dtype)
        transformed = _apply_gate_matrix(
            transformed,
            matrix,
            (qubit,),
            n_qubits,
            uses_diagonal_kernel=False,
        )
    return transformed


class _PauliExpectation(torch.autograd.Function):
    """For Hermitian P, the real expectation has gradient 2 P state."""

    @staticmethod
    def forward(
        ctx: _Context,
        state: torch.Tensor,
        operators: tuple[tuple[int, str], ...],
        n_qubits: int,
    ) -> torch.Tensor:
        transformed = _transform(state, operators, n_qubits)
        ctx.save_for_backward(state, transformed)
        ctx.operators, ctx.n_qubits = operators, n_qubits
        return (
            torch.view_as_real(state.resolve_conj())
            * torch.view_as_real(transformed.resolve_conj())
        ).sum(dim=(-1, -2))

    @staticmethod
    def backward(
        ctx: _Context,
        gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
        state, transformed = ctx.saved_tensors
        if torch.is_grad_enabled():
            transformed = _transform(state, ctx.operators, ctx.n_qubits)
        return transformed * (2 * gradient).reshape(-1, 1), None, None


def _apply_pauli_expectation(*args: object) -> torch.Tensor:
    apply: Callable[..., object] = _PauliExpectation.apply
    result = apply(*args)
    if not isinstance(result, torch.Tensor):
        raise TypeError("autograd action must return a tensor")
    return result
