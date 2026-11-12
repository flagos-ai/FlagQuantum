"""Catalog-authorized exact CX permutations with inverse-sequence derivatives."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

import torch

from ...kernels.catalog import KernelRequest, match_kernel_implementations
from ..kernel_dispatch import _require_cataloged_kernel


class _Context(Protocol):
    controls: tuple[int, ...]
    targets: tuple[int, ...]
    n_wires: int

    @property
    def saved_tensors(self) -> tuple[torch.Tensor, ...]: ...

    @property
    def needs_input_grad(self) -> tuple[bool, ...]: ...

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...


class _CXSequence(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: _Context,
        state: torch.Tensor,
        controls: tuple[int, ...],
        targets: tuple[int, ...],
        n_wires: int,
    ) -> torch.Tensor:
        request = KernelRequest(
            semantic_id="statevector.apply.cnot_sequence.local",
            device=state.device.type,
            dtype=str(state.dtype).removeprefix("torch."),
            layout="flat_statevector",
            direction="forward",
            addressing=("local",),
            providers=("triton",),
        )
        _require_cataloged_kernel(
            match_kernel_implementations(request),
            implementation_id="FQKI-TRITON-SV-003-C",
            description="double-precision CX kernel",
        )
        from ...kernels.triton.statevector_cx_fp64 import apply_cx

        ctx.controls, ctx.targets, ctx.n_wires = controls, targets, n_wires
        return apply_cx(state, controls, targets, n_wires)

    @staticmethod
    def backward(
        ctx: _Context, gradient: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        return (
            _apply_cx_sequence(
                gradient, ctx.controls[::-1], ctx.targets[::-1], ctx.n_wires
            ),
            None,
            None,
            None,
        )


def apply_cx_sequence(
    state: torch.Tensor, controls: Sequence[int], targets: Sequence[int], n_wires: int
) -> torch.Tensor:
    """Preserve arbitrary-order input derivatives through a reversible permutation."""
    return _apply_cx_sequence(state, tuple(controls), tuple(targets), n_wires)


def _apply_cx_sequence(*args: object) -> torch.Tensor:
    apply: Callable[..., object] = _CXSequence.apply
    result = apply(*args)
    if not isinstance(result, torch.Tensor):
        raise TypeError("autograd action must return a tensor")
    return result
