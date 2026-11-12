"""Exact double-precision RY derivatives in the interleaved complex layout."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

import torch

from ...core.operator_schema import canonical_opcode

if TYPE_CHECKING:
    from ...circuit import Circuit
    from ...core.ir import Instruction


class _Context(Protocol):
    wire: int
    n_wires: int

    @property
    def saved_tensors(self) -> tuple[torch.Tensor, ...]: ...

    @property
    def needs_input_grad(self) -> tuple[bool, ...]: ...

    def save_for_backward(self, *tensors: torch.Tensor) -> None: ...


def _require_fp64_kernel(state: torch.Tensor, backward: bool = False) -> None:
    from ...kernels.catalog import KernelRequest, match_kernel_implementations
    from ..kernel_dispatch import _require_cataloged_kernel

    request = KernelRequest(
        semantic_id=(
            "gradient.vjp.ry.local" if backward else "statevector.apply.ry.local"
        ),
        device=state.device.type,
        dtype=str(state.dtype).removeprefix("torch."),
        layout="flat_statevector",
        direction="vjp" if backward else "forward",
        addressing=("local",),
        providers=("triton",),
    )
    _require_cataloged_kernel(
        match_kernel_implementations(request),
        implementation_id=(
            "FQKI-TRITON-GR-090-A" if backward else "FQKI-TRITON-SV-090-A"
        ),
        description="double-precision RY kernel",
    )


def _apply_real_ry(
    state: torch.Tensor, matrix: torch.Tensor, wire: int, n_wires: int
) -> torch.Tensor:
    """Apply the real RY matrix to both float64 components of complex128."""
    if (
        state.device.type == "flagos"
        and state.dtype == torch.complex128
        and not torch.is_grad_enabled()
    ):
        _require_fp64_kernel(state)
        from ...kernels.triton.statevector_ry_fp64 import forward

        return forward(state, matrix.reshape(-1, 2, 2), wire, n_wires)
    batch = state.shape[0]
    values = torch.view_as_real(state.resolve_conj()).reshape(
        batch, 1 << wire, 2, 1 << (n_wires - wire - 1), 2
    )
    low, high = values[:, :, 0], values[:, :, 1]
    entries = matrix.real.reshape(-1, 2, 2)
    a = entries[:, 0, 0].reshape(-1, 1, 1, 1)
    b = entries[:, 0, 1].reshape(-1, 1, 1, 1)
    c = entries[:, 1, 0].reshape(-1, 1, 1, 1)
    d = entries[:, 1, 1].reshape(-1, 1, 1, 1)
    if torch.is_grad_enabled():
        result = torch.stack((a * low + b * high, c * low + d * high), dim=2)
    else:
        result = torch.empty_like(values)
        out_low, out_high = result[:, :, 0], result[:, :, 1]
        torch.mul(a, low, out=out_low)
        out_low.addcmul_(high, b)
        torch.mul(c, low, out=out_high)
        out_high.addcmul_(high, d)
    return torch.view_as_complex(result.reshape(batch, -1, 2))


class _RYRotation(torch.autograd.Function):
    """Differentiate RY directly instead of constructing four matrix gradients."""

    @staticmethod
    def forward(
        ctx: _Context,
        state: torch.Tensor,
        parameters: torch.Tensor,
        matrix: torch.Tensor,
        wire: int,
        n_wires: int,
    ) -> torch.Tensor:
        result = _apply_real_ry(state, matrix, wire, n_wires)
        ctx.save_for_backward(result, parameters)
        ctx.wire, ctx.n_wires = wire, n_wires
        return result

    @staticmethod
    def backward(
        ctx: _Context, gradient: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None, None]:
        output, parameters = ctx.saved_tensors
        wire, n_wires = ctx.wire, ctx.n_wires
        state_gradient = parameter_gradient = None
        if gradient.device.type == "flagos" and not torch.is_grad_enabled():
            _require_fp64_kernel(gradient, backward=True)
            from ...kernels.triton.statevector_ry_fp64 import backward

            half_angle = parameters[:, 0] * 0.5
            cosine, sine = half_angle.cos(), half_angle.sin()
            real = torch.stack((cosine, -sine, sine, cosine), dim=-1).reshape(-1, 2, 2)
            matrix = torch.complex(real, torch.zeros_like(real))
            state_gradient, parameter_gradient = backward(
                gradient,
                output,
                matrix,
                wire,
                n_wires,
                state=ctx.needs_input_grad[0],
                angle=ctx.needs_input_grad[1],
            )
            if (
                parameter_gradient is not None
                and parameters.shape[0] == 1
                and output.shape[0] != 1
            ):
                parameter_gradient = parameter_gradient.sum(dim=0, keepdim=True)
            return state_gradient, parameter_gradient, None, None, None
        if ctx.needs_input_grad[0]:
            half_angle = parameters[:, 0] * 0.5
            cosine, sine = half_angle.cos(), half_angle.sin()
            matrix_real = torch.stack((cosine, sine, -sine, cosine), dim=-1)
            matrix_real = matrix_real.reshape(-1, 2, 2)
            matrix = torch.complex(matrix_real, torch.zeros_like(matrix_real))
            state_gradient = _apply_real_ry(gradient, matrix, wire, n_wires)
        if ctx.needs_input_grad[1]:
            shape = (output.shape[0], 1 << wire, 2, 1 << (n_wires - wire - 1), 2)
            values = torch.view_as_real(output.resolve_conj()).reshape(shape)
            grads = torch.view_as_real(gradient.resolve_conj()).reshape(shape)
            products = torch.addcmul(
                grads[:, :, 1] * values[:, :, 0],
                grads[:, :, 0],
                values[:, :, 1],
                value=-1,
            )
            parameter_gradient = products.sum(dim=(1, 2, 3)).reshape(-1, 1) * 0.5
            if parameters.shape[0] == 1 and output.shape[0] != 1:
                parameter_gradient = parameter_gradient.sum(dim=0, keepdim=True)
        return state_gradient, parameter_gradient, None, None, None


def _apply_ry_rotation(*args: object) -> torch.Tensor:
    apply: Callable[..., object] = _RYRotation.apply
    result = apply(*args)
    if not isinstance(result, torch.Tensor):
        raise TypeError("autograd action must return a tensor")
    return result


def try_apply_ry_rotation(
    circuit: Circuit,
    instruction: Instruction,
    output: torch.Tensor,
    matrix: torch.Tensor,
    parameter_bindings: tuple[torch.Tensor, ...] | None,
    carries_matrix: bool,
) -> torch.Tensor | None:
    """Select the exact analytic RY action only for its measured state layout."""
    if (
        canonical_opcode(instruction.name) != "ry"
        or carries_matrix
        or output.device.type != "flagos"
        or output.dtype != torch.complex128
        or output.shape[-1] < 1 << 22
    ):
        return None
    from .batching import _gate_parameters

    parameters = _gate_parameters(circuit, instruction, output, parameter_bindings)
    if parameters is None:
        return None
    return _apply_ry_rotation(
        output, parameters, matrix.detach(), instruction.wires[0], circuit.n_qubits
    )
