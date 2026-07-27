"""Tape replay and gradient reduction for distributed MPS reverse mode."""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.distributed as dist
from torch.profiler import record_function

from .communication import (
    _apply_one_mps_tensor,
    _instruction_matrix_for_mps,
)
from .errors import MPSReverseContractError
from .factorization import apply_mps_pair_forward, apply_mps_qr_forward
from .records import MPSReverseTapeRecord, TorchDistributedMPSGradientResult
from .reverse_transport import receive_static_reverse_tensor, send_static_reverse_tensor
from .site_kernels import apply_rxx_contraction_bucket, apply_ry_bucket
from .state import RankOwnedMPSState


def build_mps_reverse_backward(
    *,
    state: RankOwnedMPSState,
    payloads: dict[int, Any],
    payload_factory: Callable[..., Any],
    parameters: Sequence[torch.Tensor],
    adjoints: dict[int, torch.Tensor],
    local_tensors: Mapping[int, torch.Tensor],
    reverse_segments: Sequence[Sequence[MPSReverseTapeRecord]],
    gradient_buckets: Sequence[Any],
    rank: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    compile_site_kernels: bool,
    reverse_delay_seconds: float,
    gradient_owner_ranks: Sequence[int] | None,
) -> Callable[[TorchDistributedMPSGradientResult], None]:
    """Create the single-use reverse replay closure for a recorded MPS tape."""

    bsz = batch_size
    resolved_device = device
    resolved_dtype = dtype
    _pair_forward = apply_mps_pair_forward
    _qr_forward = apply_mps_qr_forward
    _recv_static = receive_static_reverse_tensor
    _send_static = send_static_reverse_tensor

    def backward(result: TorchDistributedMPSGradientResult) -> None:
        def match_adjoint(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            """Project a stale bond adjoint onto the current output shape.

            Truncated forward splits can change a bond dimension between two
            recorded operations.  The reverse tape is intentionally
            approximate in that mode, so a downstream adjoint may carry the
            pre-truncation bond extent.  Autograd requires an exact shape;
            retain the overlapping Schmidt coordinates and zero-pad the rest.
            This is the same projection used by the truncation VJP and keeps
            approximate-gradient training alive instead of failing with a
            shape mismatch.
            """
            if tuple(value.shape) == tuple(target.shape):
                return value
            if value.ndim != target.ndim or value.shape[0] != target.shape[0]:
                raise MPSReverseContractError(
                    "MPS reverse adjoint rank/batch mismatch: "
                    f"adjoint={tuple(value.shape)} output={tuple(target.shape)}"
                )
            projected = torch.zeros_like(target)
            slices = tuple(
                slice(0, min(int(source), int(destination)))
                for source, destination in zip(value.shape, target.shape)
            )
            projected[slices] = value[slices]
            return projected

        def finite(
            value: torch.Tensor | None,
            *,
            operation: str,
            like: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if value is None:
                if like is None:
                    raise MPSReverseContractError(
                        f"missing MPS reverse VJP at {operation}"
                    )
                return torch.zeros_like(like)
            if not torch.isfinite(value).all():
                raise MPSReverseContractError(
                    f"non-finite MPS reverse VJP at {operation}"
                )
            return value.detach()

        local_gradients: list[torch.Tensor | None] = [None] * len(parameters)

        def accumulate(
            parameter_index: int, derivative: torch.Tensor, operation: str
        ) -> None:
            value = finite(derivative, operation=operation)
            previous = local_gradients[parameter_index]
            local_gradients[parameter_index] = (
                value if previous is None else previous + value
            )

        if reverse_delay_seconds > 0:
            time.sleep(float(reverse_delay_seconds) * rank)
        for segment in reverse_segments:
            if len(segment) > 1:
                owner = segment[0].compute_owner
                if rank == owner:
                    segment_inputs = tuple(
                        payloads[record.forward_sequence]
                        .inputs[0]
                        .detach()
                        .requires_grad_(True)
                        for record in segment
                    )
                    active_indices = tuple(
                        dict.fromkeys(
                            index
                            for record in segment
                            for index in record.parameter_indices
                        )
                    )
                    active = tuple(parameters[index] for index in active_indices)
                    matrices = []
                    for record in segment:
                        payload = payloads[record.forward_sequence]
                        matrix = _instruction_matrix_for_mps(
                            payload.instruction,
                            bsz=bsz,
                            device=resolved_device,
                            dtype=resolved_dtype,
                        )
                        matrices.append(
                            matrix.expand(bsz, -1, -1) if matrix.ndim == 2 else matrix
                        )
                    stacked_inputs = torch.stack(segment_inputs)
                    stacked_matrices = torch.stack(matrices)
                    all_ry = all(
                        payloads[record.forward_sequence].instruction.name == "ry"
                        for record in segment
                    )
                    with record_function(
                        "flagquantum::mps::fused_local_reverse_segment"
                    ):
                        outputs = (
                            apply_ry_bucket(
                                stacked_inputs,
                                stacked_matrices,
                                compiled=compile_site_kernels,
                            )
                            if all_ry
                            else torch.stack(
                                [
                                    _apply_one_mps_tensor(value, matrix)
                                    for value, matrix in zip(segment_inputs, matrices)
                                ]
                            )
                        )
                        try:
                            derivatives = torch.autograd.grad(
                                outputs,
                                segment_inputs + active,
                                grad_outputs=torch.stack(
                                    [
                                        match_adjoint(
                                            adjoints[record.wires[0]],
                                            outputs[index],
                                        )
                                        for index, record in enumerate(segment)
                                    ]
                                ),
                                allow_unused=True,
                                retain_graph=True,
                            )
                        except Exception as error:
                            operation_ids = ",".join(
                                record.operation_id for record in segment
                            )
                            raise MPSReverseContractError(
                                f"fused local MPS VJP failed at segment [{operation_ids}]: {error}"
                            ) from error
                    for record, derivative in zip(segment, derivatives[: len(segment)]):
                        adjoints[record.wires[0]] = finite(
                            derivative,
                            operation=record.operation_id,
                            like=segment_inputs[segment.index(record)],
                        )
                    for parameter_index, derivative in zip(
                        active_indices, derivatives[len(segment) :]
                    ):
                        if derivative is not None:
                            accumulate(
                                parameter_index, derivative, segment[0].operation_id
                            )
                for record in segment:
                    result._last_completed_record = record.reverse_sequence
                continue
            record = segment[0]
            left_wire = min(record.wires)
            peer = record.communication_peer
            sequence = 3_000_000 + record.reverse_sequence * 2
            if peer is not None and rank == peer:
                _send_static(
                    adjoints[left_wire + 1],
                    destination=record.compute_owner,
                    sequence=sequence,
                    shape=record.output_shapes[1],
                )
            if rank == record.compute_owner:
                output_adjoints = [adjoints[left_wire]]
                if len(record.wires) == 2:
                    output_adjoints.append(
                        adjoints[left_wire + 1]
                        if peer is None
                        else _recv_static(
                            next(iter(local_tensors.values())),
                            source=peer,
                            sequence=sequence,
                            shape=record.output_shapes[1],
                        )
                    )
                payload = payloads[record.forward_sequence]
                inputs = tuple(
                    value.detach().requires_grad_(True) for value in payload.inputs
                )
                active = tuple(parameters[index] for index in record.parameter_indices)
                if payload.kind == "one_site":
                    matrix = _instruction_matrix_for_mps(
                        payload.instruction,
                        bsz=bsz,
                        device=resolved_device,
                        dtype=resolved_dtype,
                    )
                    outputs = (_apply_one_mps_tensor(inputs[0], matrix),)
                elif (
                    payload.kind == "two_site"
                    and payload.factorization_pair is not None
                    and payload.factorization_outputs is not None
                ):
                    saved_differentiable = tuple(
                        (output, output_adjoint)
                        for output, output_adjoint in zip(
                            payload.factorization_outputs, output_adjoints
                        )
                        if output.requires_grad
                    )
                    if not saved_differentiable:
                        raise MPSReverseContractError(
                            f"saved factorization has no differentiable output at "
                            f"{record.operation_id}"
                        )
                    pair_adjoint = torch.autograd.grad(
                        tuple(value[0] for value in saved_differentiable),
                        (payload.factorization_pair,),
                        grad_outputs=tuple(value[1] for value in saved_differentiable),
                        allow_unused=False,
                    )[0]
                    matrix = _instruction_matrix_for_mps(
                        payload.instruction,
                        bsz=bsz,
                        device=resolved_device,
                        dtype=resolved_dtype,
                    )
                    if matrix.ndim == 2:
                        matrix = matrix.expand(bsz, -1, -1)
                    pair = apply_rxx_contraction_bucket(
                        inputs[0].unsqueeze(0),
                        inputs[1].unsqueeze(0),
                        matrix.unsqueeze(0),
                        compiled=compile_site_kernels,
                    )[0]
                    outputs = (pair,)
                    output_adjoints = [pair_adjoint]
                elif payload.kind == "two_site":
                    left, right, _ = _pair_forward(
                        inputs[0], inputs[1], payload.instruction, state
                    )
                    outputs = (left, right)
                else:
                    outputs = _qr_forward(inputs[0], inputs[1])
                differentiable = tuple(
                    (output, match_adjoint(output_adjoint, output))
                    for output, output_adjoint in zip(outputs, output_adjoints)
                    if output.requires_grad
                )
                if not differentiable:
                    raise MPSReverseContractError(
                        f"local VJP has no differentiable output at {record.operation_id}"
                    )
                derivatives = torch.autograd.grad(
                    tuple(value[0] for value in differentiable),
                    inputs + active,
                    grad_outputs=tuple(value[1] for value in differentiable),
                    allow_unused=True,
                    retain_graph=True,
                )
                if payload.factorization_pair is not None:
                    # The result is single-use. Drop the saved split graph as
                    # soon as its pair adjoint has been consumed so the next
                    # training step cannot overlap two generations of SVD
                    # checkpoints.
                    payloads[record.forward_sequence] = payload_factory(
                        payload.kind, payload.inputs, payload.instruction
                    )
                input_derivatives = derivatives[: len(inputs)]
                adjoints[left_wire] = finite(
                    input_derivatives[0],
                    operation=record.operation_id,
                    like=inputs[0],
                )
                if len(inputs) == 2:
                    right_adjoint = finite(
                        input_derivatives[1],
                        operation=record.operation_id,
                        like=inputs[1],
                    )
                    if peer is None:
                        adjoints[left_wire + 1] = right_adjoint
                    else:
                        _send_static(
                            right_adjoint,
                            destination=peer,
                            sequence=sequence + 1,
                            shape=record.input_shapes[1],
                        )
                for parameter_index, derivative in zip(
                    record.parameter_indices, derivatives[len(inputs) :]
                ):
                    if derivative is not None:
                        accumulate(parameter_index, derivative, record.operation_id)
            if peer is not None and rank == peer:
                adjoints[left_wire + 1] = _recv_static(
                    adjoints[left_wire + 1],
                    source=record.compute_owner,
                    sequence=sequence + 1,
                    shape=record.input_shapes[1],
                )
            result._last_completed_record = record.reverse_sequence
        for dtype, owner, pieces in gradient_buckets:
            flat = torch.cat(
                [
                    (
                        torch.zeros(end - start, dtype=dtype, device=resolved_device)
                        if local_gradients[index] is None
                        else local_gradients[index].reshape(-1)[start:end]
                    )
                    for index, start, end in pieces
                ]
            )
            with record_function("flagquantum::mps::gradient_collective"):
                if owner is None:
                    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
                else:
                    dist.reduce(flat, dst=owner, op=dist.ReduceOp.SUM)
            if owner is None or rank == owner:
                offset = 0
                for index, start, end in pieces:
                    count = end - start
                    parameter = parameters[index]
                    if parameter.grad is None:
                        parameter.grad = torch.zeros_like(parameter)
                    parameter.grad.reshape(-1)[start:end].add_(
                        flat[offset : offset + count]
                    )
                    offset += count
            elif gradient_owner_ranks is not None:
                for index, _, _ in pieces:
                    parameters[index].grad = None

    return backward


__all__ = ("build_mps_reverse_backward",)
