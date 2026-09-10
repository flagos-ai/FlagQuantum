"""Tape replay and gradient reduction for distributed MPS reverse mode."""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.distributed as dist
from torch.profiler import record_function

from ....simulation.mps.compiled_layers import (
    apply_mps_one_site_bucket,
    contract_mps_two_site_bucket,
)
from ....simulation.mps.rank_local import apply_rank_local_mps_instruction
from ....simulation.mps.reverse import mps_vjp
from .errors import MPSReverseContractError
from .factorization import mps_qr_forward
from .records import MPSReverseTapeRecord, TorchDistributedMPSGradientResult
from .reverse_transport import receive_static_reverse_tensor, send_static_reverse_tensor
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
    _recv_static = receive_static_reverse_tensor
    _send_static = send_static_reverse_tensor

    def backward(result: TorchDistributedMPSGradientResult) -> None:
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
                    segment_instructions = tuple(
                        payloads[record.forward_sequence].instruction
                        for record in segment
                    )
                    with record_function(
                        "flagquantum::mps::fused_local_reverse_segment"
                    ):
                        outputs = apply_mps_one_site_bucket(
                            segment_instructions,
                            segment_inputs,
                            bsz=bsz,
                            device=resolved_device,
                            dtype=resolved_dtype,
                            compile_ry=compile_site_kernels,
                        )
                        try:
                            derivatives = mps_vjp(
                                outputs,
                                segment_inputs,
                                active,
                                tuple(adjoints[record.wires[0]] for record in segment),
                            )
                        except (RuntimeError, ValueError) as error:
                            operation_ids = ",".join(
                                record.operation_id for record in segment
                            )
                            raise MPSReverseContractError(
                                f"fused local MPS VJP failed at segment [{operation_ids}]: {error}"
                            ) from error
                    for input_index, (record, derivative) in enumerate(
                        zip(segment, derivatives[: len(segment)])
                    ):
                        adjoints[record.wires[0]] = finite(
                            derivative,
                            operation=record.operation_id,
                            like=segment_inputs[input_index],
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
                    outputs = apply_mps_one_site_bucket(
                        (payload.instruction,),
                        (inputs[0],),
                        bsz=bsz,
                        device=resolved_device,
                        dtype=resolved_dtype,
                        compile_ry=compile_site_kernels,
                    )
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
                    pair = contract_mps_two_site_bucket(
                        (payload.instruction,),
                        (inputs[0],),
                        (inputs[1],),
                        bsz=bsz,
                        device=resolved_device,
                        dtype=resolved_dtype,
                        compiled=compile_site_kernels,
                    )[0]
                    outputs = (pair,)
                    output_adjoints = [pair_adjoint]
                elif payload.kind == "two_site":
                    output_tensors, _ = apply_rank_local_mps_instruction(
                        payload.instruction,
                        inputs,
                        state.config,
                        bsz=bsz,
                        device=resolved_device,
                        dtype=resolved_dtype,
                    )
                    outputs = output_tensors
                else:
                    outputs = mps_qr_forward(inputs[0], inputs[1])
                try:
                    derivatives = mps_vjp(
                        outputs,
                        inputs,
                        active,
                        output_adjoints,
                    )
                except (RuntimeError, ValueError) as error:
                    raise MPSReverseContractError(
                        f"local MPS VJP failed at {record.operation_id}: {error}"
                    ) from error
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
