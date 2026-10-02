"""Shared policy and evidence used by statevector reverse executors."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import torch

from ....core.ir import CircuitIR, _normalize_angle
from ...builder_compilation import CompiledInstruction
from .environment import get_bool
from .kernel_dispatch import KernelDispatchEvidence
from .local_execution import use_compact_global_indices

_DEFAULT_REVERSE_CHUNK_AMPLITUDES = 1 << 22


def _reverse_chunk_amplitudes() -> int:
    raw = os.getenv("FQ_STATEVECTOR_REVERSE_CHUNK_AMPLITUDES")
    value = _DEFAULT_REVERSE_CHUNK_AMPLITUDES if raw is None else int(raw)
    if value <= 0 or value & (value - 1):
        raise ValueError(
            "FQ_STATEVECTOR_REVERSE_CHUNK_AMPLITUDES must be a positive power of two"
        )
    return value


def _compact_reverse_global_indices(plan: Any, rank: int) -> bool:
    return bool(use_compact_global_indices(plan, rank))


def _reverse_exchange_workspace_enabled() -> bool:
    return os.getenv(
        "FQ_STATEVECTOR_REVERSE_EXCHANGE_WORKSPACE", "1"
    ).strip().lower() not in {"0", "false", "off", "no"}


def _persistent_wire_layout_enabled() -> bool:
    return get_bool("FQ_STATEVECTOR_PERSISTENT_WIRE_LAYOUT", True)


def _persistent_inplace_local_enabled() -> bool:
    return os.getenv(
        "FQ_STATEVECTOR_PERSISTENT_INPLACE_LOCAL", "1"
    ).strip().lower() not in {"0", "false", "off", "no"}


def _fused_vjp_pipeline_enabled() -> bool:
    return os.getenv("FQ_STATEVECTOR_FUSED_VJP_PIPELINE", "0").strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }


def _gradient_reduction_overlap_enabled() -> bool:
    return get_bool("FQ_STATEVECTOR_GRADIENT_REDUCTION_OVERLAP", True)


def _gradient_bucketing_enabled() -> bool:
    return get_bool("FQ_STATEVECTOR_GRADIENT_BUCKETING", True)


def _saved_parameter_revalidation_enabled() -> bool:
    """Whether backward repeats validation already completed during forward."""

    return get_bool("FQ_STATEVECTOR_ADJOINT_REVALIDATE_SAVED_PARAMETERS", False)


def _reverse_cross_shard_cx_packing_enabled() -> bool:
    return os.getenv(
        "FQ_STATEVECTOR_REVERSE_CROSS_SHARD_CX_PACK", "0"
    ).strip().lower() in {"1", "true", "on", "yes"}


def _bind_parameters(
    ir: CircuitIR,
    slots: Sequence[tuple[int, str, int]],
    parameters: Sequence[torch.Tensor],
    *,
    validate: bool = True,
) -> CircuitIR:
    instructions = list(ir.instructions)
    params_by_instruction: dict[int, dict[str, Any]] = {}
    validated_parameters: set[int] = set()
    for instruction_index, name, parameter_index in slots:
        instruction = instructions[instruction_index]
        if validate and parameter_index not in validated_parameters:
            _normalize_angle(
                parameters[parameter_index],
                opcode=instruction.name,
                parameter=name,
            )
            validated_parameters.add(parameter_index)
        params = params_by_instruction.setdefault(
            instruction_index, dict(instruction.params)
        )
        params[name] = parameters[parameter_index]
    for index, params in params_by_instruction.items():
        instruction = instructions[index]
        instructions[index] = CompiledInstruction(
            name=instruction.name,
            wires=instruction.wires,
            params=params,
            matrix=instruction.matrix,
            metadata=instruction.metadata,
            parameter_slots=(),
            parameter_constants=(),
        )
    return replace(ir, instructions=tuple(instructions))


def _bind_saved_parameters(
    ir: CircuitIR,
    slots: Sequence[tuple[int, str, int]],
    parameters: Sequence[torch.Tensor],
) -> CircuitIR:
    """Bind autograd-saved tensors with the configured validation rollback."""

    return _bind_parameters(
        ir,
        slots,
        parameters,
        validate=_saved_parameter_revalidation_enabled(),
    )


@dataclass
class ParameterGradientOwnership:
    parameter: str
    owner_rank: int | None
    occurrence_count: int
    reduction: str
    participating_ranks: tuple[int, ...]
    distribution: str


@dataclass
class BackwardExecutionEvidence:
    """Evidence populated only by an actually executed autograd backward."""

    status: str = "pending"
    participating_ranks: tuple[int, ...] = ()
    reduction_counts: list[int] = field(default_factory=list)
    error: str | None = None
    ownership: tuple[ParameterGradientOwnership, ...] = ()
    communication_count: int = 0
    communication_bytes: int = 0
    gradient_collective_count: int = 0
    gradient_collective_bytes: int = 0
    async_gradient_collective_count: int = 0
    overlapped_gradient_collective_count: int = 0
    persistent_layout_enabled: bool = False
    persistent_layout_swap_count: int = 0
    persistent_layout_swap_bytes: int = 0
    persistent_layout_buffer_allocation_count: int = 0
    persistent_layout_buffer_reserved_bytes: int = 0
    persistent_inplace_local_enabled: bool = False
    analytic_rotation_derivative_count: int = 0
    fused_parameter_adjoint_count: int = 0
    peak_scratch_bytes: int = 0
    saved_forward_state_reused: bool = False
    checkpoint_count: int = 0
    rematerialized_gate_count: int = 0
    exchange_chunk_amplitudes: int = _DEFAULT_REVERSE_CHUNK_AMPLITUDES
    exchange_chunk_bytes: int = 0
    exchange_workspace_allocation_count: int = 0
    exchange_workspace_reuse_count: int = 0
    exchange_workspace_reserved_bytes: int = 0
    exchange_pipeline_prefetch_count: int = 0
    intra_node_communication_count: int = 0
    intra_node_communication_bytes: int = 0
    inter_node_communication_count: int = 0
    inter_node_communication_bytes: int = 0
    kernel_dispatch_evidence: KernelDispatchEvidence = field(
        default_factory=KernelDispatchEvidence
    )


__all__ = ("BackwardExecutionEvidence", "ParameterGradientOwnership")
