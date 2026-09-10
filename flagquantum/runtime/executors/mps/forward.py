"""PyTorch-native rank-owned MPS forward executor."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import torch
import torch.distributed as dist

from ....core.ir import ensure_circuit_ir
from ....simulation.mps.models import MPSConfig
from ....simulation.mps.rank_local import (
    apply_rank_local_mps_instruction as _apply_rank_local_instruction,
)
from ....simulation.mps.rank_local import (
    instruction_matrix_for_mps as _instruction_matrix_for_mps,
)
from ....simulation.mps.rank_local import (
    tensor_nbytes as _tensor_nbytes,
)
from ....simulation.mps.state import MPSState
from .canonicalization import canonicalize_rank_owned_mps
from .compiled_layers import (
    _PreparedMPSOutput,
    device_memory_metadata,
    prepare_compiled_mps_layer,
    require_layer_cache_drained,
)
from .distribution import (
    apply_rank_boundary_gate,
    global_mps_bond_dimensions,
    rebalance_mps_if_needed,
)
from .errors import (
    MPSForwardLifetimeError,
    MPSFullMaterializationError,
    NonlocalMPSCompilationError,
)
from .factorization import (
    FactorizationWorkspacePolicy,
    MemoryProvider,
    cuda_factorization_memory_snapshot,
)
from .metadata_transport import all_gather_json
from .records import TorchDistributedMPSForwardResult
from .state import (
    MPSPartition,
    RankOwnedMPSState,
    _owner,
    initial_mps_ownership,
)
from .transport import (
    _recv_tensor_p2p,
    _send_tensor_p2p,
)

_apply_boundary_gate = apply_rank_boundary_gate

_require_layer_cache_drained = require_layer_cache_drained
_cuda_memory_fields = device_memory_metadata

_prepare_compiled_layer = prepare_compiled_mps_layer


def execute_torch_distributed_mps_forward(
    circuit_or_ir: Any,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    max_bond: int | None = None,
    cutoff: float = 0.0,
    rebalance_threshold: float = 1.5,
    canonical_center: int | None = None,
    global_error_budget: float | None = None,
    error_budget_policy: str = "enforce",
    truncation_gradient_policy: str = "unsupported",
    compile_site_kernels: bool = False,
    layer_lifecycle_callback: Callable[[dict[str, Any]], None] | None = None,
    factorization_workspace_policy: FactorizationWorkspacePolicy | None = None,
    factorization_memory_provider: MemoryProvider = cuda_factorization_memory_snapshot,
) -> TorchDistributedMPSForwardResult:
    """Execute adjacent-gate IR while retaining only rank-owned MPS sites."""

    if not dist.is_initialized():
        raise RuntimeError("rank-owned MPS execution requires torch.distributed")
    ir = ensure_circuit_ir(circuit_or_ir)
    if compile_site_kernels and any(
        instruction.name in {"rxx", "ryy", "rzz"}
        and tuple(map(int, instruction.wires))
        != tuple(sorted(map(int, instruction.wires)))
        for instruction in ir.instructions
    ):
        raise NonlocalMPSCompilationError(
            "compiled site-sharded two-site rotations require ascending adjacent wire order"
        )
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    backend = str(dist.get_backend())
    resolved_device = torch.device(device or "cpu")
    dtype = dtype or getattr(torch, ir.dtype)
    bsz = int(ir.metadata.get("batch_size", 1))
    for index, instruction in enumerate(ir.instructions):
        wires = tuple(int(wire) for wire in instruction.wires)
        if (
            instruction.metadata.get("is_channel")
            or len(wires) not in {1, 2}
            or (len(wires) == 2 and abs(wires[0] - wires[1]) != 1)
        ):
            raise NonlocalMPSCompilationError(
                f"instruction {index}:{instruction.name} on wires {wires} is "
                "not one-site or adjacent two-site; route it before MPS execution"
            )
    ownership = initial_mps_ownership(ir.n_wires, world_size)
    local_tensors = {}
    for wire in ownership[rank]:
        tensor = torch.zeros((bsz, 1, 2, 1), dtype=dtype, device=resolved_device)
        tensor[:, :, 0, :] = 1
        local_tensors[wire] = tensor
    state = RankOwnedMPSState(
        n_wires=ir.n_wires,
        bsz=bsz,
        rank=rank,
        world_size=world_size,
        config=MPSConfig(max_bond=max_bond, cutoff=cutoff),
        local_tensors=local_tensors,
        ownership=ownership,
    )
    if global_error_budget is not None and global_error_budget < 0:
        raise ValueError("global_error_budget must be non-negative")
    if error_budget_policy not in {"enforce", "report_only"}:
        raise ValueError("error_budget_policy must be enforce or report_only")
    if truncation_gradient_policy not in {"exact", "approximate", "unsupported"}:
        raise ValueError(
            "truncation_gradient_policy must be exact, approximate, or unsupported"
        )
    if compile_site_kernels and rebalance_threshold != float("inf"):
        raise ValueError("compiled site buckets require rebalance_threshold=inf")
    factorization_workspace_policy = (
        factorization_workspace_policy or FactorizationWorkspacePolicy()
    )
    local_count = boundary_count = messages = byte_count = rebalances = 0
    rebalance_messages = rebalance_bytes = 0
    local_truncation_records: list[dict[str, Any]] = []
    history = [ownership]
    precomputed: dict[int, _PreparedMPSOutput] = {}
    prepared: set[int] = set()
    layer_ends: dict[int, tuple[int, int, str, int]] = {}
    layer_lifecycle_records: list[dict[str, Any]] = []
    factorization_records: list[dict[str, Any]] = []
    layer_sequence = 0
    outputs: tuple[torch.Tensor, ...]
    split_info: Mapping[str, Any] | None
    for instruction_index, instruction in enumerate(ir.instructions):
        if (
            compile_site_kernels
            and instruction.name in {"ry", "rxx", "ryy", "rzz"}
            and instruction_index not in prepared
        ):
            layer, prepared_outputs, layer_factorization_records = (
                _prepare_compiled_layer(
                    ir.instructions,
                    layer_start=instruction_index,
                    state=state,
                    bsz=bsz,
                    device=resolved_device,
                    dtype=dtype,
                    factorization_workspace_policy=factorization_workspace_policy,
                    factorization_memory_provider=factorization_memory_provider,
                )
            )
            factorization_records.extend(layer_factorization_records)
            if precomputed:
                raise MPSForwardLifetimeError(
                    "new compiled MPS layer began before the previous cache drained"
                )
            precomputed.update(prepared_outputs)
            prepared.update(index for index, _, _ in layer)
            layer_end = layer[-1][0]
            layer_ends[layer_end] = (
                layer_sequence,
                instruction_index,
                instruction.name,
                len(prepared_outputs),
            )
            layer_sequence += 1
        wires = tuple(int(wire) for wire in instruction.wires)
        owners = {_owner(state.ownership, wire) for wire in wires}
        if len(wires) == 1:
            if rank in owners:
                if instruction_index in precomputed:
                    cached_output = precomputed.pop(instruction_index)
                    if len(cached_output) != 1:
                        raise MPSForwardLifetimeError(
                            "one-site compiled MPS instruction requires one cached tensor"
                        )
                    outputs = cached_output
                else:
                    outputs, _ = _apply_rank_local_instruction(
                        instruction,
                        (state.local_tensors[wires[0]],),
                        state.config,
                        bsz=bsz,
                        device=resolved_device,
                        dtype=dtype,
                    )
                state.local_tensors[wires[0]] = outputs[0]
                local_count += 1
        elif len(owners) == 1:
            owner = next(iter(owners))
            if rank == owner:
                left = min(wires)
                if instruction_index in precomputed:
                    cached_output = precomputed.pop(instruction_index)
                    if len(cached_output) != 3:
                        raise MPSForwardLifetimeError(
                            "two-site compiled MPS instruction requires two cached "
                            "tensors and split metadata"
                        )
                    updated_left, updated_right, split_info = cached_output
                else:
                    outputs, split_info = _apply_rank_local_instruction(
                        instruction,
                        (
                            state.local_tensors[left],
                            state.local_tensors[left + 1],
                        ),
                        state.config,
                        bsz=bsz,
                        device=resolved_device,
                        dtype=dtype,
                    )
                    updated_left, updated_right = outputs
                    assert split_info is not None
                state.local_tensors[left] = updated_left
                state.local_tensors[left + 1] = updated_right
                local_truncation_records.append(
                    {**dict(split_info), "bond": left, "source": "rank_local_gate"}
                )
                local_count += 1
        else:
            matrix = _instruction_matrix_for_mps(
                instruction, bsz=bsz, device=resolved_device, dtype=dtype
            )
            event_messages, event_bytes, record = _apply_boundary_gate(
                state, instruction, matrix
            )
            del matrix
            boundary_count += 1
            messages += event_messages
            byte_count += event_bytes
            if record is not None:
                local_truncation_records.append(record)
        if len(wires) == 2:
            rebalanced, event_messages, event_bytes = rebalance_mps_if_needed(
                state, rebalance_threshold
            )
            rebalance_messages += event_messages
            rebalance_bytes += event_bytes
            if rebalanced:
                rebalances += 1
                history.append(state.ownership)
        if instruction_index in layer_ends:
            sequence, layer_start, kind, prepared_entries = layer_ends.pop(
                instruction_index
            )
            _require_layer_cache_drained(
                precomputed,
                layer_sequence=sequence,
                layer_start=layer_start,
                layer_end=instruction_index,
            )
            lifecycle_record: dict[str, Any] = {
                "layer_sequence": sequence,
                "kind": kind,
                "instruction_start": layer_start,
                "instruction_end": instruction_index,
                "instruction_count": instruction_index - layer_start + 1,
                "prepared_entry_count": prepared_entries,
                "remaining_precomputed_entries": 0,
                "local_tensor_bytes": sum(
                    _tensor_nbytes(tensor) for tensor in state.local_tensors.values()
                ),
                "last_operation": f"{instruction_index}:{instruction.name}",
                **_cuda_memory_fields(resolved_device),
            }
            layer_lifecycle_records.append(lifecycle_record)
            if layer_lifecycle_callback is not None:
                layer_lifecycle_callback(dict(lifecycle_record))
    _require_layer_cache_drained(
        precomputed,
        layer_sequence=layer_sequence,
        layer_start=len(ir.instructions),
        layer_end=len(ir.instructions),
    )
    canonicalization = canonicalize_rank_owned_mps(state, center=canonical_center)
    bond_dimensions = global_mps_bond_dimensions(state)
    gathered_records = all_gather_json(local_truncation_records)
    truncation_records = tuple(
        record for rank_records in gathered_records for record in rank_records
    )
    truncation_error = sum(
        float(record["discarded_weight"]) for record in truncation_records
    )
    if (
        global_error_budget is not None
        and truncation_error > global_error_budget
        and error_budget_policy == "enforce"
    ):
        raise RuntimeError(
            "distributed MPS truncation error exceeded global_error_budget: "
            f"{truncation_error} > {global_error_budget}"
        )
    if truncation_gradient_policy == "exact" and any(
        float(record["discarded_weight"]) > 0 for record in truncation_records
    ):
        raise RuntimeError(
            "exact truncation-gradient policy cannot describe a truncated forward"
        )
    local_bytes = torch.tensor(
        [sum(_tensor_nbytes(tensor) for tensor in state.local_tensors.values())],
        dtype=torch.int64,
        device=resolved_device,
    )
    gathered_bytes = [torch.zeros_like(local_bytes) for _ in range(world_size)]
    dist.all_gather(gathered_bytes, local_bytes)
    return TorchDistributedMPSForwardResult(
        shard_state=state,
        local_gate_count=local_count,
        boundary_gate_count=boundary_count,
        boundary_messages=messages,
        boundary_bytes=byte_count,
        rebalance_messages=rebalance_messages,
        rebalance_bytes=rebalance_bytes,
        rebalance_count=rebalances,
        partition_history=tuple(history),
        bond_dimensions=bond_dimensions,
        rank_tensor_bytes=tuple(int(value.item()) for value in gathered_bytes),
        canonicalization=canonicalization,
        truncation_records=truncation_records,
        global_error_budget=global_error_budget,
        error_budget_policy=error_budget_policy,
        truncation_gradient_policy=truncation_gradient_policy,
        backend=backend,
        layer_lifecycle_records=tuple(layer_lifecycle_records),
        factorization_records=tuple(factorization_records),
        layer_cache_empty_at_return=True,
        gate_matrix_materialization="per_instruction_last_use",
    )


def gather_mps_for_validation(
    result: TorchDistributedMPSForwardResult, *, destination_rank: int = 0
) -> MPSState | None:
    """Explicit test-only rank gather; never called by the production executor."""

    state = result.shard_state
    tensors: dict[int, torch.Tensor] = {}
    reference = next(iter(state.local_tensors.values()))
    for wire in range(state.n_wires):
        owner = state.owner(wire)
        if owner == destination_rank:
            if state.rank == destination_rank:
                tensors[wire] = state.local_tensors[wire]
        elif state.rank == owner:
            _send_tensor_p2p(state.local_tensors[wire], dst=destination_rank)
        elif state.rank == destination_rank:
            tensors[wire] = _recv_tensor_p2p(src=owner, reference=reference)
    if state.rank != destination_rank:
        return None
    return MPSState(
        [tensors[wire] for wire in range(state.n_wires)],
        config=state.config,
    )


__all__ = [
    "MPSFullMaterializationError",
    "MPSForwardLifetimeError",
    "MPSPartition",
    "NonlocalMPSCompilationError",
    "RankOwnedMPSState",
    "TorchDistributedMPSForwardResult",
    "execute_torch_distributed_mps_forward",
    "gather_mps_for_validation",
]
