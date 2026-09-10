"""Deterministic explicit reverse mode for rank-owned distributed MPS."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ....core.ir import Instruction, ensure_circuit_ir
from ....simulation.mps.compiled_layers import (
    apply_compiled_mps_one_site_bucket,
    contract_mps_two_site_bucket,
)
from ....simulation.mps.observables import transfer_mps_operator_environment
from ....simulation.mps.rank_local import (
    apply_one_mps_tensor,
    apply_rank_local_mps_instruction,
    instruction_matrix_for_mps,
    tensor_nbytes,
)
from ....simulation.mps.reverse import (
    factor_mps_reverse_pair,
    factor_mps_reverse_pair_bucket,
)
from ....simulation.mps.site_kernels import site_kernel_bucket_capacity
from .compiled_layers import collect_compiled_mps_layer
from .errors import NonlocalMPSCompilationError
from .factorization import mps_qr_forward
from .records import (
    MPSReverseCheckpointPolicy,
    MPSReverseContractError,
    MPSReverseTape,
    MPSReverseTapeRecord,
    ReverseCheckpointBudget,
    TorchDistributedMPSGradientResult,
    build_mps_reverse_tape_record,
)
from .reverse_observables import (
    SiteShardedZZScanResult,
    mps_expectation_and_adjoints,
    mps_heisenberg_energy_and_adjoints,
    parse_mps_heisenberg_terms,
    parse_mps_z_zz_terms,
)
from .reverse_planning import (
    build_mps_gradient_ownership,
    build_mps_parameter_layout,
    cached_mps_gradient_buckets,
    cached_mps_reverse_segments,
    plan_mps_canonicalization_bonds,
    validate_mps_svd_gaps,
)
from .reverse_replay import build_mps_reverse_backward
from .reverse_transport import (
    ReverseLayerHaloPrefetch,
    _ReverseRecordMetadata,
    all_reduce_reverse_layer_records,
    begin_reverse_layer_halo_prefetch,
    broadcast_reverse_record,
    finish_reverse_layer_halo_prefetch,
    receive_reverse_tensor,
    receive_static_reverse_tensor,
    send_reverse_tensor,
    send_static_reverse_tensor,
    static_shape_generation,
)
from .reverse_z_observables import (
    mps_fused_z_zz_mse_and_adjoints,
    mps_multi_observable_mse_and_adjoints,
    mps_site_sharded_z_zz_scan,
    site_sharded_z_zz_objective_pipeline,
    site_sharded_z_zz_observations,
)
from .state import (
    RankOwnedMPSState,
    initialize_reverse_mps_state,
    normalize_rank_owned_initial_tensors,
)

_rank_owned_initial_tensors = normalize_rank_owned_initial_tensors
_broadcast_record = broadcast_reverse_record
_all_reduce_layer_records = all_reduce_reverse_layer_records
_begin_layer_halo_prefetch = begin_reverse_layer_halo_prefetch
_finish_layer_halo_prefetch = finish_reverse_layer_halo_prefetch
_fused_z_zz_mse_and_adjoints = mps_fused_z_zz_mse_and_adjoints
_heisenberg_mpo_energy_and_adjoints = mps_heisenberg_energy_and_adjoints
_multi_observable_mse_and_adjoints = mps_multi_observable_mse_and_adjoints
_expectation_and_adjoints = mps_expectation_and_adjoints
_parse_heisenberg_hamiltonian_terms = parse_mps_heisenberg_terms
_parse_z_zz_terms = parse_mps_z_zz_terms
_planned_canonicalization_bonds = plan_mps_canonicalization_bonds
_qr_forward = mps_qr_forward
_recv = receive_reverse_tensor
_recv_static = receive_static_reverse_tensor
_record = build_mps_reverse_tape_record
_send = send_reverse_tensor
_send_static = send_static_reverse_tensor
_shape_generation = static_shape_generation
_site_sharded_z_zz_scan = mps_site_sharded_z_zz_scan
_transfer = transfer_mps_operator_environment
_validate_svd_gaps = validate_mps_svd_gaps

_LayerHaloPrefetch = ReverseLayerHaloPrefetch
_PreparedTwoSite = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Mapping[str, Any],
]
_SavedFactorization = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class _ReversePayload:
    kind: str
    inputs: tuple[torch.Tensor, ...]
    instruction: Instruction | None
    factorization_pair: torch.Tensor | None = None
    factorization_outputs: tuple[torch.Tensor, torch.Tensor] | None = None


def _prepare_reverse_one_site_layer(
    layer: Sequence[tuple[int, Instruction, tuple[int, ...]]],
    *,
    state: RankOwnedMPSState,
    checkpoint_budget: ReverseCheckpointBudget,
    bsz: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    for _, _, wires in layer:
        wire = wires[0]
        owner = state.owner(wire)
        checkpoint_budget.reserve(
            tensor_nbytes(state.local_tensors[wire]) if state.rank == owner else 0
        )
    buckets: dict[
        tuple[tuple[int, ...], tuple[int, ...]],
        list[tuple[int, Instruction, int]],
    ] = {}
    for index, instruction, wires in layer:
        wire = wires[0]
        if state.owner(wire) != state.rank:
            continue
        matrix = instruction_matrix_for_mps(
            instruction, bsz=bsz, device=device, dtype=dtype
        )
        if matrix.ndim == 2:
            matrix = matrix.expand(bsz, -1, -1)
        key = tuple(state.local_tensors[wire].shape), tuple(matrix.shape)
        buckets.setdefault(key, []).append((index, instruction, wire))
    prepared: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for bucket in buckets.values():
        _, sample, sample_wire = bucket[0]
        sample_matrix = instruction_matrix_for_mps(
            sample, bsz=bsz, device=device, dtype=dtype
        )
        if sample_matrix.ndim == 2:
            sample_matrix = sample_matrix.expand(bsz, -1, -1)
        capacity = site_kernel_bucket_capacity(
            state.local_tensors[sample_wire], sample_matrix
        )
        for start in range(0, len(bucket), capacity):
            chunk = bucket[start : start + capacity]
            inputs = tuple(
                state.local_tensors[wire].detach().clone() for _, _, wire in chunk
            )
            with torch.no_grad():
                outputs = apply_compiled_mps_one_site_bucket(
                    tuple(instruction for _, instruction, _ in chunk),
                    inputs,
                    bsz=bsz,
                    device=device,
                    dtype=dtype,
                )
            for position, (index, _, wire) in enumerate(chunk):
                output = outputs[position].detach()
                state.local_tensors[wire] = output
                prepared[index] = inputs[position], output
    return prepared


def _reserve_reverse_two_site_layer(
    layer: Sequence[tuple[int, Instruction, int]],
    *,
    state: RankOwnedMPSState,
    checkpoint_budget: ReverseCheckpointBudget,
    global_shapes: Mapping[int, Sequence[int]],
    bsz: int,
    dtype: torch.dtype,
    save_exact_factorizations: bool,
) -> set[int]:
    reservations = []
    for index, _, left_wire in layer:
        left_owner = state.owner(left_wire)
        required = 0
        optional = 0
        if state.rank == left_owner:
            element_size = torch.empty((), dtype=dtype).element_size()
            required = element_size * sum(
                math.prod(global_shapes[wire]) for wire in (left_wire, left_wire + 1)
            )
            left_shape = global_shapes[left_wire]
            right_shape = global_shapes[left_wire + 1]
            pair_bytes = (
                bsz * int(left_shape[1]) * 4 * int(right_shape[3]) * element_size
            )
            optional = 4 * pair_bytes
        reservations.append((index, required, optional))
    selected: set[int] = set()
    if state.config.svd_driver == "gesvda":
        checkpoint_budget.reserve(sum(item[1] for item in reservations))
        if save_exact_factorizations:
            optional_counts = [item[2] for item in reservations]
            if checkpoint_budget.reserve_factorization_batch(optional_counts):
                selected.update(item[0] for item in reservations)
            else:
                for index, _, optional in reservations:
                    if checkpoint_budget.reserve_factorization(optional):
                        selected.add(index)
    else:
        for index, required, optional in reservations:
            checkpoint_budget.reserve(required)
            if save_exact_factorizations and checkpoint_budget.reserve_factorization(
                optional
            ):
                selected.add(index)
    return selected


def _prepare_local_reverse_two_site_layer(
    layer: Sequence[tuple[int, Instruction, int]],
    *,
    state: RankOwnedMPSState,
    selected_factorizations: set[int],
    bsz: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[int, _PreparedTwoSite], dict[int, _SavedFactorization]]:
    buckets: dict[
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
        list[tuple[int, Instruction, int]],
    ] = {}
    for item in layer:
        _, instruction, left_wire = item
        if (
            state.owner(left_wire) != state.rank
            or state.owner(left_wire + 1) != state.rank
        ):
            continue
        matrix = instruction_matrix_for_mps(
            instruction, bsz=bsz, device=device, dtype=dtype
        )
        if matrix.ndim == 2:
            matrix = matrix.expand(bsz, -1, -1)
        shapes = (
            tuple(state.local_tensors[left_wire].shape),
            tuple(state.local_tensors[left_wire + 1].shape),
            tuple(matrix.shape),
        )
        buckets.setdefault(shapes, []).append(item)
    prepared: dict[int, _PreparedTwoSite] = {}
    saved: dict[int, _SavedFactorization] = {}
    for bucket in buckets.values():
        _, sample, sample_wire = bucket[0]
        sample_matrix = instruction_matrix_for_mps(
            sample, bsz=bsz, device=device, dtype=dtype
        )
        if sample_matrix.ndim == 2:
            sample_matrix = sample_matrix.expand(bsz, -1, -1)
        capacity = site_kernel_bucket_capacity(
            state.local_tensors[sample_wire],
            state.local_tensors[sample_wire + 1],
            sample_matrix,
        )
        for start in range(0, len(bucket), capacity):
            chunk = bucket[start : start + capacity]
            left_inputs = tuple(
                state.local_tensors[wire].detach().clone() for _, _, wire in chunk
            )
            right_inputs = tuple(
                state.local_tensors[wire + 1].detach().clone() for _, _, wire in chunk
            )
            with torch.no_grad():
                pair_matrices = contract_mps_two_site_bucket(
                    tuple(instruction for _, instruction, _ in chunk),
                    left_inputs,
                    right_inputs,
                    bsz=bsz,
                    device=device,
                    dtype=dtype,
                    compiled=True,
                )
            full_rank = min(int(pair_matrices.shape[-2]), int(pair_matrices.shape[-1]))
            batched_truncation = (
                any(index in selected_factorizations for index, _, _ in chunk)
                and state.config.max_bond is not None
                and int(state.config.max_bond) < full_rank
            )
            factorizations = factor_mps_reverse_pair_bucket(
                pair_matrices,
                left_dim=int(left_inputs[0].shape[1]),
                right_dim=int(right_inputs[0].shape[-1]),
                config=state.config,
                batched_truncated_split=batched_truncation,
            )
            for (index, _, wire), factorization, left, right in zip(
                chunk, factorizations, left_inputs, right_inputs
            ):
                pair_leaf, after_left, after_right, info = factorization
                detached_left = after_left.detach()
                detached_right = after_right.detach()
                state.local_tensors[wire] = detached_left
                state.local_tensors[wire + 1] = detached_right
                prepared[index] = left, right, detached_left, detached_right, info
                if index in selected_factorizations:
                    saved[index] = pair_leaf, after_left, after_right
    return prepared, saved


def _prepare_prefetched_reverse_two_site_layer(
    layer: Sequence[tuple[int, Instruction, int]],
    prefetch: ReverseLayerHaloPrefetch,
    *,
    state: RankOwnedMPSState,
    selected_factorizations: set[int],
    bsz: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[int, _PreparedTwoSite], dict[int, _SavedFactorization]]:
    prepared: dict[int, _PreparedTwoSite] = {}
    saved: dict[int, _SavedFactorization] = {}
    for index, instruction, left_wire in layer:
        left_owner = state.owner(left_wire)
        right_owner = state.owner(left_wire + 1)
        if (
            state.rank != left_owner
            or left_owner == right_owner
            or index not in prefetch.received
        ):
            continue
        before_left = state.local_tensors[left_wire].detach().clone()
        before_right = prefetch.received[index].detach().clone()
        if index in selected_factorizations:
            with torch.no_grad():
                contracted = contract_mps_two_site_bucket(
                    (instruction,),
                    (before_left,),
                    (before_right,),
                    bsz=bsz,
                    device=device,
                    dtype=dtype,
                    compiled=True,
                )[0]
            pair_leaf, after_left, after_right, info = factor_mps_reverse_pair(
                contracted,
                left_dim=int(before_left.shape[1]),
                right_dim=int(before_right.shape[-1]),
                config=state.config,
            )
            saved[index] = pair_leaf, after_left, after_right
        else:
            with torch.enable_grad():
                outputs, info = apply_rank_local_mps_instruction(
                    instruction,
                    (
                        before_left.detach().requires_grad_(True),
                        before_right.detach().requires_grad_(True),
                    ),
                    state.config,
                    bsz=bsz,
                    device=device,
                    dtype=dtype,
                )
                after_left, after_right = outputs
        if info is None:
            raise MPSReverseContractError(
                "two-site reverse preparation requires split metadata"
            )
        prepared[index] = (
            before_left,
            before_right,
            after_left.detach(),
            after_right.detach(),
            dict(info),
        )
    return prepared, saved


def _collect_reverse_layer_metadata(
    layer: Sequence[tuple[int, Instruction, int]],
    prepared: Mapping[int, _PreparedTwoSite],
    *,
    state: RankOwnedMPSState,
    template: torch.Tensor,
) -> tuple[dict[int, _ReverseRecordMetadata], int]:
    entries = tuple(
        (index, state.owner(left_wire))
        for index, _, left_wire in layer
        if state.owner(left_wire) != state.owner(left_wire + 1)
    )
    indices = {index for index, _ in entries}
    local_metadata: dict[int, _ReverseRecordMetadata] = {
        index: {
            "input_shapes": (values[0].shape, values[1].shape),
            "output_shapes": (values[2].shape, values[3].shape),
            "split_info": dict(values[4]),
        }
        for index, values in prepared.items()
        if index in indices
    }
    return _all_reduce_layer_records(local_metadata, entries, template), int(
        bool(entries)
    )


def _static_exact_qr_record(
    left_shape: Sequence[int],
    right_shape: Sequence[int],
    *,
    max_bond: int | None,
    cutoff: float,
) -> _ReverseRecordMetadata | None:
    """Derive split metadata when exact QR makes it rank-independent.

    This is deliberately unavailable for SVD/truncation paths: their retained
    rank and numerical diagnostics must continue to come from the compute
    owner.  Avoiding that dynamic broadcast also avoids a CUDA-to-CPU schema
    decode for every untruncated two-site gate.
    """
    left = tuple(int(value) for value in left_shape)
    right = tuple(int(value) for value in right_shape)
    if len(left) != 4 or len(right) != 4 or left[0] != right[0]:
        return None
    full_rank = min(left[1] * left[2], right[2] * right[3])
    if cutoff > 0 or (max_bond is not None and int(max_bond) < full_rank):
        return None
    output_left = (left[0], left[1], left[2], full_rank)
    output_right = (right[0], full_rank, right[2], right[3])
    return {
        "input_shapes": (left, right),
        "output_shapes": (output_left, output_right),
        "split_info": {
            "rank": full_rank,
            "original_rank": full_rank,
            "discarded_weight": 0.0,
        },
    }


def _validate_reverse_request(
    *,
    instructions: Sequence[Instruction],
    gradient_policy: str,
    degeneracy_tolerance: float,
    initial_bond_dimension: int,
    initial_mps_tensors_provided: bool,
    initial_mps_left_canonical: bool,
    canonicalization_policy: str,
    compile_site_kernels: bool,
    gradient_owner_ranks: Sequence[int] | None,
    world_size: int,
    observable: Mapping[int, str] | None,
    observable_terms: Sequence[tuple[Mapping[int, str], torch.Tensor | float]] | None,
    hamiltonian_terms: Sequence[tuple[Mapping[int, str], torch.Tensor | float]] | None,
) -> None:
    if gradient_policy not in {"exact", "approximate"}:
        raise ValueError("gradient_policy must be exact or approximate")
    if degeneracy_tolerance < 0:
        raise ValueError("degeneracy_tolerance must be non-negative")
    if initial_bond_dimension < 1:
        raise ValueError("initial_bond_dimension must be positive")
    if canonicalization_policy not in {"none", "dirty", "full"}:
        raise ValueError("canonicalization_policy must be none, dirty or full")
    if initial_mps_left_canonical and not initial_mps_tensors_provided:
        raise MPSReverseContractError(
            "initial_mps_left_canonical requires initial_mps_tensors"
        )
    if gradient_owner_ranks is not None and any(
        int(owner) < 0 or int(owner) >= world_size for owner in gradient_owner_ranks
    ):
        raise ValueError("gradient owner rank is outside the process group")
    if compile_site_kernels and any(
        instruction.name in {"rxx", "ryy", "rzz"}
        and tuple(map(int, instruction.wires))
        != tuple(sorted(map(int, instruction.wires)))
        for instruction in instructions
    ):
        raise MPSReverseContractError(
            "compiled site-sharded two-site rotations require ascending adjacent wire order"
        )
    if (
        sum(
            item is not None
            for item in (observable, observable_terms, hamiltonian_terms)
        )
        > 1
    ):
        raise ValueError(
            "observable, observable_terms and hamiltonian_terms are mutually exclusive"
        )


def _evaluate_reverse_objective(
    state: RankOwnedMPSState,
    tape: MPSReverseTape,
    *,
    observable: Mapping[int, str] | None,
    observable_terms: Sequence[tuple[Mapping[int, str], torch.Tensor | float]] | None,
    hamiltonian_terms: Sequence[tuple[Mapping[int, str], torch.Tensor | float]] | None,
    compile_observables: bool,
) -> tuple[torch.Tensor, dict[int, torch.Tensor], str]:
    if hamiltonian_terms is not None:
        coefficients = _parse_heisenberg_hamiltonian_terms(state, hamiltonian_terms)
        if coefficients is None:
            raise MPSReverseContractError(
                "hamiltonian_terms currently supports Z fields and adjacent "
                "XX, YY, or ZZ couplings"
            )
        value, adjoints = _heisenberg_mpo_energy_and_adjoints(state, coefficients)
        return value, adjoints, "heisenberg_mpo_five_channel_scan"
    if observable_terms is None:
        adjoint_wires = {wire for record in tape.records for wire in record.wires}
        value, adjoints = _expectation_and_adjoints(
            state,
            observable or {0: "z"},
            adjoint_wires=tuple(adjoint_wires),
        )
        return value, adjoints, "single_observable_scan"
    fused = _parse_z_zz_terms(state, observable_terms) is not None
    value, adjoints = _multi_observable_mse_and_adjoints(
        state, observable_terms, compiled_observables=compile_observables
    )
    execution = "fused_z_zz_channel_scan" if fused else "single_observable_scan"
    return value, adjoints, execution


def execute_torch_distributed_mps_reverse(
    circuit_or_ir: Any,
    *,
    observable: Mapping[int, str] | None = None,
    observable_terms: (
        Sequence[tuple[Mapping[int, str], torch.Tensor | float]] | None
    ) = None,
    hamiltonian_terms: (
        Sequence[tuple[Mapping[int, str], torch.Tensor | float]] | None
    ) = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    max_bond: int | None = None,
    cutoff: float = 0.0,
    gradient_policy: str = "exact",
    gradient_tolerance: float = 0.0,
    checkpoint_policy: MPSReverseCheckpointPolicy | None = None,
    tape_identity_override: str | None = None,
    reverse_delay_seconds: float = 0.0,
    degeneracy_tolerance: float = 1e-7,
    initial_bond_dimension: int = 1,
    initial_mps_tensors: Mapping[int, torch.Tensor] | None = None,
    initial_mps_left_canonical: bool = False,
    site_ownership: Sequence[Sequence[int]] | None = None,
    compile_site_kernels: bool = False,
    prefetch_layer_halos: bool = True,
    compile_observables: bool = False,
    fuse_local_reverse: bool = True,
    gradient_owner_ranks: Sequence[int] | None = None,
    gradient_bucket_bytes: int = 25 * 1024 * 1024,
    canonicalization_policy: str = "dirty",
    svd_driver: str | None = "gesvd",
) -> TorchDistributedMPSGradientResult:
    """Build an explicit rank-owned tape; call ``result.backward()`` on all ranks."""

    if not dist.is_initialized():
        raise RuntimeError("distributed MPS reverse requires torch.distributed")
    ir = ensure_circuit_ir(circuit_or_ir)
    rank, world = dist.get_rank(), dist.get_world_size()
    _validate_reverse_request(
        instructions=ir.instructions,
        gradient_policy=gradient_policy,
        degeneracy_tolerance=degeneracy_tolerance,
        initial_bond_dimension=initial_bond_dimension,
        initial_mps_tensors_provided=initial_mps_tensors is not None,
        initial_mps_left_canonical=initial_mps_left_canonical,
        canonicalization_policy=canonicalization_policy,
        compile_site_kernels=compile_site_kernels,
        gradient_owner_ranks=gradient_owner_ranks,
        world_size=world,
        observable=observable,
        observable_terms=observable_terms,
        hamiltonian_terms=hamiltonian_terms,
    )
    policy = checkpoint_policy or MPSReverseCheckpointPolicy()
    save_exact_factorizations = (
        policy.save_two_site_factorizations or gradient_policy == "exact"
    )
    parameters, instruction_parameter_indices = build_mps_parameter_layout(ir)
    initialization = initialize_reverse_mps_state(
        ir,
        rank=rank,
        world_size=world,
        device=device,
        dtype=dtype,
        max_bond=max_bond,
        cutoff=cutoff,
        svd_driver=svd_driver,
        initial_bond_dimension=initial_bond_dimension,
        initial_mps_tensors=initial_mps_tensors,
        site_ownership=site_ownership,
    )
    state = initialization.state
    local_tensors = state.local_tensors
    global_shapes = initialization.global_shapes
    bsz = initialization.batch_size
    resolved_device = initialization.device
    resolved_dtype = initialization.dtype
    records: list[MPSReverseTapeRecord] = []
    payloads: dict[int, _ReversePayload] = {}
    checkpoint_budget = ReverseCheckpointBudget(policy, resolved_device)
    initial_mps_canonical = initial_mps_tensors is None or initial_mps_left_canonical
    dirty_bonds = set() if initial_mps_canonical else set(range(ir.n_wires - 1))
    static_qr_metadata_records = 0
    dynamic_metadata_broadcasts = 0
    layer_halo_message_count = 0
    layer_halo_payload_bytes = 0
    layer_halo_intra_node_bytes = 0
    layer_halo_inter_node_bytes = 0
    layer_halo_wait_seconds = 0.0

    precomputed_ry: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    precomputed_rxx: dict[
        int,
        tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Mapping[str, Any]
        ],
    ] = {}
    precomputed_factorizations: dict[
        int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = {}
    precomputed_layer_metadata: dict[int, _ReverseRecordMetadata] = {}
    prefetched_rxx_indices: set[int] = set()
    prefetched_rxx_halos: dict[int, torch.Tensor] = {}
    reserved_ry: set[int] = set()
    reserved_rxx: set[int] = set()
    selected_factorizations: set[int] = set()
    info: Mapping[str, Any] | None
    metadata: _ReverseRecordMetadata | None
    for instruction_index, instruction in enumerate(ir.instructions):
        if (
            compile_site_kernels
            and instruction.name == "ry"
            and instruction_index not in reserved_ry
        ):
            layer = collect_compiled_mps_layer(ir.instructions, instruction_index)
            reserved_ry.update(index for index, _, _ in layer)
            precomputed_ry.update(
                _prepare_reverse_one_site_layer(
                    layer,
                    state=state,
                    checkpoint_budget=checkpoint_budget,
                    bsz=bsz,
                    device=resolved_device,
                    dtype=resolved_dtype,
                )
            )
        if (
            compile_site_kernels
            and instruction.name in {"rxx", "ryy", "rzz"}
            and instruction_index not in reserved_rxx
        ):
            layer = tuple(
                (index, candidate, min(wires))
                for index, candidate, wires in collect_compiled_mps_layer(
                    ir.instructions, instruction_index
                )
            )
            reserved_rxx.update(index for index, _, _ in layer)
            selected_factorizations.update(
                _reserve_reverse_two_site_layer(
                    layer,
                    state=state,
                    checkpoint_budget=checkpoint_budget,
                    global_shapes=global_shapes,
                    bsz=bsz,
                    dtype=resolved_dtype,
                    save_exact_factorizations=save_exact_factorizations,
                )
            )
            layer_prefetch, layer_prefetched_indices = (
                _begin_layer_halo_prefetch(layer, state, global_shapes)
                if prefetch_layer_halos
                else (None, frozenset())
            )
            prefetched_rxx_indices.update(layer_prefetched_indices)
            prepared, saved = _prepare_local_reverse_two_site_layer(
                layer,
                state=state,
                selected_factorizations=selected_factorizations,
                bsz=bsz,
                device=resolved_device,
                dtype=resolved_dtype,
            )
            precomputed_rxx.update(prepared)
            precomputed_factorizations.update(saved)
            layer_halo_wait_seconds += _finish_layer_halo_prefetch(layer_prefetch)
            if layer_prefetch is not None:
                prepared, saved = _prepare_prefetched_reverse_two_site_layer(
                    layer,
                    layer_prefetch,
                    state=state,
                    selected_factorizations=selected_factorizations,
                    bsz=bsz,
                    device=resolved_device,
                    dtype=resolved_dtype,
                )
                precomputed_rxx.update(prepared)
                precomputed_factorizations.update(saved)
                metadata, metadata_broadcasts = _collect_reverse_layer_metadata(
                    layer,
                    prepared,
                    state=state,
                    template=next(iter(local_tensors.values())),
                )
                precomputed_layer_metadata.update(metadata)
                dynamic_metadata_broadcasts += metadata_broadcasts
                layer_halo_message_count += layer_prefetch.message_count
                layer_halo_payload_bytes += layer_prefetch.payload_bytes
                layer_halo_intra_node_bytes += layer_prefetch.intra_node_payload_bytes
                layer_halo_inter_node_bytes += layer_prefetch.inter_node_payload_bytes
                prefetched_rxx_halos.update(layer_prefetch.received)
        wires = tuple(int(wire) for wire in instruction.wires)
        if len(wires) not in {1, 2} or (
            len(wires) == 2 and abs(wires[0] - wires[1]) != 1
        ):
            raise NonlocalMPSCompilationError(
                f"instruction {instruction_index}:{instruction.name} requires MPS routing"
            )
        if len(wires) == 1:
            owner = state.owner(wires[0])
            shape = global_shapes[wires[0]]
            prospective = (
                tensor_nbytes(state.local_tensors[wires[0]]) if rank == owner else 0
            )
            if instruction_index not in reserved_ry:
                checkpoint_budget.reserve(prospective)
            if rank == owner:
                if instruction_index in precomputed_ry:
                    before, after = precomputed_ry.pop(instruction_index)
                else:
                    before = state.local_tensors[wires[0]].detach().clone()
                    matrix = instruction_matrix_for_mps(
                        instruction,
                        bsz=bsz,
                        device=resolved_device,
                        dtype=resolved_dtype,
                    )
                    with torch.no_grad():
                        after = apply_one_mps_tensor(before, matrix)
                    state.local_tensors[wires[0]] = after
                payloads[len(records)] = _ReversePayload(
                    "one_site", (before,), instruction
                )
            records.append(
                _record(
                    index=len(records),
                    kind="one_site",
                    wires=wires,
                    compute_owner=owner,
                    owner_ranks=(owner,),
                    parameter_indices=instruction_parameter_indices[instruction_index],
                    input_shapes=(shape,),
                    output_shapes=(shape,),
                )
            )
            continue

        left_wire = min(wires)
        left_owner, right_owner = state.owner(left_wire), state.owner(left_wire + 1)
        sequence = 10_000 + len(records) * 2
        if instruction_index in prefetched_rxx_indices:
            prefetched_rxx_indices.remove(instruction_index)
            if rank == left_owner:
                halo = prefetched_rxx_halos.pop(instruction_index)
        elif left_owner != right_owner:
            if rank == right_owner:
                _send_static(
                    state.local_tensors[left_wire + 1],
                    destination=left_owner,
                    sequence=sequence,
                    shape=global_shapes[left_wire + 1],
                )
            elif rank == left_owner:
                halo = _recv_static(
                    next(iter(local_tensors.values())),
                    source=right_owner,
                    sequence=sequence,
                    shape=global_shapes[left_wire + 1],
                )
        elif rank == left_owner:
            halo = state.local_tensors[left_wire + 1]
        metadata = None
        prospective = 0
        optional = 0
        if rank == left_owner:
            prospective = tensor_nbytes(state.local_tensors[left_wire]) + tensor_nbytes(
                halo
            )
            if (
                save_exact_factorizations
                and instruction_index not in reserved_rxx
                and wires == tuple(sorted(wires))
            ):
                pair_bytes = (
                    bsz
                    * int(state.local_tensors[left_wire].shape[1])
                    * 2
                    * 2
                    * int(halo.shape[-1])
                    * state.local_tensors[left_wire].element_size()
                )
                optional = 4 * pair_bytes
        if instruction_index not in reserved_rxx:
            checkpoint_budget.reserve(prospective)
            if save_exact_factorizations and checkpoint_budget.reserve_factorization(
                optional
            ):
                selected_factorizations.add(instruction_index)
        if rank == left_owner:
            saved_factorization = precomputed_factorizations.pop(
                instruction_index, None
            )
            if instruction_index in precomputed_rxx:
                before_left, before_right, after_left, after_right, info = (
                    precomputed_rxx.pop(instruction_index)
                )
            else:
                before_left = state.local_tensors[left_wire].detach().clone()
                before_right = halo.detach().clone()
                if instruction_index in selected_factorizations and wires == tuple(
                    sorted(wires)
                ):
                    with torch.no_grad():
                        contracted = contract_mps_two_site_bucket(
                            (instruction,),
                            (before_left,),
                            (before_right,),
                            bsz=bsz,
                            device=resolved_device,
                            dtype=resolved_dtype,
                            compiled=compile_site_kernels,
                        )[0]
                    pair_leaf, after_left, after_right, info = factor_mps_reverse_pair(
                        contracted,
                        left_dim=int(before_left.shape[1]),
                        right_dim=int(before_right.shape[-1]),
                        config=state.config,
                    )
                    saved_factorization = (pair_leaf, after_left, after_right)
                else:
                    with torch.enable_grad():
                        outputs, info = apply_rank_local_mps_instruction(
                            instruction,
                            (
                                before_left.detach().requires_grad_(True),
                                before_right.detach().requires_grad_(True),
                            ),
                            state.config,
                            bsz=bsz,
                            device=resolved_device,
                            dtype=resolved_dtype,
                        )
                        after_left, after_right = outputs
                after_left = after_left.detach()
                after_right = after_right.detach()
            state.local_tensors[left_wire] = after_left
            if left_owner == right_owner:
                state.local_tensors[left_wire + 1] = after_right
            payloads[len(records)] = _ReversePayload(
                "two_site",
                (before_left, before_right),
                instruction,
                *(
                    (saved_factorization[0], saved_factorization[1:])
                    if saved_factorization is not None
                    else (None, None)
                ),
            )
            if info is None:
                raise MPSReverseContractError(
                    "two-site reverse execution requires split metadata"
                )
            metadata = {
                "input_shapes": (before_left.shape, before_right.shape),
                "output_shapes": (after_left.shape, after_right.shape),
                "split_info": dict(info),
            }
        if left_owner != right_owner:
            if rank == left_owner:
                _send(after_right, destination=right_owner, sequence=sequence + 1)
            elif rank == right_owner:
                state.local_tensors[left_wire + 1] = _recv(
                    state.local_tensors[left_wire + 1],
                    source=left_owner,
                    sequence=sequence + 1,
                )
        static_metadata = _static_exact_qr_record(
            global_shapes[left_wire],
            global_shapes[left_wire + 1],
            max_bond=max_bond,
            cutoff=cutoff,
        )
        if static_metadata is None:
            if instruction_index in precomputed_layer_metadata:
                metadata = precomputed_layer_metadata.pop(instruction_index)
            else:
                dynamic_metadata_broadcasts += 1
                metadata = _broadcast_record(
                    metadata, left_owner, next(iter(local_tensors.values()))
                )
        else:
            static_qr_metadata_records += 1
            if rank == left_owner:
                if metadata is None:
                    raise MPSReverseContractError(
                        "compute owner is missing split metadata"
                    )
                if (
                    metadata["input_shapes"] != static_metadata["input_shapes"]
                    or metadata["output_shapes"] != static_metadata["output_shapes"]
                ):
                    raise MPSReverseContractError(
                        "owner-local exact QR shapes differ from the static MPS plan"
                    )
            metadata = static_metadata
        split = metadata["split_info"]
        full_rank = int(split.get("original_rank", split.get("rank", 0)))
        split["method"] = (
            "qr"
            if cutoff <= 0 and (max_bond is None or int(max_bond) >= full_rank)
            else "svd"
        )
        for wire, output_shape in zip(
            (left_wire, left_wire + 1), metadata["output_shapes"]
        ):
            batch, left_bond, physical, right_bond = output_shape
            global_shapes[wire] = (batch, left_bond, physical, right_bond)
        dirty_bonds.discard(left_wire)
        if left_wire + 1 < ir.n_wires - 1:
            dirty_bonds.add(left_wire + 1)
        records.append(
            _record(
                index=len(records),
                kind="two_site",
                wires=wires,
                compute_owner=left_owner,
                owner_ranks=tuple(sorted({left_owner, right_owner})),
                parameter_indices=instruction_parameter_indices[instruction_index],
                **metadata,
            )
        )

    # A two-site split already returns a valid MPS factorization and its VJP is
    # recorded explicitly.  A full-chain QR sweep is therefore optional when
    # the supplied initial state is certified left-canonical; expectation and
    # reverse contractions do not require the final state to remain canonical.
    dirty_bonds_before_canonicalization = tuple(sorted(dirty_bonds))
    canonical_wires = _planned_canonicalization_bonds(
        ir.n_wires, dirty_bonds_before_canonicalization, canonicalization_policy
    )
    for left_wire in canonical_wires:
        left_owner, right_owner = state.owner(left_wire), state.owner(left_wire + 1)
        sequence = 1_000_000 + len(records) * 2
        if left_owner != right_owner:
            if rank == right_owner:
                _send_static(
                    state.local_tensors[left_wire + 1],
                    destination=left_owner,
                    sequence=sequence,
                    shape=global_shapes[left_wire + 1],
                )
            elif rank == left_owner:
                halo = _recv_static(
                    next(iter(local_tensors.values())),
                    source=right_owner,
                    sequence=sequence,
                    shape=global_shapes[left_wire + 1],
                )
        elif rank == left_owner:
            halo = state.local_tensors[left_wire + 1]
        metadata = None
        prospective = 0
        if rank == left_owner:
            prospective = tensor_nbytes(state.local_tensors[left_wire]) + tensor_nbytes(
                halo
            )
        checkpoint_budget.reserve(prospective)
        if rank == left_owner:
            before_left = state.local_tensors[left_wire].detach().clone()
            before_right = halo.detach().clone()
            with torch.no_grad():
                after_left, after_right = _qr_forward(before_left, before_right)
            state.local_tensors[left_wire] = after_left
            if left_owner == right_owner:
                state.local_tensors[left_wire + 1] = after_right
            payloads[len(records)] = _ReversePayload(
                "canonicalize_left", (before_left, before_right), None
            )
            metadata = {
                "input_shapes": (before_left.shape, before_right.shape),
                "output_shapes": (after_left.shape, after_right.shape),
                "split_info": {},
            }
        if left_owner != right_owner:
            if rank == left_owner:
                _send(after_right, destination=right_owner, sequence=sequence + 1)
            elif rank == right_owner:
                state.local_tensors[left_wire + 1] = _recv(
                    state.local_tensors[left_wire + 1],
                    source=left_owner,
                    sequence=sequence + 1,
                )
        metadata = _broadcast_record(
            metadata, left_owner, next(iter(local_tensors.values()))
        )
        metadata["split_info"]["method"] = "qr"
        for wire, output_shape in zip(
            (left_wire, left_wire + 1), metadata["output_shapes"]
        ):
            batch, left_bond, physical, right_bond = output_shape
            global_shapes[wire] = (batch, left_bond, physical, right_bond)
        records.append(
            _record(
                index=len(records),
                kind="canonicalize_left",
                wires=(left_wire, left_wire + 1),
                compute_owner=left_owner,
                owner_ranks=tuple(sorted({left_owner, right_owner})),
                **metadata,
            )
        )

    tape = MPSReverseTape.build(
        records,
        saved_tensor_bytes=checkpoint_budget.saved_bytes,
        checkpoint_policy=policy,
    )
    tape.validate_distributed(identity_override=tape_identity_override)
    discarded = sum(record.discarded_weight for record in tape.records)
    if gradient_policy == "exact" and discarded > 0:
        raise MPSReverseContractError(
            "exact gradient policy rejects truncated MPS forward"
        )
    if gradient_policy == "approximate" and discarded > gradient_tolerance:
        raise MPSReverseContractError(
            f"discarded weight {discarded:.9g} exceeds gradient tolerance "
            f"{gradient_tolerance:.9g}"
        )
    _validate_svd_gaps(
        tape,
        degeneracy_tolerance,
        allow_degenerate=gradient_policy == "approximate",
    )
    value, adjoints, objective_execution = _evaluate_reverse_objective(
        state,
        tape,
        observable=observable,
        observable_terms=observable_terms,
        hamiltonian_terms=hamiltonian_terms,
        compile_observables=compile_observables,
    )
    ownership_records = build_mps_gradient_ownership(
        tape,
        len(parameters),
        world_size=world,
        optimizer_owner_ranks=gradient_owner_ranks,
    )
    reverse_segments, reverse_segment_cache_hit = cached_mps_reverse_segments(
        tape, fuse_owner_local=fuse_local_reverse
    )
    fused_segment_count = sum(len(segment) > 1 for segment in reverse_segments)
    reverse_autograd_calls = sum(
        segment[0].compute_owner == rank for segment in reverse_segments
    )
    gradient_buckets, gradient_bucket_cache_hit = cached_mps_gradient_buckets(
        parameters, gradient_owner_ranks, max_bucket_bytes=gradient_bucket_bytes
    )
    gradient_bucket_payload_bytes = sum(
        sum(end - start for _, start, end in pieces)
        * torch.empty((), dtype=dtype).element_size()
        for dtype, _, pieces in gradient_buckets
    )

    backward = build_mps_reverse_backward(
        state=state,
        payloads=payloads,
        payload_factory=_ReversePayload,
        parameters=parameters,
        adjoints=adjoints,
        local_tensors=local_tensors,
        reverse_segments=reverse_segments,
        gradient_buckets=gradient_buckets,
        rank=rank,
        batch_size=bsz,
        device=resolved_device,
        dtype=resolved_dtype,
        compile_site_kernels=compile_site_kernels,
        reverse_delay_seconds=reverse_delay_seconds,
        gradient_owner_ranks=gradient_owner_ranks,
    )

    return TorchDistributedMPSGradientResult(
        value=value,
        parameters=parameters,
        tape=tape,
        ownership=ownership_records,
        discarded_weight=discarded,
        gradient_policy=gradient_policy,
        gradient_tolerance=gradient_tolerance,
        world_size=world,
        rank=rank,
        _backward=backward,
        objective_scan_pairs=int(objective_execution != "single_observable_scan"),
        objective_scan_forward_messages=(
            world - 1 if objective_execution != "single_observable_scan" else 0
        ),
        objective_scan_reverse_messages=(
            world - 1 if objective_execution != "single_observable_scan" else 0
        ),
        objective_execution=objective_execution,
        reverse_tape_segments=len(reverse_segments),
        fused_reverse_segments=fused_segment_count,
        reverse_autograd_grad_invocations=reverse_autograd_calls,
        reverse_python_dispatches=len(reverse_segments),
        gradient_collective_count=len(gradient_buckets),
        gradient_collective_bytes=gradient_bucket_payload_bytes,
        gradient_bucket_count=len(gradient_buckets),
        gradient_bucket_fill_ratio=(
            gradient_bucket_payload_bytes
            / (len(gradient_buckets) * gradient_bucket_bytes)
            if gradient_buckets
            else 0.0
        ),
        gradient_owner_ranks=(
            tuple(int(owner) for owner in gradient_owner_ranks)
            if gradient_owner_ranks is not None
            else ()
        ),
        canonicalization_policy=canonicalization_policy,
        initial_mps_canonical=initial_mps_canonical,
        dirty_bonds=dirty_bonds_before_canonicalization,
        planned_canonicalization_bonds=tuple(canonical_wires),
        qr_factorization_count=sum(
            record.factorization_method == "qr" for record in tape.records
        ),
        svd_factorization_count=sum(
            record.factorization_method == "svd" for record in tape.records
        ),
        static_qr_metadata_records=static_qr_metadata_records,
        dynamic_metadata_broadcasts=dynamic_metadata_broadcasts,
        reverse_segment_cache_hit=reverse_segment_cache_hit,
        gradient_bucket_cache_hit=gradient_bucket_cache_hit,
        layer_halo_message_count=layer_halo_message_count,
        layer_halo_payload_bytes=layer_halo_payload_bytes,
        layer_halo_intra_node_bytes=layer_halo_intra_node_bytes,
        layer_halo_inter_node_bytes=layer_halo_inter_node_bytes,
        layer_halo_wait_seconds=layer_halo_wait_seconds,
    )


__all__ = [
    "MPSReverseCheckpointPolicy",
    "MPSReverseContractError",
    "SiteShardedZZScanResult",
    "TorchDistributedMPSGradientResult",
    "execute_torch_distributed_mps_reverse",
    "site_sharded_z_zz_objective_pipeline",
    "site_sharded_z_zz_observations",
]
