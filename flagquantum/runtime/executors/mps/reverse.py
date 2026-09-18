"""Deterministic explicit reverse mode for rank-owned distributed MPS."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.distributed as dist

from ....core.ir import Instruction, ensure_circuit_ir
from .records import (
    MPSReverseCheckpointPolicy,
    MPSReverseContractError,
    MPSReverseTape,
    MPSReverseTapeRecord,
    ReverseCheckpointBudget,
    TorchDistributedMPSGradientResult,
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
from .reverse_tape import ReverseTapeBuilder, _ReversePayload
from .reverse_transport import ReverseLayerHaloPrefetch, _ReverseRecordMetadata
from .reverse_z_observables import (
    mps_multi_observable_mse_and_adjoints,
    site_sharded_z_zz_objective_pipeline,
    site_sharded_z_zz_observations,
)
from .state import (
    RankOwnedMPSState,
    initialize_reverse_mps_state,
)

_LayerHaloPrefetch = ReverseLayerHaloPrefetch


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
        coefficients = parse_mps_heisenberg_terms(state, hamiltonian_terms)
        if coefficients is None:
            raise MPSReverseContractError(
                "hamiltonian_terms currently supports Z fields and adjacent "
                "XX, YY, or ZZ couplings"
            )
        value, adjoints = mps_heisenberg_energy_and_adjoints(state, coefficients)
        return value, adjoints, "heisenberg_mpo_five_channel_scan"
    if observable_terms is None:
        adjoint_wires = {wire for record in tape.records for wire in record.wires}
        value, adjoints = mps_expectation_and_adjoints(
            state,
            observable or {0: "z"},
            adjoint_wires=tuple(adjoint_wires),
        )
        return value, adjoints, "single_observable_scan"
    fused = parse_mps_z_zz_terms(state, observable_terms) is not None
    value, adjoints = mps_multi_observable_mse_and_adjoints(
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

    builder = ReverseTapeBuilder(
        ir,
        rank=rank,
        state=state,
        local_tensors=local_tensors,
        global_shapes=global_shapes,
        bsz=bsz,
        resolved_device=resolved_device,
        resolved_dtype=resolved_dtype,
        checkpoint_budget=checkpoint_budget,
        save_exact_factorizations=save_exact_factorizations,
        compile_site_kernels=compile_site_kernels,
        prefetch_layer_halos=prefetch_layer_halos,
        max_bond=max_bond,
        cutoff=cutoff,
        instruction_parameter_indices=instruction_parameter_indices,
        records=records,
        payloads=payloads,
        dirty_bonds=dirty_bonds,
        reserved_ry=reserved_ry,
        reserved_rxx=reserved_rxx,
        selected_factorizations=selected_factorizations,
        precomputed_ry=precomputed_ry,
        precomputed_rxx=precomputed_rxx,
        precomputed_factorizations=precomputed_factorizations,
        precomputed_layer_metadata=precomputed_layer_metadata,
        prefetched_rxx_indices=prefetched_rxx_indices,
        prefetched_rxx_halos=prefetched_rxx_halos,
    )
    for instruction_index, instruction in enumerate(ir.instructions):
        builder.record(instruction_index, instruction)

    # A two-site split already returns a valid MPS factorization and its VJP is
    # recorded explicitly.  A full-chain QR sweep is therefore optional when
    # the supplied initial state is certified left-canonical; expectation and
    # reverse contractions do not require the final state to remain canonical.
    dirty_bonds_before_canonicalization = tuple(sorted(dirty_bonds))
    canonical_wires = plan_mps_canonicalization_bonds(
        ir.n_wires, dirty_bonds_before_canonicalization, canonicalization_policy
    )
    for left_wire in canonical_wires:
        builder.canonicalize_bond(left_wire)

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
    validate_mps_svd_gaps(
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
        static_qr_metadata_records=builder.static_qr_metadata_records,
        dynamic_metadata_broadcasts=builder.dynamic_metadata_broadcasts,
        reverse_segment_cache_hit=reverse_segment_cache_hit,
        gradient_bucket_cache_hit=gradient_bucket_cache_hit,
        layer_halo_message_count=builder.layer_halo_message_count,
        layer_halo_payload_bytes=builder.layer_halo_payload_bytes,
        layer_halo_intra_node_bytes=builder.layer_halo_intra_node_bytes,
        layer_halo_inter_node_bytes=builder.layer_halo_inter_node_bytes,
        layer_halo_wait_seconds=builder.layer_halo_wait_seconds,
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
