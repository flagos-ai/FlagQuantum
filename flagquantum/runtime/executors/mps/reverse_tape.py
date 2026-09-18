"""Reverse-tape construction for rank-owned distributed MPS."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ....core.ir import CircuitIR, Instruction
from ....simulation.mps.compiled_layers import (
    apply_compiled_mps_one_site_bucket,
    contract_mps_two_site_bucket,
)
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
    MPSReverseContractError,
    MPSReverseTapeRecord,
    ReverseCheckpointBudget,
    build_mps_reverse_tape_record,
)
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
)
from .state import RankOwnedMPSState

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
                chunk, factorizations, left_inputs, right_inputs, strict=True
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
    info: Mapping[str, Any] | None
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
    return all_reduce_reverse_layer_records(local_metadata, entries, template), int(
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


class ReverseTapeBuilder:
    """Record one reverse-mode tape entry per instruction, and canonicalize."""

    def __init__(
        self,
        ir: CircuitIR,
        *,
        rank: int,
        state: RankOwnedMPSState,
        local_tensors: dict[int, torch.Tensor],
        global_shapes: dict[int, tuple[int, int, int, int]],
        bsz: int,
        resolved_device: torch.device,
        resolved_dtype: torch.dtype,
        checkpoint_budget: ReverseCheckpointBudget,
        save_exact_factorizations: bool,
        compile_site_kernels: bool,
        prefetch_layer_halos: bool,
        max_bond: int | None,
        cutoff: float,
        instruction_parameter_indices: tuple[tuple[int, ...], ...],
        records: list[MPSReverseTapeRecord],
        payloads: dict[int, _ReversePayload],
        dirty_bonds: set[int],
        reserved_ry: set[int],
        reserved_rxx: set[int],
        selected_factorizations: set[int],
        precomputed_ry: dict[int, tuple[torch.Tensor, torch.Tensor]],
        precomputed_rxx: dict[int, _PreparedTwoSite],
        precomputed_factorizations: dict[int, _SavedFactorization],
        precomputed_layer_metadata: dict[int, _ReverseRecordMetadata],
        prefetched_rxx_indices: set[int],
        prefetched_rxx_halos: dict[int, torch.Tensor],
    ) -> None:
        self.ir = ir
        self.rank = rank
        self.state = state
        self.local_tensors = local_tensors
        self.global_shapes = global_shapes
        self.bsz = bsz
        self.resolved_device = resolved_device
        self.resolved_dtype = resolved_dtype
        self.checkpoint_budget = checkpoint_budget
        self.save_exact_factorizations = save_exact_factorizations
        self.compile_site_kernels = compile_site_kernels
        self.prefetch_layer_halos = prefetch_layer_halos
        self.max_bond = max_bond
        self.cutoff = cutoff
        self.instruction_parameter_indices = instruction_parameter_indices
        self.records = records
        self.payloads = payloads
        self.dirty_bonds = dirty_bonds
        self.reserved_ry = reserved_ry
        self.reserved_rxx = reserved_rxx
        self.selected_factorizations = selected_factorizations
        self.precomputed_ry = precomputed_ry
        self.precomputed_rxx = precomputed_rxx
        self.precomputed_factorizations = precomputed_factorizations
        self.precomputed_layer_metadata = precomputed_layer_metadata
        self.prefetched_rxx_indices = prefetched_rxx_indices
        self.prefetched_rxx_halos = prefetched_rxx_halos
        self.static_qr_metadata_records: int = 0
        self.dynamic_metadata_broadcasts: int = 0
        self.layer_halo_message_count: int = 0
        self.layer_halo_payload_bytes: int = 0
        self.layer_halo_intra_node_bytes: int = 0
        self.layer_halo_inter_node_bytes: int = 0
        self.layer_halo_wait_seconds: float = 0.0

    def record(self, instruction_index: int, instruction: Instruction) -> None:
        """Replay one instruction forward and append the entry it needs."""

        self._capture_compiled_layer(instruction_index, instruction)
        wires = tuple(int(wire) for wire in instruction.wires)
        if len(wires) not in {1, 2} or (
            len(wires) == 2 and abs(wires[0] - wires[1]) != 1
        ):
            raise NonlocalMPSCompilationError(
                f"instruction {instruction_index}:{instruction.name} requires MPS routing"
            )
        if len(wires) == 1:
            self._record_one_site(instruction_index, instruction, wires)
            return
        self._record_two_site(instruction_index, instruction, wires)

    def _capture_compiled_layer(
        self,
        instruction_index: int,
        instruction: Instruction,
    ) -> None:
        """Precompute the compiled site layer this instruction begins, if any."""

        if (
            self.compile_site_kernels
            and instruction.name == "ry"
            and instruction_index not in self.reserved_ry
        ):
            layer = collect_compiled_mps_layer(self.ir.instructions, instruction_index)
            self.reserved_ry.update(index for index, _, _ in layer)
            self.precomputed_ry.update(
                _prepare_reverse_one_site_layer(
                    layer,
                    state=self.state,
                    checkpoint_budget=self.checkpoint_budget,
                    bsz=self.bsz,
                    device=self.resolved_device,
                    dtype=self.resolved_dtype,
                )
            )
        if (
            self.compile_site_kernels
            and instruction.name in {"rxx", "ryy", "rzz"}
            and instruction_index not in self.reserved_rxx
        ):
            two_site_layer = tuple(
                (index, candidate, min(wires))
                for index, candidate, wires in collect_compiled_mps_layer(
                    self.ir.instructions, instruction_index
                )
            )
            self.reserved_rxx.update(index for index, _, _ in two_site_layer)
            self.selected_factorizations.update(
                _reserve_reverse_two_site_layer(
                    two_site_layer,
                    state=self.state,
                    checkpoint_budget=self.checkpoint_budget,
                    global_shapes=self.global_shapes,
                    bsz=self.bsz,
                    dtype=self.resolved_dtype,
                    save_exact_factorizations=self.save_exact_factorizations,
                )
            )
            layer_prefetch, layer_prefetched_indices = (
                begin_reverse_layer_halo_prefetch(
                    two_site_layer, self.state, self.global_shapes
                )
                if self.prefetch_layer_halos
                else (None, frozenset())
            )
            self.prefetched_rxx_indices.update(layer_prefetched_indices)
            prepared, saved = _prepare_local_reverse_two_site_layer(
                two_site_layer,
                state=self.state,
                selected_factorizations=self.selected_factorizations,
                bsz=self.bsz,
                device=self.resolved_device,
                dtype=self.resolved_dtype,
            )
            self.precomputed_rxx.update(prepared)
            self.precomputed_factorizations.update(saved)
            self.layer_halo_wait_seconds += finish_reverse_layer_halo_prefetch(
                layer_prefetch
            )
            if layer_prefetch is not None:
                prepared, saved = _prepare_prefetched_reverse_two_site_layer(
                    two_site_layer,
                    layer_prefetch,
                    state=self.state,
                    selected_factorizations=self.selected_factorizations,
                    bsz=self.bsz,
                    device=self.resolved_device,
                    dtype=self.resolved_dtype,
                )
                self.precomputed_rxx.update(prepared)
                self.precomputed_factorizations.update(saved)
                layer_metadata, metadata_broadcasts = _collect_reverse_layer_metadata(
                    two_site_layer,
                    prepared,
                    state=self.state,
                    template=next(iter(self.local_tensors.values())),
                )
                self.precomputed_layer_metadata.update(layer_metadata)
                self.dynamic_metadata_broadcasts += metadata_broadcasts
                self.layer_halo_message_count += layer_prefetch.message_count
                self.layer_halo_payload_bytes += layer_prefetch.payload_bytes
                self.layer_halo_intra_node_bytes += (
                    layer_prefetch.intra_node_payload_bytes
                )
                self.layer_halo_inter_node_bytes += (
                    layer_prefetch.inter_node_payload_bytes
                )
                self.prefetched_rxx_halos.update(layer_prefetch.received)

    def _record_one_site(
        self,
        instruction_index: int,
        instruction: Instruction,
        wires: tuple[int, ...],
    ) -> None:
        """Record one rank-local one-site gate."""

        if len(wires) == 1:
            owner = self.state.owner(wires[0])
            shape = self.global_shapes[wires[0]]
            prospective = (
                tensor_nbytes(self.state.local_tensors[wires[0]])
                if self.rank == owner
                else 0
            )
            if instruction_index not in self.reserved_ry:
                self.checkpoint_budget.reserve(prospective)
            if self.rank == owner:
                if instruction_index in self.precomputed_ry:
                    before, after = self.precomputed_ry.pop(instruction_index)
                else:
                    before = self.state.local_tensors[wires[0]].detach().clone()
                    matrix = instruction_matrix_for_mps(
                        instruction,
                        bsz=self.bsz,
                        device=self.resolved_device,
                        dtype=self.resolved_dtype,
                    )
                    with torch.no_grad():
                        after = apply_one_mps_tensor(before, matrix)
                    self.state.local_tensors[wires[0]] = after
                self.payloads[len(self.records)] = _ReversePayload(
                    "one_site", (before,), instruction
                )
            self.records.append(
                build_mps_reverse_tape_record(
                    index=len(self.records),
                    kind="one_site",
                    wires=wires,
                    compute_owner=owner,
                    owner_ranks=(owner,),
                    parameter_indices=self.instruction_parameter_indices[
                        instruction_index
                    ],
                    input_shapes=(shape,),
                    output_shapes=(shape,),
                )
            )
            return

    def _record_two_site(
        self,
        instruction_index: int,
        instruction: Instruction,
        wires: tuple[int, ...],
    ) -> None:
        """Record one nearest-neighbour two-site gate and its metadata."""

        info: Mapping[str, Any] | None
        metadata: _ReverseRecordMetadata | None
        left_wire = min(wires)
        left_owner, right_owner = self.state.owner(left_wire), self.state.owner(
            left_wire + 1
        )
        sequence = 10_000 + len(self.records) * 2
        if instruction_index in self.prefetched_rxx_indices:
            self.prefetched_rxx_indices.remove(instruction_index)
            if self.rank == left_owner:
                halo = self.prefetched_rxx_halos.pop(instruction_index)
        elif left_owner != right_owner:
            if self.rank == right_owner:
                send_static_reverse_tensor(
                    self.state.local_tensors[left_wire + 1],
                    destination=left_owner,
                    sequence=sequence,
                    shape=self.global_shapes[left_wire + 1],
                )
            elif self.rank == left_owner:
                halo = receive_static_reverse_tensor(
                    next(iter(self.local_tensors.values())),
                    source=right_owner,
                    sequence=sequence,
                    shape=self.global_shapes[left_wire + 1],
                )
        elif self.rank == left_owner:
            halo = self.state.local_tensors[left_wire + 1]
        metadata = None
        prospective = 0
        optional = 0
        if self.rank == left_owner:
            prospective = tensor_nbytes(
                self.state.local_tensors[left_wire]
            ) + tensor_nbytes(halo)
            if (
                self.save_exact_factorizations
                and instruction_index not in self.reserved_rxx
                and wires == tuple(sorted(wires))
            ):
                pair_bytes = (
                    self.bsz
                    * int(self.state.local_tensors[left_wire].shape[1])
                    * 2
                    * 2
                    * int(halo.shape[-1])
                    * self.state.local_tensors[left_wire].element_size()
                )
                optional = 4 * pair_bytes
        if instruction_index not in self.reserved_rxx:
            self.checkpoint_budget.reserve(prospective)
            if (
                self.save_exact_factorizations
                and self.checkpoint_budget.reserve_factorization(optional)
            ):
                self.selected_factorizations.add(instruction_index)
        if self.rank == left_owner:
            saved_factorization = self.precomputed_factorizations.pop(
                instruction_index, None
            )
            if instruction_index in self.precomputed_rxx:
                before_left, before_right, after_left, after_right, info = (
                    self.precomputed_rxx.pop(instruction_index)
                )
            else:
                before_left = self.state.local_tensors[left_wire].detach().clone()
                before_right = halo.detach().clone()
                if (
                    instruction_index in self.selected_factorizations
                    and wires == tuple(sorted(wires))
                ):
                    with torch.no_grad():
                        contracted = contract_mps_two_site_bucket(
                            (instruction,),
                            (before_left,),
                            (before_right,),
                            bsz=self.bsz,
                            device=self.resolved_device,
                            dtype=self.resolved_dtype,
                            compiled=self.compile_site_kernels,
                        )[0]
                    pair_leaf, after_left, after_right, info = factor_mps_reverse_pair(
                        contracted,
                        left_dim=int(before_left.shape[1]),
                        right_dim=int(before_right.shape[-1]),
                        config=self.state.config,
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
                            self.state.config,
                            bsz=self.bsz,
                            device=self.resolved_device,
                            dtype=self.resolved_dtype,
                        )
                        after_left, after_right = outputs
                after_left = after_left.detach()
                after_right = after_right.detach()
            self.state.local_tensors[left_wire] = after_left
            if left_owner == right_owner:
                self.state.local_tensors[left_wire + 1] = after_right
            self.payloads[len(self.records)] = _ReversePayload(
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
            if self.rank == left_owner:
                send_reverse_tensor(
                    after_right, destination=right_owner, sequence=sequence + 1
                )
            elif self.rank == right_owner:
                self.state.local_tensors[left_wire + 1] = receive_reverse_tensor(
                    self.state.local_tensors[left_wire + 1],
                    source=left_owner,
                    sequence=sequence + 1,
                )
        static_metadata = _static_exact_qr_record(
            self.global_shapes[left_wire],
            self.global_shapes[left_wire + 1],
            max_bond=self.max_bond,
            cutoff=self.cutoff,
        )
        if static_metadata is None:
            if instruction_index in self.precomputed_layer_metadata:
                metadata = self.precomputed_layer_metadata.pop(instruction_index)
            else:
                self.dynamic_metadata_broadcasts += 1
                metadata = broadcast_reverse_record(
                    metadata, left_owner, next(iter(self.local_tensors.values()))
                )
        else:
            self.static_qr_metadata_records += 1
            if self.rank == left_owner:
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
            if self.cutoff <= 0
            and (self.max_bond is None or int(self.max_bond) >= full_rank)
            else "svd"
        )
        for wire, output_shape in zip(
            (left_wire, left_wire + 1), metadata["output_shapes"], strict=True
        ):
            batch, left_bond, physical, right_bond = output_shape
            self.global_shapes[wire] = (batch, left_bond, physical, right_bond)
        self.dirty_bonds.discard(left_wire)
        if left_wire + 1 < self.ir.n_wires - 1:
            self.dirty_bonds.add(left_wire + 1)
        self.records.append(
            build_mps_reverse_tape_record(
                index=len(self.records),
                kind="two_site",
                wires=wires,
                compute_owner=left_owner,
                owner_ranks=tuple(sorted({left_owner, right_owner})),
                parameter_indices=self.instruction_parameter_indices[instruction_index],
                **metadata,
            )
        )

    def canonicalize_bond(self, left_wire: int) -> None:
        """Sweep one planned bond, leaving the pair left-canonical."""

        metadata: _ReverseRecordMetadata | None
        left_owner, right_owner = self.state.owner(left_wire), self.state.owner(
            left_wire + 1
        )
        sequence = 1_000_000 + len(self.records) * 2
        if left_owner != right_owner:
            if self.rank == right_owner:
                send_static_reverse_tensor(
                    self.state.local_tensors[left_wire + 1],
                    destination=left_owner,
                    sequence=sequence,
                    shape=self.global_shapes[left_wire + 1],
                )
            elif self.rank == left_owner:
                halo = receive_static_reverse_tensor(
                    next(iter(self.local_tensors.values())),
                    source=right_owner,
                    sequence=sequence,
                    shape=self.global_shapes[left_wire + 1],
                )
        elif self.rank == left_owner:
            halo = self.state.local_tensors[left_wire + 1]
        metadata = None
        prospective = 0
        if self.rank == left_owner:
            prospective = tensor_nbytes(
                self.state.local_tensors[left_wire]
            ) + tensor_nbytes(halo)
        self.checkpoint_budget.reserve(prospective)
        if self.rank == left_owner:
            before_left = self.state.local_tensors[left_wire].detach().clone()
            before_right = halo.detach().clone()
            with torch.no_grad():
                after_left, after_right = mps_qr_forward(before_left, before_right)
            self.state.local_tensors[left_wire] = after_left
            if left_owner == right_owner:
                self.state.local_tensors[left_wire + 1] = after_right
            self.payloads[len(self.records)] = _ReversePayload(
                "canonicalize_left", (before_left, before_right), None
            )
            metadata = {
                "input_shapes": (before_left.shape, before_right.shape),
                "output_shapes": (after_left.shape, after_right.shape),
                "split_info": {},
            }
        if left_owner != right_owner:
            if self.rank == left_owner:
                send_reverse_tensor(
                    after_right, destination=right_owner, sequence=sequence + 1
                )
            elif self.rank == right_owner:
                self.state.local_tensors[left_wire + 1] = receive_reverse_tensor(
                    self.state.local_tensors[left_wire + 1],
                    source=left_owner,
                    sequence=sequence + 1,
                )
        metadata = broadcast_reverse_record(
            metadata, left_owner, next(iter(self.local_tensors.values()))
        )
        metadata["split_info"]["method"] = "qr"
        for wire, output_shape in zip(
            (left_wire, left_wire + 1), metadata["output_shapes"], strict=True
        ):
            batch, left_bond, physical, right_bond = output_shape
            self.global_shapes[wire] = (batch, left_bond, physical, right_bond)
        self.records.append(
            build_mps_reverse_tape_record(
                index=len(self.records),
                kind="canonicalize_left",
                wires=(left_wire, left_wire + 1),
                compute_owner=left_owner,
                owner_ranks=tuple(sorted({left_owner, right_owner})),
                **metadata,
            )
        )


__all__ = ("ReverseTapeBuilder",)
