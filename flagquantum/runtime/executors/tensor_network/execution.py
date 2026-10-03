"""Independent-slice tensor-network execution and output reduction.

Each rank contracts complete, disjoint slice assignments and the resulting
outputs are summed. This path supports differentiable reductions and a local
development mirror. It does not execute the owned or sharded intermediate DAG
handled by :mod:`distributed_execution`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from ....compute import get_platform_runtime
from ....simulation.mps.rank_local import (
    tensor_nbytes as _tensor_nbytes,
)
from ....simulation.tensor_network.contraction import (
    _build_slicing_plan,
    _slice_nodes,
)
from ....simulation.tensor_network.entrypoints import (
    _amplitude_batch_projection,
    _amplitude_projection,
    build_tensor_network,
    build_tensor_network_expectation,
    run_tensor_network,
)
from ....simulation.tensor_network.models import (
    PairContractionStep,
    TensorNetworkNode,
    TensorNetworkSlicingPlan,
)
from ....simulation.tensor_network.observables import _expectation_batch_projection
from ....simulation.tensor_network.path_search import (
    _contract_nodes_quality_multistart,
    _label_dims,
)
from ....simulation.tensor_network.stages import (
    execute_pair_steps as _execute_pair_steps,
)
from ...distributed.backend_policy import (
    DistributedBackendPolicy,
    _resolve_backend_policy,
    _should_use_torch_distributed,
)
from ...distributed.context import (
    TorchDistributedContext,
    _rank_placement_summary,
    init_torch_distributed,
    resolve_node_count,
)
from ...planner.tn_calibration import TNWorkingSetCalibration
from .joint_planning import (
    DistributedTNWorkingSetPolicy,
)
from .plan_cache import (
    _load_persistent_plan,
    _persistent_plan_key,
    _write_persistent_plan,
)
from .sliced_tasks import DistributedTNSliceTask, plan_distributed_tn_slice_tasks
from .state import (
    DistributedTensorNetworkAmplitude,
    DistributedTensorNetworkAmplitudes,
    DistributedTensorNetworkExpectation,
    DistributedTensorNetworkExpectations,
    DistributedTensorNetworkState,
    _context_claim_evidence_type,
)


def _execution_slice_tasks(
    slicing: TensorNetworkSlicingPlan,
    *,
    world_size: int,
    local_world_size: int,
) -> tuple[DistributedTNSliceTask, ...]:
    """Use the canonical topology-aware task plan for TN execution."""

    return plan_distributed_tn_slice_tasks(
        slicing,
        world_size=world_size,
        local_world_size=local_world_size,
    ).tasks


class _DistributedAllReduceSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, tensor: torch.Tensor) -> torch.Tensor:
        out = tensor.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out

    @staticmethod
    def backward(ctx: object, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        return (grad_output,)


def _all_reduce_sum_autograd(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.requires_grad:
        out = tensor.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out
    apply: Callable[[torch.Tensor], object] = _DistributedAllReduceSum.apply
    result = apply(tensor)
    if not isinstance(result, torch.Tensor):
        raise TypeError("TN slice reduction must return a tensor")
    return result


def _zero_for_output(
    nodes: Sequence[TensorNetworkNode], output_labels: Sequence[int]
) -> torch.Tensor:
    dims = _label_dims(nodes)
    reference = nodes[0].tensor if nodes else torch.empty((), dtype=torch.complex64)
    return torch.zeros(
        tuple(dims[label] for label in output_labels),
        dtype=reference.dtype,
        device=reference.device,
    )


def _contract_assigned_tensor_slices(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    tasks: Sequence[DistributedTNSliceTask],
    *,
    steps: Sequence[PairContractionStep] | None = None,
) -> torch.Tensor:
    partial = _zero_for_output(nodes, output_labels)
    if not tasks:
        return partial
    for task in tasks:
        subnodes = _slice_nodes(nodes, dict(task.assignments))
        selected_steps = (
            tuple(steps)
            if steps is not None
            else _contract_nodes_quality_multistart(subnodes, output_labels)
        )
        subtotal = _execute_pair_steps(subnodes, output_labels, selected_steps)
        partial = partial + subtotal
    return partial


def _sparse_working_set_preflight(
    slicing: TensorNetworkSlicingPlan,
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    *,
    max_working_set_bytes: int | None,
    working_set_safety_factor: float,
    working_set_policy: DistributedTNWorkingSetPolicy | None,
    memory_calibration: TNWorkingSetCalibration | None,
    world_size: int,
) -> dict[str, Any]:
    """Build and enforce the rank-local sparse-contraction memory contract."""

    tensor_peak = int(slicing.peak_bytes)
    output = _zero_for_output(nodes, output_labels)
    output_bytes = _tensor_nbytes(output)
    if working_set_policy is not None and memory_calibration is not None:
        raise ValueError(
            "TN working-set policy and memory calibration are mutually exclusive"
        )
    reference = nodes[0].tensor
    if memory_calibration is not None:
        accelerator_name = "cpu"
        if reference.is_cuda:
            device = next(
                (
                    item
                    for item in get_platform_runtime("cuda").discover()
                    if item.index == reference.device.index and item.available
                ),
                None,
            )
            if device is None:
                raise ValueError("TN memory calibration device is unavailable")
            accelerator_name = device.name
        if not memory_calibration.applies_to(
            accelerator_name=accelerator_name,
            complex_bytes=reference.element_size(),
            world_size=world_size,
        ):
            raise ValueError("TN memory calibration scope mismatch")
        workspace = 0
        communication_buffer = 0
        predicted = memory_calibration.apply(tensor_peak)
        allocator_headroom = predicted - tensor_peak
        policy_summary = None
        model = "measured_reserved_memory_calibration"
    elif working_set_policy is None:
        workspace = 0
        communication_buffer = 0
        allocator_headroom = max(
            0, ceil(tensor_peak * (working_set_safety_factor - 1.0))
        )
        policy_summary = None
        model = "calibrated_peak_safety_factor"
    else:
        working_set_policy.validate()
        workspace = ceil(
            tensor_peak * working_set_policy.kernel_workspace_output_multiplier
        )
        communication_buffer = ceil(
            output_bytes * working_set_policy.communication_buffer_output_multiplier
        )
        subtotal = tensor_peak + workspace + communication_buffer
        allocator_headroom = max(
            working_set_policy.minimum_allocator_headroom_bytes,
            ceil(subtotal * working_set_policy.allocator_headroom_fraction),
        )
        policy_summary = asdict(working_set_policy)
        model = "explicit_end_to_end_components"
    if memory_calibration is None:
        predicted = tensor_peak + workspace + communication_buffer + allocator_headroom
    budget_satisfied = max_working_set_bytes is None or predicted <= int(
        max_working_set_bytes
    )
    summary = {
        "model": model,
        "tensor_peak_bytes": tensor_peak,
        "kernel_workspace_bytes": workspace,
        "communication_buffer_bytes": communication_buffer,
        "allocator_headroom_bytes": allocator_headroom,
        "predicted_working_set_bytes": predicted,
        "max_working_set_bytes": max_working_set_bytes,
        "budget_satisfied": budget_satisfied,
        "working_set_policy": policy_summary,
        "memory_calibration_identity": (
            None if memory_calibration is None else memory_calibration.identity
        ),
        "fail_closed": True,
    }
    if not budget_satisfied and max_working_set_bytes is not None:
        raise ValueError(
            "tensor-network working-set preflight rejected execution: "
            f"predicted={predicted} bytes exceeds "
            f"budget={int(max_working_set_bytes)} bytes"
        )
    return summary


@dataclass(frozen=True)
class _SparseContractionOutcome:
    """One sparse-contraction result with the placement it actually used."""

    value: torch.Tensor
    tasks: tuple[DistributedTNSliceTask, ...]
    rank_partial_bytes: Mapping[int, int]
    semantics: str
    scalability_blockers: tuple[str, ...]
    working_set_preflight: dict[str, Any]


class GatheredRankBytes(Mapping[int, int]):
    """Per-rank reduction payload sizes, kept on the device until they are read.

    The contraction gathers these sizes on the device because every rank
    publishes every rank's `local_memory_bytes_by_rank`, and reporting the local
    rank's bytes with zeros for the others would describe a topology that did
    not run. The numbers are evidence about the run rather than input to it, so
    converting them inside the contraction put a device-to-host copy in the
    region a profiler measures: the sample then carried the cost of publishing a
    number nobody had asked for yet. The gathered tensors are kept and converted
    when the summary that publishes them is built, which is after execution.

    The mapping is read-only, and its content is what the dict it replaces
    held, so a caller cannot tell the two apart except by when it synchronizes.
    The conversion is done once and kept, because the sizes it reads were fixed
    by the collective that produced them.
    """

    __slots__ = ("_gathered", "_values")

    def __init__(self, gathered: Sequence[torch.Tensor]) -> None:
        self._gathered = tuple(gathered)
        self._values: tuple[int, ...] | None = None

    def _read(self) -> tuple[int, ...]:
        if self._values is None:
            if self._gathered:
                self._values = tuple(
                    int(value) for value in torch.cat(list(self._gathered)).tolist()
                )
            else:
                self._values = ()
        return self._values

    def __getitem__(self, rank: int) -> int:
        values = self._read()
        index = int(rank)
        if index < 0:
            index += len(values)
        if not 0 <= index < len(values):
            raise KeyError(rank)
        return values[index]

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self._gathered)))

    def __len__(self) -> int:
        return len(self._gathered)

    def __repr__(self) -> str:
        return f"GatheredRankBytes(ranks={len(self._gathered)})"


def _gather_rank_partial_bytes(
    local_bytes: int, *, local_tensor: torch.Tensor
) -> GatheredRankBytes:
    """Collect every rank's reduction payload, not just this rank's.

    `local_memory_bytes_by_rank` is read as per-rank evidence, so reporting the
    local rank's bytes and zeros for the others would describe a topology that
    did not run. The gather rides the same device and dtype family as the
    contracted tensor because a backend may reject a CPU collective (NCCL) or a
    CPU-only process group may reject a CUDA one (gloo).

    The local size is written by a device-side fill rather than by copying a
    one-element host list across, because that copy was a host-to-device
    transfer issued inside the measured region to publish a number the host
    already knew. `fill_` takes the value as an argument and issues no transfer.
    """

    world_size = dist.get_world_size()
    local = torch.zeros(1, dtype=torch.int64, device=local_tensor.device)
    local.fill_(int(local_bytes))
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    return GatheredRankBytes(gathered)


def _distributed_sparse_contraction(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    *,
    world_size: int,
    local_world_size: int,
    context: Any | None,
    max_intermediate_size: int | None,
    max_intermediate_bytes: int | None,
    max_working_set_bytes: int | None,
    working_set_safety_factor: float,
    working_set_policy: DistributedTNWorkingSetPolicy | None,
    memory_calibration: TNWorkingSetCalibration | None,
    plan_cache_path: str | Path | None,
    sliced_labels: Sequence[int] | None,
    max_slices: int | None,
    max_recomputation_factor: float | None,
) -> _SparseContractionOutcome:
    if working_set_safety_factor < 1.0:
        raise ValueError("working_set_safety_factor must be >= 1")
    if max_working_set_bytes is not None:
        if int(max_working_set_bytes) < 1:
            raise ValueError("max_working_set_bytes must be positive")
        if memory_calibration is None:
            workset_limited_intermediate = int(
                int(max_working_set_bytes) / float(working_set_safety_factor)
            )
        else:
            available = (
                int(max_working_set_bytes)
                - memory_calibration.fixed_reserved_overhead_bytes
            )
            workset_limited_intermediate = int(
                available / memory_calibration.recommended_safety_factor
            )
        if workset_limited_intermediate < 1:
            raise ValueError("max_working_set_bytes is too small for the safety factor")
        max_intermediate_bytes = (
            workset_limited_intermediate
            if max_intermediate_bytes is None
            else min(int(max_intermediate_bytes), workset_limited_intermediate)
        )
    cache_key = _persistent_plan_key(
        nodes,
        output_labels,
        max_intermediate_size=max_intermediate_size,
        max_intermediate_bytes=max_intermediate_bytes,
        sliced_labels=sliced_labels,
    )
    if context is not None and context.initialized:
        payload: list[
            tuple[TensorNetworkSlicingPlan, tuple[PairContractionStep, ...]] | None
        ] = [None]
        if context.rank == 0:
            cached = (
                None
                if plan_cache_path is None
                else _load_persistent_plan(plan_cache_path, expected_key=cache_key)
            )
            if cached is not None:
                slicing, shared_steps = cached
            else:
                slicing = _build_slicing_plan(
                    nodes,
                    output_labels,
                    max_intermediate_size=max_intermediate_size,
                    max_intermediate_bytes=max_intermediate_bytes,
                    sliced_labels=sliced_labels,
                )
                representative_tasks = _execution_slice_tasks(
                    slicing,
                    world_size=world_size,
                    local_world_size=local_world_size,
                )
                representative = _slice_nodes(
                    nodes,
                    dict(representative_tasks[0].assignments),
                )
                shared_steps = _contract_nodes_quality_multistart(
                    representative,
                    output_labels,
                )
                if plan_cache_path is not None:
                    _write_persistent_plan(
                        plan_cache_path,
                        cache_key=cache_key,
                        slicing=slicing,
                        steps=shared_steps,
                    )
            payload[0] = (slicing, shared_steps)
        dist.broadcast_object_list(payload, src=0, device=context.device)
        combined_plan = payload[0]
        if combined_plan is None:
            raise RuntimeError("rank-zero tensor-network plan was not broadcast")
        slicing, shared_steps = combined_plan
    else:
        slicing = _build_slicing_plan(
            nodes,
            output_labels,
            max_intermediate_size=max_intermediate_size,
            max_intermediate_bytes=max_intermediate_bytes,
            sliced_labels=sliced_labels,
        )
    slicing.validate_economics(
        max_slices=max_slices,
        max_recomputation_factor=max_recomputation_factor,
    )
    working_set_preflight = _sparse_working_set_preflight(
        slicing,
        nodes,
        output_labels,
        max_working_set_bytes=max_working_set_bytes,
        working_set_safety_factor=working_set_safety_factor,
        working_set_policy=working_set_policy,
        memory_calibration=memory_calibration,
        world_size=world_size,
    )
    tasks = _execution_slice_tasks(
        slicing,
        world_size=world_size,
        local_world_size=local_world_size,
    )
    partial_bytes: Mapping[int, int] = {}
    if context is not None and context.initialized:
        if tasks and shared_steps is None:
            raise RuntimeError(
                "rank-zero tensor-network contraction DAG was not broadcast"
            )
        local_tasks = tuple(task for task in tasks if task.owner_rank == context.rank)
        value = _contract_assigned_tensor_slices(
            nodes,
            output_labels,
            local_tasks,
            steps=shared_steps,
        )
        partial_bytes = _gather_rank_partial_bytes(
            _tensor_nbytes(value), local_tensor=value
        )
        value = _all_reduce_sum_autograd(value)
        # Distinct disjoint slices were contracted on distinct ranks and their
        # sparse outputs were reduced across those ranks, so the returned scalar
        # is genuine rank sharding. `run_distributed_tensor_network` reports a
        # different value because it all-reduces the full state instead.
        semantics = "sharded_across_ranks"
        blockers: tuple[str, ...] = ()
    else:
        partials = []
        # One process contracted every simulated rank's slices, so these sizes
        # are measured rather than collected, and they are already on the host.
        simulated_partial_bytes: dict[int, int] = {}
        for task_rank in range(world_size):
            local_tasks = tuple(task for task in tasks if task.owner_rank == task_rank)
            partial = _contract_assigned_tensor_slices(
                nodes, output_labels, local_tasks
            )
            simulated_partial_bytes[task_rank] = _tensor_nbytes(partial)
            partials.append(partial)
        partial_bytes = simulated_partial_bytes
        value = (
            sum(partials[1:], partials[0])
            if partials
            else _zero_for_output(nodes, output_labels)
        )
        # One process contracted every simulated rank's slices, so the slice
        # ownership is a development mirror rather than a placement.
        semantics = (
            "local_simulated_slice_parallel_sparse_output_reduction"
            if world_size > 1
            else "single_device_sparse_output"
        )
        blockers = (
            ("slice_parallel_local_development_simulator",) if world_size > 1 else ()
        )
    return _SparseContractionOutcome(
        value=value,
        tasks=tasks,
        rank_partial_bytes=partial_bytes,
        semantics=semantics,
        scalability_blockers=blockers,
        working_set_preflight=working_set_preflight,
    )


def _resolve_rank_placement(
    *,
    world_size: int,
    local_world_size: int | None,
    context: TorchDistributedContext | None,
) -> dict[str, Any]:
    """Resolve the placement this TN execution runs at, and where it came from.

    Slice tasks are assigned to nodes through `local_world_size`, so a result
    that reported one placement while the task plan used another would describe
    a topology the collective never ran on. The caller's explicit value wins
    because a caller that initialized its own process group knows its host
    layout, and the process-group environment is the fallback.
    """

    if local_world_size is None:
        if context is not None:
            resolved = context.local_world_size
            source = "process_group_environment"
        else:
            resolved = int(world_size)
            source = "single_process"
    else:
        resolved = int(local_world_size)
        source = "caller"
    if resolved < 1:
        raise ValueError("local_world_size must be a positive rank count")
    if int(world_size) % resolved:
        raise ValueError(
            "local_world_size must divide world_size so every node hosts the "
            f"same number of ranks: local_world_size={resolved} "
            f"world_size={world_size}"
        )
    placement = _rank_placement_summary(context, world_size=int(world_size))
    placement["local_world_size"] = resolved
    placement["local_rank"] = (context.local_rank if context else 0) % resolved
    placement["node_rank"] = (context.rank if context else 0) // resolved
    placement["node_count"] = resolve_node_count(int(world_size), resolved)
    placement["local_world_size_source"] = source
    return placement


def _prepare_distributed_context(
    *,
    world_size: int,
    local_world_size: int | None,
    distributed_executor: str,
    backend: str | None,
    init_method: str,
    rank: int | None,
    local_rank: int | None,
    options: dict[str, Any],
) -> tuple[
    DistributedBackendPolicy, TorchDistributedContext | None, int, dict[str, Any]
]:
    """Resolve backend policy, initialize the requested group, place the ranks."""

    backend_policy = _resolve_backend_policy(options)
    context = None
    if _should_use_torch_distributed(
        distributed_executor,
        backend_policy,
        world_size=world_size,
    ):
        context = init_torch_distributed(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            device=options.get("device"),
            force_initialize=distributed_executor == "torch",
        )
        world_size = context.world_size
        options["device"] = context.device
    placement = _resolve_rank_placement(
        world_size=world_size,
        local_world_size=local_world_size,
        context=context,
    )
    return backend_policy, context, world_size, placement


def distributed_tensor_network_amplitude(
    circuit_or_ir: Any,
    bitstring: int | str | Sequence[int],
    *,
    world_size: int = 1,
    max_intermediate_size: int | None = None,
    max_intermediate_bytes: int | None = None,
    max_working_set_bytes: int | None = None,
    working_set_safety_factor: float = 4.0,
    working_set_policy: DistributedTNWorkingSetPolicy | None = None,
    memory_calibration: TNWorkingSetCalibration | None = None,
    plan_cache_path: str | Path | None = None,
    sliced_labels: Sequence[int] | None = None,
    max_slices: int | None = 4096,
    max_recomputation_factor: float | None = 64.0,
    distributed_executor: str = "auto",
    backend: str | None = None,
    init_method: str = "env://",
    rank: int | None = None,
    local_rank: int | None = None,
    local_world_size: int | None = None,
    **options: Any,
) -> DistributedTensorNetworkAmplitude:
    """Compute one amplitude by distributing internal-edge slice tasks."""

    _, context, world_size, placement = _prepare_distributed_context(
        world_size=world_size,
        local_world_size=local_world_size,
        distributed_executor=distributed_executor,
        backend=backend,
        init_method=init_method,
        rank=rank,
        local_rank=local_rank,
        options=options,
    )

    plan = build_tensor_network(
        circuit_or_ir,
        bsz=int(options.get("bsz", 1)),
        device=options.get("device", "cpu"),
        dtype=options.get("dtype"),
    )
    nodes, output_labels = _amplitude_projection(plan, bitstring)
    outcome = _distributed_sparse_contraction(
        nodes,
        output_labels,
        world_size=world_size,
        local_world_size=placement["local_world_size"],
        context=context,
        max_intermediate_size=max_intermediate_size,
        max_intermediate_bytes=max_intermediate_bytes,
        max_working_set_bytes=max_working_set_bytes,
        working_set_safety_factor=working_set_safety_factor,
        working_set_policy=working_set_policy,
        memory_calibration=memory_calibration,
        plan_cache_path=plan_cache_path,
        sliced_labels=sliced_labels,
        max_slices=max_slices,
        max_recomputation_factor=max_recomputation_factor,
    )
    return DistributedTensorNetworkAmplitude(
        value=outcome.value.reshape(plan.bsz),
        world_size=world_size,
        tasks=outcome.tasks,
        rank_partial_bytes=outcome.rank_partial_bytes,
        distribution_semantics=outcome.semantics,
        working_set_preflight=outcome.working_set_preflight,
        rank_placement=placement,
        claim_evidence_type=_context_claim_evidence_type(context),
        scalability_blockers=outcome.scalability_blockers,
    )


def distributed_tensor_network_expectation(
    circuit_or_ir: Any,
    *,
    x: Sequence[int] | None = None,
    y: Sequence[int] | None = None,
    z: Sequence[int] | None = None,
    world_size: int = 1,
    max_intermediate_size: int | None = None,
    max_intermediate_bytes: int | None = None,
    max_working_set_bytes: int | None = None,
    working_set_safety_factor: float = 4.0,
    working_set_policy: DistributedTNWorkingSetPolicy | None = None,
    memory_calibration: TNWorkingSetCalibration | None = None,
    plan_cache_path: str | Path | None = None,
    sliced_labels: Sequence[int] | None = None,
    max_slices: int | None = 4096,
    max_recomputation_factor: float | None = 64.0,
    distributed_executor: str = "auto",
    backend: str | None = None,
    init_method: str = "env://",
    rank: int | None = None,
    local_rank: int | None = None,
    local_world_size: int | None = None,
    **options: Any,
) -> DistributedTensorNetworkExpectation:
    """Compute one Pauli-product expectation without materializing the state."""

    _, context, world_size, placement = _prepare_distributed_context(
        world_size=world_size,
        local_world_size=local_world_size,
        distributed_executor=distributed_executor,
        backend=backend,
        init_method=init_method,
        rank=rank,
        local_rank=local_rank,
        options=options,
    )

    ket_plan = build_tensor_network(
        circuit_or_ir,
        bsz=int(options.get("bsz", 1)),
        device=options.get("device", "cpu"),
        dtype=options.get("dtype"),
    )
    plan = build_tensor_network_expectation(ket_plan, x=x, y=y, z=z)
    outcome = _distributed_sparse_contraction(
        plan.nodes,
        plan.output_labels,
        world_size=world_size,
        local_world_size=placement["local_world_size"],
        context=context,
        max_intermediate_size=max_intermediate_size,
        max_intermediate_bytes=max_intermediate_bytes,
        max_working_set_bytes=max_working_set_bytes,
        working_set_safety_factor=working_set_safety_factor,
        working_set_policy=working_set_policy,
        memory_calibration=memory_calibration,
        plan_cache_path=plan_cache_path,
        sliced_labels=sliced_labels,
        max_slices=max_slices,
        max_recomputation_factor=max_recomputation_factor,
    )
    return DistributedTensorNetworkExpectation(
        value=torch.real(outcome.value.reshape(plan.bsz)),
        world_size=world_size,
        tasks=outcome.tasks,
        rank_partial_bytes=outcome.rank_partial_bytes,
        observable_qubits=plan.observable_qubits,
        distribution_semantics=outcome.semantics,
        working_set_preflight=outcome.working_set_preflight,
        rank_placement=placement,
        claim_evidence_type=_context_claim_evidence_type(context),
        scalability_blockers=outcome.scalability_blockers,
    )


def distributed_tensor_network_amplitudes(
    circuit_or_ir: Any,
    bitstrings: Sequence[int | str | Sequence[int]],
    *,
    world_size: int = 1,
    max_intermediate_size: int | None = None,
    max_intermediate_bytes: int | None = None,
    max_working_set_bytes: int | None = None,
    working_set_safety_factor: float = 4.0,
    working_set_policy: DistributedTNWorkingSetPolicy | None = None,
    memory_calibration: TNWorkingSetCalibration | None = None,
    plan_cache_path: str | Path | None = None,
    sliced_labels: Sequence[int] | None = None,
    max_slices: int | None = 4096,
    max_recomputation_factor: float | None = 64.0,
    distributed_executor: str = "auto",
    backend: str | None = None,
    init_method: str = "env://",
    rank: int | None = None,
    local_rank: int | None = None,
    local_world_size: int | None = None,
    **options: Any,
) -> DistributedTensorNetworkAmplitudes:
    """Compute a small amplitude batch with shared planning and contraction."""

    _, context, world_size, placement = _prepare_distributed_context(
        world_size=world_size,
        local_world_size=local_world_size,
        distributed_executor=distributed_executor,
        backend=backend,
        init_method=init_method,
        rank=rank,
        local_rank=local_rank,
        options=options,
    )

    plan = build_tensor_network(
        circuit_or_ir,
        bsz=int(options.get("bsz", 1)),
        device=options.get("device", "cpu"),
        dtype=options.get("dtype"),
    )
    nodes, output_labels = _amplitude_batch_projection(plan, bitstrings)
    outcome = _distributed_sparse_contraction(
        nodes,
        output_labels,
        world_size=world_size,
        local_world_size=placement["local_world_size"],
        context=context,
        max_intermediate_size=max_intermediate_size,
        max_intermediate_bytes=max_intermediate_bytes,
        max_working_set_bytes=max_working_set_bytes,
        working_set_safety_factor=working_set_safety_factor,
        working_set_policy=working_set_policy,
        memory_calibration=memory_calibration,
        plan_cache_path=plan_cache_path,
        sliced_labels=sliced_labels,
        max_slices=max_slices,
        max_recomputation_factor=max_recomputation_factor,
    )
    return DistributedTensorNetworkAmplitudes(
        values=outcome.value.reshape(plan.bsz, len(bitstrings)),
        world_size=world_size,
        tasks=outcome.tasks,
        rank_partial_bytes=outcome.rank_partial_bytes,
        target_count=len(bitstrings),
        distribution_semantics=outcome.semantics,
        working_set_preflight=outcome.working_set_preflight,
        rank_placement=placement,
        claim_evidence_type=_context_claim_evidence_type(context),
        scalability_blockers=outcome.scalability_blockers,
    )


def distributed_tensor_network_expectations(
    circuit_or_ir: Any,
    observables: Sequence[Mapping[str, Sequence[int]]],
    *,
    world_size: int = 1,
    max_intermediate_size: int | None = None,
    max_intermediate_bytes: int | None = None,
    max_working_set_bytes: int | None = None,
    working_set_safety_factor: float = 4.0,
    working_set_policy: DistributedTNWorkingSetPolicy | None = None,
    memory_calibration: TNWorkingSetCalibration | None = None,
    plan_cache_path: str | Path | None = None,
    sliced_labels: Sequence[int] | None = None,
    max_slices: int | None = 4096,
    max_recomputation_factor: float | None = 64.0,
    distributed_executor: str = "auto",
    backend: str | None = None,
    init_method: str = "env://",
    rank: int | None = None,
    local_rank: int | None = None,
    local_world_size: int | None = None,
    **options: Any,
) -> DistributedTensorNetworkExpectations:
    """Compute several Pauli products through one distributed contraction."""

    _, context, world_size, placement = _prepare_distributed_context(
        world_size=world_size,
        local_world_size=local_world_size,
        distributed_executor=distributed_executor,
        backend=backend,
        init_method=init_method,
        rank=rank,
        local_rank=local_rank,
        options=options,
    )

    plan = build_tensor_network(
        circuit_or_ir,
        bsz=int(options.get("bsz", 1)),
        device=options.get("device", "cpu"),
        dtype=options.get("dtype"),
    )
    nodes, output_labels = _expectation_batch_projection(plan, observables)
    outcome = _distributed_sparse_contraction(
        nodes,
        output_labels,
        world_size=world_size,
        local_world_size=placement["local_world_size"],
        context=context,
        max_intermediate_size=max_intermediate_size,
        max_intermediate_bytes=max_intermediate_bytes,
        max_working_set_bytes=max_working_set_bytes,
        working_set_safety_factor=working_set_safety_factor,
        working_set_policy=working_set_policy,
        memory_calibration=memory_calibration,
        plan_cache_path=plan_cache_path,
        sliced_labels=sliced_labels,
        max_slices=max_slices,
        max_recomputation_factor=max_recomputation_factor,
    )
    return DistributedTensorNetworkExpectations(
        values=torch.real(outcome.value.reshape(plan.bsz, len(observables))),
        world_size=world_size,
        tasks=outcome.tasks,
        rank_partial_bytes=outcome.rank_partial_bytes,
        observable_count=len(observables),
        distribution_semantics=outcome.semantics,
        working_set_preflight=outcome.working_set_preflight,
        rank_placement=placement,
        claim_evidence_type=_context_claim_evidence_type(context),
        scalability_blockers=outcome.scalability_blockers,
    )


def run_distributed_tensor_network(
    circuit_or_ir: Any,
    *,
    world_size: int = 1,
    max_intermediate_size: int | None = None,
    sliced_labels: Sequence[int] | None = None,
    distributed_executor: str = "auto",
    backend: str | None = None,
    init_method: str = "env://",
    rank: int | None = None,
    local_rank: int | None = None,
    local_world_size: int | None = None,
    **options: Any,
) -> DistributedTensorNetworkState:
    """Run tensor-network contraction with torch.distributed slice parallelism."""

    backend_policy, context, world_size, placement = _prepare_distributed_context(
        world_size=world_size,
        local_world_size=local_world_size,
        distributed_executor=distributed_executor,
        backend=backend,
        init_method=init_method,
        rank=rank,
        local_rank=local_rank,
        options=options,
    )

    plan = build_tensor_network(
        circuit_or_ir,
        bsz=int(options.get("bsz", 1)),
        device=options.get("device", "cpu"),
        dtype=options.get("dtype"),
    )
    slicing = plan.slicing_plan(
        max_intermediate_size=max_intermediate_size, sliced_labels=sliced_labels
    )
    tasks = _execution_slice_tasks(
        slicing,
        world_size=world_size,
        local_world_size=placement["local_world_size"],
    )
    state_cache = None
    local_simulation = False
    rank_partial_bytes: Mapping[int, int] = {}
    if context is not None and context.initialized:
        local_tasks = tuple(task for task in tasks if task.owner_rank == context.rank)
        partial = _contract_assigned_tensor_slices(
            plan.nodes, plan.output_labels, local_tasks
        )
        rank_partial_bytes = _gather_rank_partial_bytes(
            _tensor_nbytes(partial), local_tensor=partial
        )
        partial = _all_reduce_sum_autograd(partial)
        state_cache = partial.reshape(plan.bsz, 2**plan.n_qubits)
    elif world_size > 1 and backend_policy.torch_backend == "local_tensor":
        partials = []
        # One process contracted every simulated rank's slices, so these sizes
        # are measured rather than collected, and they are already on the host.
        simulated_partial_bytes: dict[int, int] = {}
        for task_rank in range(world_size):
            local_tasks = tuple(task for task in tasks if task.owner_rank == task_rank)
            partial = _contract_assigned_tensor_slices(
                plan.nodes, plan.output_labels, local_tasks
            )
            simulated_partial_bytes[task_rank] = _tensor_nbytes(partial)
            partials.append(partial)
        rank_partial_bytes = simulated_partial_bytes
        if partials:
            total = partials[0]
            for partial in partials[1:]:
                total = total + partial
        else:
            total = _zero_for_output(plan.nodes, plan.output_labels)
        state_cache = total.reshape(plan.bsz, 2**plan.n_qubits)
        local_simulation = True
    local = run_tensor_network(
        circuit_or_ir,
        contraction_strategy="sliced",
        max_intermediate_size=max_intermediate_size,
        sliced_labels=slicing.sliced_labels,
        **options,
    )
    if state_cache is not None:
        local._state_cache = state_cache
    return DistributedTensorNetworkState(
        local,
        world_size=world_size,
        tasks=tasks,
        context=context,
        state_cache=state_cache,
        backend_policy=backend_policy,
        local_simulation=local_simulation,
        rank_partial_bytes=rank_partial_bytes,
        rank_placement=placement,
    )


__all__ = [
    "distributed_tensor_network_amplitude",
    "distributed_tensor_network_amplitudes",
    "distributed_tensor_network_expectation",
    "distributed_tensor_network_expectations",
    "run_distributed_tensor_network",
]
