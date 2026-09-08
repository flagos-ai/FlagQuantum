"""Distributed tensor-network reduction execution.

This module provides FlagQuantum-native distributed result objects without
depending on an external graph or tensor-network package. Development backends
must preserve the same rank ownership and communication semantics as production
backends, while production backends are expected to execute one logical workload
across rank-local shards instead of replicating the full circuit per rank.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import asdict
from math import ceil
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from ....providers.platform import resolve_platform_device
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
from ....version import __version__
from ...distributed.backend_policy import (
    DistributedBackendPolicy,
    _resolve_backend_policy,
    _should_use_torch_distributed,
)
from ...distributed.context import (
    TorchDistributedContext,
    init_torch_distributed,
)
from ...planner.tn_calibration import TNWorkingSetCalibration
from .joint_planning import (
    DistributedTNWorkingSetPolicy,
)
from .sliced_tasks import DistributedTNSliceTask, plan_distributed_tn_slice_tasks
from .state import (
    DistributedTensorNetworkAmplitude,
    DistributedTensorNetworkAmplitudes,
    DistributedTensorNetworkExpectation,
    DistributedTensorNetworkExpectations,
    DistributedTensorNetworkState,
)

_PERSISTENT_PLAN_SCHEMA = "flagquantum.distributed_tn_plan.v1"


def _execution_slice_tasks(
    slicing: TensorNetworkSlicingPlan,
    *,
    world_size: int,
    context: TorchDistributedContext | None,
) -> tuple[DistributedTNSliceTask, ...]:
    """Use the canonical topology-aware task plan for TN execution."""

    local_world_size = context.local_world_size if context is not None else world_size
    return plan_distributed_tn_slice_tasks(
        slicing,
        world_size=world_size,
        local_world_size=local_world_size,
    ).tasks


@contextmanager
def _plan_file_lock(path: Path, *, exclusive: bool):
    """Serialize cache writers while allowing concurrent readers."""

    import fcntl

    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(
            handle.fileno(),
            fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH,
        )
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _persistent_plan_key(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    *,
    max_intermediate_size: int | None,
    max_intermediate_bytes: int | None,
    sliced_labels: Sequence[int] | None,
) -> str:
    payload = {
        "schema": _PERSISTENT_PLAN_SCHEMA,
        "nodes": [
            {
                "name": node.name,
                "labels": list(node.labels),
                "shape": list(node.tensor.shape),
                "element_size": node.tensor.element_size(),
            }
            for node in nodes
        ],
        "output_labels": list(output_labels),
        "max_intermediate_size": max_intermediate_size,
        "max_intermediate_bytes": max_intermediate_bytes,
        "sliced_labels": None if sliced_labels is None else list(sliced_labels),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _pair_contraction_step_from_payload(
    payload: Mapping[str, Any],
) -> PairContractionStep:
    """Restore tuple-valued fields erased by JSON serialization."""

    return PairContractionStep(
        **{
            **payload,
            "left_labels": tuple(payload["left_labels"]),
            "right_labels": tuple(payload["right_labels"]),
            "output_labels": tuple(payload["output_labels"]),
            "output_shape": tuple(payload["output_shape"]),
        }
    )


def _load_persistent_plan(
    path: str | Path,
    *,
    expected_key: str,
) -> tuple[TensorNetworkSlicingPlan, tuple[PairContractionStep, ...]] | None:
    target = Path(path)
    if not target.is_file():
        return None
    try:
        with _plan_file_lock(target, exclusive=False):
            payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, KeyError):
        return None
    if (
        payload.get("schema_version") != _PERSISTENT_PLAN_SCHEMA
        or payload.get("cache_key") != expected_key
        or payload.get("flagquantum_version") != __version__
        or payload.get("torch_version") != torch.__version__
    ):
        return None
    try:
        slicing_payload = dict(payload["slicing"])
        for field in ("sliced_labels", "slice_shape"):
            slicing_payload[field] = tuple(slicing_payload[field])
        slicing_payload["contraction_path"] = tuple(
            _pair_contraction_step_from_payload(item)
            for item in slicing_payload["contraction_path"]
        )
        slicing = TensorNetworkSlicingPlan(**slicing_payload)
        steps = tuple(
            _pair_contraction_step_from_payload(item) for item in payload["steps"]
        )
    except (TypeError, ValueError, KeyError):
        return None
    return slicing, steps


def _write_persistent_plan(
    path: str | Path,
    *,
    cache_key: str,
    slicing: TensorNetworkSlicingPlan,
    steps: Sequence[PairContractionStep],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": _PERSISTENT_PLAN_SCHEMA,
        "flagquantum_version": __version__,
        "torch_version": torch.__version__,
        "cache_key": cache_key,
        "slicing": asdict(slicing),
        "steps": [asdict(step) for step in steps],
    }
    with _plan_file_lock(target, exclusive=True):
        temporary = target.with_suffix(
            target.suffix + f".{hashlib.sha256(cache_key.encode()).hexdigest()[:8]}.tmp"
        )
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)


class _DistributedAllReduceSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor) -> torch.Tensor:
        out = tensor.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        return (grad_output,)


def _all_reduce_sum_autograd(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.requires_grad:
        out = tensor.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out
    return _DistributedAllReduceSum.apply(tensor)


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
    accelerator_name = (
        resolve_platform_device(reference.device).name if reference.is_cuda else "cpu"
    )
    if memory_calibration is not None:
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
    if not budget_satisfied:
        raise ValueError(
            "tensor-network working-set preflight rejected execution: "
            f"predicted={predicted} bytes exceeds "
            f"budget={int(max_working_set_bytes)} bytes"
        )
    return summary


def _distributed_sparse_contraction(
    nodes: Sequence[TensorNetworkNode],
    output_labels: Sequence[int],
    *,
    world_size: int,
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
) -> tuple[
    torch.Tensor,
    tuple[DistributedTNSliceTask, ...],
    dict[int, int],
    str,
    dict[str, Any],
]:
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
        payload = [None]
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
                    context=context,
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
        context=context,
    )
    partial_bytes: dict[int, int] = {}
    if context is not None and context.initialized:
        if tasks and shared_steps is None:
            raise RuntimeError(
                "rank-zero tensor-network contraction DAG was not broadcast"
            )
        local_tasks = tuple(
            task for task in tasks if task.owner_rank == context.rank
        )
        value = _contract_assigned_tensor_slices(
            nodes,
            output_labels,
            local_tasks,
            steps=shared_steps,
        )
        partial_bytes[context.rank] = _tensor_nbytes(value)
        value = _all_reduce_sum_autograd(value)
        semantics = "slice_parallel_sparse_output_all_reduce"
    else:
        partials = []
        for task_rank in range(world_size):
            local_tasks = tuple(
                task for task in tasks if task.owner_rank == task_rank
            )
            partial = _contract_assigned_tensor_slices(
                nodes, output_labels, local_tasks
            )
            partial_bytes[task_rank] = _tensor_nbytes(partial)
            partials.append(partial)
        value = (
            sum(partials[1:], partials[0])
            if partials
            else _zero_for_output(nodes, output_labels)
        )
        semantics = (
            "local_simulated_slice_parallel_sparse_output_reduction"
            if world_size > 1
            else "single_device_sparse_output"
        )
    return value, tasks, partial_bytes, semantics, working_set_preflight


def _prepare_distributed_context(
    *,
    world_size: int,
    distributed_executor: str,
    backend: str | None,
    init_method: str,
    rank: int | None,
    local_rank: int | None,
    options: dict[str, Any],
) -> tuple[DistributedBackendPolicy, TorchDistributedContext | None, int]:
    """Resolve backend policy and initialize the requested process group."""

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
    return backend_policy, context, world_size


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
    **options: Any,
) -> DistributedTensorNetworkAmplitude:
    """Compute one amplitude by distributing internal-edge slice tasks."""

    _, context, world_size = _prepare_distributed_context(
        world_size=world_size,
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
    value, tasks, partial_bytes, semantics, preflight = _distributed_sparse_contraction(
        nodes,
        output_labels,
        world_size=world_size,
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
        value=value.reshape(plan.bsz),
        world_size=world_size,
        tasks=tasks,
        rank_partial_bytes=partial_bytes,
        distribution_semantics=semantics,
        working_set_preflight=preflight,
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
    **options: Any,
) -> DistributedTensorNetworkExpectation:
    """Compute one Pauli-product expectation without materializing the state."""

    _, context, world_size = _prepare_distributed_context(
        world_size=world_size,
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
    value, tasks, partial_bytes, semantics, preflight = _distributed_sparse_contraction(
        plan.nodes,
        plan.output_labels,
        world_size=world_size,
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
        value=torch.real(value.reshape(plan.bsz)),
        world_size=world_size,
        tasks=tasks,
        rank_partial_bytes=partial_bytes,
        observable_wires=plan.observable_wires,
        distribution_semantics=semantics,
        working_set_preflight=preflight,
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
    **options: Any,
) -> DistributedTensorNetworkAmplitudes:
    """Compute a small amplitude batch with shared planning and contraction."""

    _, context, world_size = _prepare_distributed_context(
        world_size=world_size,
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
    value, tasks, partial_bytes, semantics, preflight = _distributed_sparse_contraction(
        nodes,
        output_labels,
        world_size=world_size,
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
        values=value.reshape(plan.bsz, len(bitstrings)),
        world_size=world_size,
        tasks=tasks,
        rank_partial_bytes=partial_bytes,
        target_count=len(bitstrings),
        distribution_semantics=semantics,
        working_set_preflight=preflight,
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
    **options: Any,
) -> DistributedTensorNetworkExpectations:
    """Compute several Pauli products through one distributed contraction."""

    _, context, world_size = _prepare_distributed_context(
        world_size=world_size,
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
    value, tasks, partial_bytes, semantics, preflight = _distributed_sparse_contraction(
        nodes,
        output_labels,
        world_size=world_size,
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
        values=torch.real(value.reshape(plan.bsz, len(observables))),
        world_size=world_size,
        tasks=tasks,
        rank_partial_bytes=partial_bytes,
        observable_count=len(observables),
        distribution_semantics=semantics,
        working_set_preflight=preflight,
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
    **options: Any,
) -> DistributedTensorNetworkState:
    """Run tensor-network contraction with torch.distributed slice parallelism."""

    backend_policy, context, world_size = _prepare_distributed_context(
        world_size=world_size,
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
        context=context,
    )
    state_cache = None
    local_simulation = False
    rank_partial_bytes: dict[int, int] = {}
    if context is not None and context.initialized:
        local_tasks = tuple(
            task for task in tasks if task.owner_rank == context.rank
        )
        partial = _contract_assigned_tensor_slices(
            plan.nodes, plan.output_labels, local_tasks
        )
        rank_partial_bytes[context.rank] = _tensor_nbytes(partial)
        partial = _all_reduce_sum_autograd(partial)
        state_cache = partial.reshape(plan.bsz, 2**plan.n_wires)
    elif world_size > 1 and backend_policy.torch_backend == "local_tensor":
        partials = []
        for task_rank in range(world_size):
            local_tasks = tuple(
                task for task in tasks if task.owner_rank == task_rank
            )
            partial = _contract_assigned_tensor_slices(
                plan.nodes, plan.output_labels, local_tasks
            )
            rank_partial_bytes[task_rank] = _tensor_nbytes(partial)
            partials.append(partial)
        if partials:
            total = partials[0]
            for partial in partials[1:]:
                total = total + partial
        else:
            total = _zero_for_output(plan.nodes, plan.output_labels)
        state_cache = total.reshape(plan.bsz, 2**plan.n_wires)
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
    )


__all__ = [
    "distributed_tensor_network_amplitude",
    "distributed_tensor_network_amplitudes",
    "distributed_tensor_network_expectation",
    "distributed_tensor_network_expectations",
    "run_distributed_tensor_network",
]
