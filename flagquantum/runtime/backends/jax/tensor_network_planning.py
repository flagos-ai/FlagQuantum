# ruff: noqa: F401, F821
"""Tensor-network parameter gradients and representation planning."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace
from itertools import product
from typing import Any, Callable, Mapping, Sequence

from ....core.ir import CircuitIR, ensure_circuit_ir
from ...distributed.backend_policy import (
    DistributedBackendPolicy,
    resolve_distributed_backend_policy,
)
from .common import communication_tier as _communication_tier
from .common import env_int as _env_int
from .common import node_count as _node_count
from .common import product_int as _product
from .common import rank_for_wire as _rank_for_wire
from .common import split_contiguous as _split_contiguous
from .planning_core import JAXDistributedQuantumPlan
from .release_policy import (
    attach_evidence_contract as _attach_distributed_evidence_contract,
)
from .release_policy import (
    attach_mps_backward_readiness as _attach_mps_backward_readiness,
)
from .release_policy import (
    attach_statevector_claimability as _attach_statevector_claimability,
)


def jax_sliced_tensor_network_parameter_value_and_grad(
    circuit_builder: Callable[[Any], Any],
    parameters: Any,
    *,
    n_wires: int,
    world_size: int | None = None,
    local_world_size: int | None = None,
    bsz: int = 1,
    complex_bytes: int | None = None,
    dtype: Any | None = None,
    max_intermediate_size: int | None = None,
    sliced_labels: Sequence[int] | None = None,
    observable: str = "z_sum",
    observable_wires: Sequence[int] | None = None,
    hamiltonian_terms: Sequence[tuple[float, Sequence[tuple[int, str]]]] = (),
    distributed_backend_policy: DistributedBackendPolicy | None = None,
    distributed_profile: str | None = None,
    jax_backend: str | None = None,
    torch_backend: str | None = None,
    compute_backend: str = "auto",
    collective_backend: str = "local_simulated",
    jit: bool = False,
) -> JAXSlicedTensorNetworkParameterGradientResult:
    """Reverse-mode sliced TN value/gradient for a parameterized circuit builder.

    This path differentiates the JAX gate matrices produced from FlagQuantum IR
    parameters.  It contracts observable tensor networks directly, instead of
    materializing a dense statevector as the loss path.
    """

    torch = _require_torch()
    jax, jnp = _require_jax()
    policy = _resolve_policy(
        distributed_backend_policy=distributed_backend_policy,
        distributed_profile=distributed_profile,
        jax_backend=jax_backend,
        torch_backend=torch_backend,
    )
    resolved_collective_backend = _resolve_collective_backend(
        collective_backend, policy
    )
    resolved_compute_backend = _resolve_tn_compute_backend(compute_backend, policy)
    if (
        resolved_compute_backend == "pmap"
        and resolved_collective_backend == "local_simulated"
    ):
        resolved_collective_backend = "pmap"
    resolved_world_size = _resolve_world_size(world_size, policy)
    resolved_local_world_size = _resolve_local_world_size(
        local_world_size,
        world_size=resolved_world_size,
        policy=policy,
    )
    if complex_bytes is None:
        complex_bytes = 16 if dtype == torch.complex128 else 8
    static_parameters = _torch_parameters_for_static_build(
        parameters, complex_bytes=complex_bytes
    )
    parameter_shape = tuple(int(dim) for dim in static_parameters.shape)
    example_circuit = circuit_builder(static_parameters)
    static_plan = _static_expectation_plan_for_parameterized_tn(
        example_circuit,
        bsz=bsz,
        complex_bytes=complex_bytes,
        observable=observable,
        observable_wires=observable_wires,
        hamiltonian_terms=hamiltonian_terms,
    )
    slicing = static_plan.slicing_plan(
        max_intermediate_size=max_intermediate_size, sliced_labels=sliced_labels
    )
    tasks = _tn_tasks(slicing.sliced_labels, slicing.slice_shape, resolved_world_size)
    active_ranks = tuple(sorted({int(rank) for rank, _ in tasks}))
    if resolved_world_size > 1 and (int(slicing.n_slices) < 2 or len(active_ranks) < 2):
        raise RuntimeError(
            "JAX parameterized sliced tensor-network reverse mode requires at least two slice tasks assigned to multiple ranks."
        )
    jax_parameters = _jax_parameter_array_from_input(
        parameters, complex_bytes=complex_bytes
    )
    compute_dtype = "complex128" if int(complex_bytes) == 16 else "complex64"

    def _loss(parameter_array: Any) -> Any:
        from .kernel import _JAXParameterProxy, _set_active_jax_compute_dtype

        previous_dtype = _set_active_jax_compute_dtype(compute_dtype)
        try:
            parameter_array = jnp.asarray(
                parameter_array, dtype=_jax_real_dtype(complex_bytes)
            )
            circuit = circuit_builder(_JAXParameterProxy(parameter_array))
            return _jax_parameterized_tn_observable_loss(
                circuit,
                n_wires=int(n_wires),
                bsz=static_plan.bsz,
                complex_bytes=complex_bytes,
                tasks=tasks,
                output_labels=static_plan.output_labels,
                world_size=resolved_world_size,
                compute_backend=resolved_compute_backend,
                collective_backend=resolved_collective_backend,
                observable=observable,
                observable_wires=observable_wires,
                hamiltonian_terms=hamiltonian_terms,
            )
        finally:
            _set_active_jax_compute_dtype(previous_dtype)

    value_and_grad = jax.value_and_grad(_loss)
    if jit:
        value_and_grad = jax.jit(value_and_grad)
    value, gradient = value_and_grad(jax_parameters)
    jax_plan = plan_jax_distributed_quantum_backend(
        example_circuit,
        mode="tensor_network",
        world_size=resolved_world_size,
        local_world_size=resolved_local_world_size,
        bsz=static_plan.bsz,
        complex_bytes=complex_bytes,
        max_intermediate_size=max_intermediate_size,
        sliced_labels=slicing.sliced_labels,
        distributed_backend_policy=policy,
    )
    from .kernel import _JAXParameterProxy, _set_active_jax_compute_dtype

    previous_dtype = _set_active_jax_compute_dtype(compute_dtype)
    try:
        summary_circuit = circuit_builder(_JAXParameterProxy(jax_parameters))
        summary_nodes, _ = _jax_parameterized_tn_expectation_nodes(
            summary_circuit,
            int(n_wires),
            bsz=static_plan.bsz,
            complex_bytes=complex_bytes,
            ops={},
        )
    finally:
        _set_active_jax_compute_dtype(previous_dtype)
    _partials, _reduced, compute_execution, collective_execution = (
        _jax_contract_tensor_slices_by_backend(
            summary_nodes,
            static_plan.output_labels,
            tasks,
            world_size=resolved_world_size,
            compute_backend=resolved_compute_backend,
            collective_backend=resolved_collective_backend,
        )
    )
    return JAXSlicedTensorNetworkParameterGradientResult(
        value=value,
        gradient=gradient,
        slicing=slicing,
        jax_plan=jax_plan,
        backend_policy=policy,
        n_wires=int(n_wires),
        bsz=static_plan.bsz,
        complex_bytes=complex_bytes,
        parameter_shape=parameter_shape,
        local_world_size=resolved_local_world_size,
        node_count=_node_count(resolved_world_size, resolved_local_world_size),
        compute_backend=resolved_compute_backend,
        compute_execution=compute_execution,
        collective_backend=resolved_collective_backend,
        collective_execution=collective_execution,
        observable=str(observable),
    )


def _tn_tasks(
    sliced_labels: Sequence[int],
    slice_shape: Sequence[int],
    world_size: int,
) -> tuple[tuple[int, tuple[tuple[int, int], ...]], ...]:
    labels = tuple(int(label) for label in sliced_labels)
    ranges = [range(int(size)) for size in slice_shape]
    tasks = []
    for task_index, values in enumerate(product(*ranges) if ranges else [()]):
        tasks.append(
            (
                task_index % max(1, int(world_size)),
                tuple(zip(labels, tuple(int(value) for value in values))),
            )
        )
    return tuple(tasks)


def _tensor_network_plan(
    ir: CircuitIR,
    *,
    policy: DistributedBackendPolicy,
    world_size: int,
    local_world_size: int,
    bsz: int,
    complex_bytes: int,
    max_intermediate_size: int | None,
    sliced_labels: Sequence[int] | None,
) -> JAXDistributedQuantumPlan:
    from ....simulation.tensor import build_tensor_network

    contraction_plan = build_tensor_network(ir, bsz=bsz, device="cpu")
    slicing = contraction_plan.slicing_plan(
        max_intermediate_size=max_intermediate_size,
        sliced_labels=sliced_labels,
    )
    tasks = _tn_tasks(slicing.sliced_labels, slicing.slice_shape, world_size)
    tasks_by_rank = {
        rank: sum(1 for task_rank, _ in tasks if task_rank == rank)
        for rank in range(world_size)
    }
    output_bytes = int(bsz) * (2 ** int(ir.n_wires)) * int(complex_bytes)
    local_memory = tuple(
        int(tasks_by_rank[rank] * output_bytes) for rank in range(world_size)
    )
    rank_ownership = tuple(
        {
            "rank": rank,
            "state_partition": "tensor_network_slices",
            "slice_task_count": tasks_by_rank[rank],
            "local_memory_bytes": local_memory[rank],
        }
        for rank in range(world_size)
    )
    blockers = (
        "jax_pmap_tensor_network_slice_executor_pending",
        "rank_local_jax_kernel_is_not_capacity_scaling",
        "tn_output_state_reduction_still_materializes_full_output",
    )
    gradient_blockers = ("jax_sharded_tensor_network_reverse_contraction_pending",)
    return JAXDistributedQuantumPlan(
        mode="tensor_network",
        n_wires=ir.n_wires,
        world_size=world_size,
        local_world_size=local_world_size,
        node_count=_node_count(world_size, local_world_size),
        backend_policy=policy,
        rank_ownership=rank_ownership,
        communication_tiers={
            "model": "jax_pmap_tensor_network_slice_reduce_planned",
            "collective": "psum",
            "reduction_tensor_bytes": output_bytes,
            "inter_node_collective_possible": bool(
                _node_count(world_size, local_world_size) > 1 and world_size > 1
            ),
            "note": "Exact physical bytes depend on XLA collective lowering.",
        },
        local_memory_bytes_by_rank=local_memory,
        blockers=blockers,
        gradient_blockers=gradient_blockers,
        task_summary={
            "sliced_labels": slicing.sliced_labels,
            "slice_shape": slicing.slice_shape,
            "slice_task_count": len(tasks),
            "tasks_by_rank": tasks_by_rank,
            "max_intermediate_size": max_intermediate_size,
        },
    )
