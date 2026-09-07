"""Sharded statevector execution and parameter-gradient entrypoints."""

from __future__ import annotations

from typing import Any, Callable, Sequence

from ....distributed.backend_policy import DistributedBackendPolicy
from ..array_conversions import (
    _apply_gate_to_jax_shards,
    _gate_matrix_as_jax,
    _jax_parameter_array_from_input,
    _parameterized_gate_matrix_as_jax,
    _torch_parameters_for_static_build,
)
from ..backend_dispatch import plan_jax_distributed_quantum_backend
from ..planning_core import _as_ir
from ..runtime_environment import (
    _jax_complex_dtype,
    _jax_device_count_summary,
    _jax_real_dtype,
    _jnp_device_put,
    _require_jax,
    _require_jax_production_backward_ready,
    _require_torch,
    _resolve_jax_backward_backend,
    _resolve_jax_device,
    _resolve_local_world_size,
    _resolve_policy,
    _resolve_world_size,
    _torch_complex_dtype,
)
from ..statevector_gradient_records import (
    JAXShardedStatevectorParameterGradientResult,
    _initialize_jax_statevector_shard,
)
from ..statevector_kernels import (
    _jax_pmap_statevector_parameter_loss,
    _jax_shard_map_statevector_parameter_loss,
    _jax_sharded_statevector_loss_from_shards,
    _statevector_pmap_backward_blockers,
    _statevector_shard_map_backward_blockers,
)
from .records import JAXShardedStatevectorResult, JAXStatevectorShardState


def run_jax_sharded_statevector(
    circuit_or_ir: Any,
    *,
    world_size: int | None = None,
    local_world_size: int | None = None,
    bsz: int = 1,
    complex_bytes: int = 8,
    distributed_backend_policy: DistributedBackendPolicy | None = None,
    distributed_profile: str | None = None,
    jax_backend: str | None = None,
    torch_backend: str | None = None,
    device: str | None = None,
) -> JAXShardedStatevectorResult:
    """Execute a true amplitude-sharded statevector path with JAX shard arrays.

    This is the executable development backend for the statevector part of the
    JAX distributed plan: one logical state is partitioned across rank shards,
    gates update only owned amplitudes, and cross-shard gates exchange the
    minimal amplitude groups implied by the gate. Production pmap/shard_map
    transport and sharded backward are reported as blockers until implemented.
    """

    from ...statevector.planning import plan_distributed_statevector

    policy = _resolve_policy(
        distributed_backend_policy=distributed_backend_policy,
        distributed_profile=distributed_profile,
        jax_backend=jax_backend,
        torch_backend=torch_backend,
    )
    resolved_world_size = _resolve_world_size(world_size, policy)
    resolved_local_world_size = _resolve_local_world_size(
        local_world_size,
        world_size=resolved_world_size,
        policy=policy,
    )
    ir = _as_ir(circuit_or_ir)
    jax_dtype = _jax_complex_dtype(complex_bytes)
    torch_dtype = _torch_complex_dtype(complex_bytes)
    jax_device = _resolve_jax_device(device)
    plan = plan_distributed_statevector(
        ir,
        world_size=resolved_world_size,
        local_world_size=resolved_local_world_size,
        bsz=bsz,
        complex_bytes=complex_bytes,
    )
    jax_plan = plan_jax_distributed_quantum_backend(
        ir,
        mode="statevector",
        world_size=resolved_world_size,
        local_world_size=resolved_local_world_size,
        bsz=bsz,
        complex_bytes=complex_bytes,
        distributed_backend_policy=policy,
    )
    shards = tuple(
        _initialize_jax_statevector_shard(
            plan, rank=rank, dtype=jax_dtype, device=jax_device
        )
        for rank in range(plan.world_size)
    )
    local_gate_count = 0
    distributed_gate_count = 0
    simulated_comm_count = 0
    simulated_comm_bytes = 0
    for instruction, gate_plan in zip(ir.instructions, plan.gate_plans):
        matrix, diagonal = _gate_matrix_as_jax(
            instruction, torch_dtype=torch_dtype, jax_dtype=jax_dtype
        )
        matrix = _jnp_device_put(matrix, jax_device)
        shards = _apply_gate_to_jax_shards(
            shards, matrix, instruction.wires, plan=plan, diagonal=diagonal
        )
        if gate_plan.communication == "local":
            local_gate_count += 1
        else:
            distributed_gate_count += 1
            simulated_comm_count += 1
            simulated_comm_bytes += int(gate_plan.estimated_transfer_bytes)
    return JAXShardedStatevectorResult(
        shards=shards,
        plan=plan,
        jax_plan=jax_plan,
        backend_policy=policy,
        local_gate_count=local_gate_count,
        distributed_gate_count=distributed_gate_count,
        simulated_communication_count=simulated_comm_count,
        simulated_communication_bytes=simulated_comm_bytes,
    )


def _jax_parameterized_statevector_shards(
    circuit: Any,
    *,
    plan: Any,
    complex_bytes: int,
    device: Any | None,
) -> tuple[tuple[JAXStatevectorShardState, ...], int, int, int, int]:
    jax_dtype = _jax_complex_dtype(complex_bytes)
    shards = tuple(
        _initialize_jax_statevector_shard(
            plan, rank=rank, dtype=jax_dtype, device=device
        )
        for rank in range(plan.world_size)
    )
    local_gate_count = 0
    distributed_gate_count = 0
    simulated_comm_count = 0
    simulated_comm_bytes = 0
    for instruction, gate_plan in zip(circuit.to_ir().instructions, plan.gate_plans):
        matrix, diagonal = _parameterized_gate_matrix_as_jax(
            instruction, complex_bytes=complex_bytes
        )
        matrix = _jnp_device_put(matrix, device)
        shards = _apply_gate_to_jax_shards(
            shards, matrix, instruction.wires, plan=plan, diagonal=diagonal
        )
        if gate_plan.communication == "local":
            local_gate_count += 1
        else:
            distributed_gate_count += 1
            simulated_comm_count += 1
            simulated_comm_bytes += int(gate_plan.estimated_transfer_bytes)
    return (
        shards,
        local_gate_count,
        distributed_gate_count,
        simulated_comm_count,
        simulated_comm_bytes,
    )


def jax_sharded_statevector_parameter_value_and_grad(
    circuit_builder: Callable[[Any], Any],
    parameters: Any,
    *,
    n_wires: int,
    world_size: int | None = None,
    local_world_size: int | None = None,
    bsz: int = 1,
    complex_bytes: int | None = None,
    dtype: Any | None = None,
    observable: str = "z_sum",
    observable_wires: Sequence[int] | None = None,
    distributed_backend_policy: DistributedBackendPolicy | None = None,
    distributed_profile: str | None = None,
    jax_backend: str | None = None,
    torch_backend: str | None = None,
    device: str | None = None,
    backward_backend: str = "auto",
    jit: bool = False,
) -> JAXShardedStatevectorParameterGradientResult:
    """Reverse-mode value/gradient for an amplitude-sharded parameterized circuit."""

    import numpy as np

    from ...statevector.planning import plan_distributed_statevector

    torch = _require_torch()
    jax, jnp = _require_jax()
    policy = _resolve_policy(
        distributed_backend_policy=distributed_backend_policy,
        distributed_profile=distributed_profile,
        jax_backend=jax_backend,
        torch_backend=torch_backend,
    )
    resolved_world_size = _resolve_world_size(world_size, policy)
    resolved_local_world_size = _resolve_local_world_size(
        local_world_size,
        world_size=resolved_world_size,
        policy=policy,
    )
    resolved_backward_backend = _resolve_jax_backward_backend(backward_backend, policy)
    if local_world_size is None and resolved_backward_backend in {"pmap", "shard_map"}:
        device_summary = _jax_device_count_summary()
        if (
            resolved_backward_backend == "pmap"
            and int(device_summary["global_device_count"]) >= int(resolved_world_size)
            and int(device_summary["local_device_count"]) > 0
        ):
            resolved_local_world_size = min(
                int(resolved_world_size), int(device_summary["local_device_count"])
            )
        elif (
            resolved_backward_backend == "shard_map"
            and int(device_summary["process_count"]) == 1
            and int(device_summary["local_device_count"]) >= int(resolved_world_size)
        ):
            resolved_local_world_size = int(resolved_world_size)
    if complex_bytes is None:
        complex_bytes = 16 if dtype == torch.complex128 else 8
    static_parameters = _torch_parameters_for_static_build(
        parameters, complex_bytes=complex_bytes
    )
    parameter_shape = tuple(int(dim) for dim in static_parameters.shape)
    example_circuit = circuit_builder(static_parameters)
    ir = _as_ir(example_circuit)
    if int(ir.n_wires) != int(n_wires):
        raise ValueError(
            f"n_wires={n_wires} does not match circuit IR n_wires={ir.n_wires}."
        )
    plan = plan_distributed_statevector(
        ir,
        world_size=resolved_world_size,
        local_world_size=resolved_local_world_size,
        bsz=bsz,
        complex_bytes=complex_bytes,
    )
    if resolved_backward_backend == "pmap":
        production_blockers = _statevector_pmap_backward_blockers(plan)
    elif resolved_backward_backend == "shard_map":
        production_blockers = _statevector_shard_map_backward_blockers(plan)
    else:
        production_blockers = ()
    production_preflight = _require_jax_production_backward_ready(
        mode="statevector",
        backend=resolved_backward_backend,
        world_size=resolved_world_size,
        blockers=production_blockers,
    )
    jax_plan = plan_jax_distributed_quantum_backend(
        ir,
        mode="statevector",
        world_size=resolved_world_size,
        local_world_size=resolved_local_world_size,
        bsz=bsz,
        complex_bytes=complex_bytes,
        distributed_backend_policy=policy,
    )
    jax_device = (
        None
        if resolved_backward_backend in {"pmap", "shard_map"}
        else _resolve_jax_device(device)
    )
    if resolved_backward_backend in {"pmap", "shard_map"}:
        jax_parameters = np.asarray(
            static_parameters.detach().cpu()
            if torch.is_tensor(static_parameters)
            else static_parameters
        ).astype(
            np.float64 if int(complex_bytes) == 16 else np.float32,
            copy=True,
        )
    else:
        jax_parameters = _jax_parameter_array_from_input(
            parameters, complex_bytes=complex_bytes
        )
    compute_dtype = "complex128" if int(complex_bytes) == 16 else "complex64"

    def _loss(parameter_array: Any) -> Any:
        from ..kernel import _JAXParameterProxy, _set_active_jax_compute_dtype

        previous_dtype = _set_active_jax_compute_dtype(compute_dtype)
        try:
            parameter_array = jnp.asarray(
                parameter_array, dtype=_jax_real_dtype(complex_bytes)
            )
            circuit = circuit_builder(_JAXParameterProxy(parameter_array))
            shards, _local, _distributed, _comm_count, _comm_bytes = (
                _jax_parameterized_statevector_shards(
                    circuit,
                    plan=plan,
                    complex_bytes=complex_bytes,
                    device=jax_device,
                )
            )
            return _jax_sharded_statevector_loss_from_shards(
                shards,
                n_wires=int(n_wires),
                observable=observable,
                observable_wires=observable_wires,
            )
        finally:
            _set_active_jax_compute_dtype(previous_dtype)

    if resolved_backward_backend == "pmap":

        def loss_function(parameter_array: Any) -> Any:
            return _jax_pmap_statevector_parameter_loss(
                circuit_builder,
                parameter_array,
                plan=plan,
                complex_bytes=complex_bytes,
                observable=observable,
                observable_wires=observable_wires,
            )

    elif resolved_backward_backend == "shard_map":

        def loss_function(parameter_array: Any) -> Any:
            return _jax_shard_map_statevector_parameter_loss(
                circuit_builder,
                parameter_array,
                plan=plan,
                complex_bytes=complex_bytes,
                observable=observable,
                observable_wires=observable_wires,
            )

    else:
        loss_function = _loss
    value_and_grad = jax.value_and_grad(loss_function)
    if jit and resolved_backward_backend not in {"pmap", "shard_map"}:
        value_and_grad = jax.jit(value_and_grad)
    value, gradient = value_and_grad(jax_parameters)

    from ..kernel import _JAXParameterProxy, _set_active_jax_compute_dtype

    previous_dtype = _set_active_jax_compute_dtype(compute_dtype)
    try:
        if resolved_backward_backend in {"pmap", "shard_map"}:
            local_gate_count = sum(
                1 for gate_plan in plan.gate_plans if gate_plan.communication == "local"
            )
            distributed_gate_count = sum(
                1 for gate_plan in plan.gate_plans if gate_plan.communication != "local"
            )
            comm_count = distributed_gate_count
            comm_bytes = sum(
                int(gate_plan.estimated_transfer_bytes)
                for gate_plan in plan.gate_plans
                if gate_plan.communication != "local"
            )
        else:
            summary_circuit = circuit_builder(_JAXParameterProxy(jax_parameters))
            (
                _shards,
                local_gate_count,
                distributed_gate_count,
                comm_count,
                comm_bytes,
            ) = _jax_parameterized_statevector_shards(
                summary_circuit,
                plan=plan,
                complex_bytes=complex_bytes,
                device=jax_device,
            )
    finally:
        _set_active_jax_compute_dtype(previous_dtype)

    return JAXShardedStatevectorParameterGradientResult(
        value=value,
        gradient=gradient,
        plan=plan,
        jax_plan=jax_plan,
        backend_policy=policy,
        parameter_shape=parameter_shape,
        local_gate_count=local_gate_count,
        distributed_gate_count=distributed_gate_count,
        simulated_communication_count=comm_count,
        simulated_communication_bytes=comm_bytes,
        observable=str(observable),
        backward_backend=resolved_backward_backend,
        backward_execution=str(production_preflight["execution"]),
        production_device_summary=production_preflight["device_summary"],
        production_blockers=tuple(production_preflight["blockers"]),
    )
