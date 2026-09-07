"""MPS parameter gradients, backend blockers, and parameter-flow planning."""

from __future__ import annotations

from typing import Any, Callable, Sequence

from ...distributed.backend_policy import DistributedBackendPolicy
from .array_conversions import (
    _jax_parameter_array_from_input,
    _torch_parameters_for_static_build,
)
from .backend_dispatch import plan_jax_distributed_quantum_backend
from .mps.execution import _jax_parameterized_mps_rank_tensors
from .mps.gradient_result import JAXShardedMPSParameterGradientResult
from .mps.planning import (
    _mps_pmap_backward_blockers,
    plan_jax_sharded_mps_parameter_flow,
)
from .mps_boundary_exchange import _execute_local_mps_boundary_adjoint_exchange
from .mps_gradient_ownership import (
    _execute_local_mps_parameter_gradient_ownership,
    _jax_sharded_mps_z_sum_from_rank_tensors,
)
from .mps_kernels import _rank_shards_from_jax_mps_tensors
from .planning_core import _as_ir
from .runtime_environment import (
    _jax_real_dtype,
    _require_jax,
    _require_jax_production_backward_ready,
    _require_torch,
    _resolve_jax_backward_backend,
    _resolve_jax_device,
    _resolve_local_world_size,
    _resolve_policy,
    _resolve_world_size,
)


def jax_sharded_mps_parameter_value_and_grad(
    circuit_builder: Callable[[Any], Any],
    parameters: Any,
    *,
    n_wires: int,
    world_size: int | None = None,
    local_world_size: int | None = None,
    bsz: int = 1,
    complex_bytes: int | None = None,
    dtype: Any | None = None,
    max_bond: int | None = None,
    cutoff: float = 0.0,
    observable: str = "z_sum",
    observable_wires: Sequence[int] | None = None,
    distributed_backend_policy: DistributedBackendPolicy | None = None,
    distributed_profile: str | None = None,
    jax_backend: str | None = None,
    torch_backend: str | None = None,
    device: str | None = None,
    backward_backend: str = "auto",
    jit: bool = False,
) -> JAXShardedMPSParameterGradientResult:
    """Reverse-mode value/gradient for a site-sharded parameterized MPS circuit."""

    from ...distributed.engine import _instruction_is_boundary_local, _mps_shards

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
    shard_plans = _mps_shards(ir.n_wires, resolved_world_size)
    parameter_flow_plan = plan_jax_sharded_mps_parameter_flow(
        ir,
        world_size=resolved_world_size,
        local_world_size=resolved_local_world_size,
        distributed_backend_policy=policy,
    )
    boundary_preflight_count = sum(
        1
        for instruction in ir
        if _instruction_is_boundary_local(instruction, shard_plans)
    )
    resolved_backward_backend = _resolve_jax_backward_backend(backward_backend, policy)
    production_preflight = _require_jax_production_backward_ready(
        mode="mps",
        backend=resolved_backward_backend,
        world_size=resolved_world_size,
        blockers=(
            _mps_pmap_backward_blockers(
                world_size=resolved_world_size,
                boundary_sync_count=boundary_preflight_count,
            )
            if resolved_backward_backend != "local_simulated"
            else ()
        ),
    )
    jax_plan = plan_jax_distributed_quantum_backend(
        ir,
        mode="mps",
        world_size=resolved_world_size,
        local_world_size=resolved_local_world_size,
        bsz=bsz,
        complex_bytes=complex_bytes,
        max_bond=max_bond,
        distributed_backend_policy=policy,
    )
    jax_device = _resolve_jax_device(device)
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
            rank_tensors, *_ = _jax_parameterized_mps_rank_tensors(
                circuit,
                n_wires=int(n_wires),
                bsz=bsz,
                shard_plans=shard_plans,
                complex_bytes=complex_bytes,
                max_bond=max_bond,
                cutoff=cutoff,
                local_world_size=resolved_local_world_size,
                device=jax_device,
            )
            if str(observable) not in {"z", "z_sum"}:
                raise ValueError(
                    "JAX sharded MPS parameter gradients currently support observable='z_sum' or 'z'."
                )
            return _jax_sharded_mps_z_sum_from_rank_tensors(
                rank_tensors,
                n_wires=int(n_wires),
                bsz=bsz,
                complex_bytes=complex_bytes,
                observable_wires=observable_wires,
            )
        finally:
            _set_active_jax_compute_dtype(previous_dtype)

    value_and_grad = jax.value_and_grad(_loss)
    if jit:
        value_and_grad = jax.jit(value_and_grad)
    value, gradient = value_and_grad(jax_parameters)

    from .kernel import _JAXParameterProxy, _set_active_jax_compute_dtype

    previous_dtype = _set_active_jax_compute_dtype(compute_dtype)
    try:
        summary_circuit = circuit_builder(_JAXParameterProxy(jax_parameters))
        (
            rank_tensors,
            owned_instruction_count,
            sharded_kernel_count,
            boundary_sync_count,
            boundary_transfer_bytes,
            truncation_records,
            boundary_protocols,
        ) = _jax_parameterized_mps_rank_tensors(
            summary_circuit,
            n_wires=int(n_wires),
            bsz=bsz,
            shard_plans=shard_plans,
            complex_bytes=complex_bytes,
            max_bond=max_bond,
            cutoff=cutoff,
            local_world_size=resolved_local_world_size,
            device=jax_device,
        )
    finally:
        _set_active_jax_compute_dtype(previous_dtype)

    boundary_adjoint_exchange_evidence = _execute_local_mps_boundary_adjoint_exchange(
        rank_tensors,
        boundary_protocols,
        shard_plans,
    )
    parameter_gradient_ownership_evidence = (
        _execute_local_mps_parameter_gradient_ownership(
            circuit_builder,
            jax_parameters,
            gradient,
            parameter_flow_plan,
            shard_plans,
        )
    )

    return JAXShardedMPSParameterGradientResult(
        value=value,
        gradient=gradient,
        rank_shards=_rank_shards_from_jax_mps_tensors(rank_tensors, shard_plans),
        shard_plans=tuple(shard_plans),
        jax_plan=jax_plan,
        backend_policy=policy,
        n_wires=ir.n_wires,
        bsz=bsz,
        complex_bytes=complex_bytes,
        parameter_shape=parameter_shape,
        max_bond=max_bond,
        cutoff=float(cutoff),
        owned_instruction_count=owned_instruction_count,
        sharded_kernel_count=sharded_kernel_count,
        boundary_sync_count=boundary_sync_count,
        boundary_transfer_bytes=boundary_transfer_bytes,
        truncation_records=tuple(truncation_records),
        boundary_protocols=tuple(boundary_protocols),
        boundary_adjoint_exchange_evidence=boundary_adjoint_exchange_evidence,
        parameter_gradient_ownership_evidence=parameter_gradient_ownership_evidence,
        observable=str(observable),
        backward_backend=resolved_backward_backend,
        backward_execution=str(production_preflight["execution"]),
        production_device_summary=production_preflight["device_summary"],
        production_blockers=tuple(production_preflight["blockers"]),
        parameters=jax_parameters,
    )
