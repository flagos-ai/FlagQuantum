"""Local, pair-exchange, all-to-all, pmap, and shard-map statevector kernels."""

from __future__ import annotations

from typing import Any, Callable, Sequence

from ....simulation.jax_statevector import (
    jax_apply_local_statevector_gate,
    jax_combine_pair_exchanged_statevector,
    jax_gate_basis_in_for_delta_and_local_input,
    jax_local_positions_for_gate_input,
    jax_rank_mask_for_touched_delta,
)
from ....simulation.jax_statevector import (
    jax_sharded_statevector_rank_loss as _jax_sharded_statevector_rank_loss_from_local_amplitudes,
)
from .array_conversions import (
    _jax_basis_indices_for_wires,
    _parameterized_gate_matrix_as_jax,
)
from .planning_core import _jax_global_indices_by_rank_for_plan
from .runtime_environment import (
    _jax_complex_dtype,
    _jax_pmap_device_assignment,
    _require_jax,
)
from .statevector_records import JAXStatevectorShardState


def _jax_sharded_statevector_loss_from_shards(
    shards: Sequence[JAXStatevectorShardState],
    *,
    n_wires: int,
    observable: str,
    observable_wires: Sequence[int] | None,
) -> Any:
    _, jnp = _require_jax()
    normalized = str(observable)
    if normalized == "state_norm":
        total = jnp.zeros(
            (), dtype=jnp.real(shards[0].amplitudes).dtype if shards else jnp.float32
        )
        for shard in shards:
            total = total + jnp.sum(jnp.abs(shard.amplitudes) ** 2)
        return jnp.real(total)
    if normalized not in {"z", "z_sum"}:
        raise ValueError(
            "JAX sharded statevector parameter gradients currently support observable='z_sum', 'z', or 'state_norm'."
        )
    wires = (
        tuple(range(int(n_wires)))
        if observable_wires is None
        else tuple(int(wire) for wire in observable_wires)
    )
    total = jnp.zeros(
        (), dtype=jnp.real(shards[0].amplitudes).dtype if shards else jnp.float32
    )
    for shard in shards:
        probs = jnp.abs(shard.amplitudes) ** 2
        for wire in wires:
            bit = (shard.global_indices >> (int(n_wires) - 1 - int(wire))) & 1
            signs = 1.0 - 2.0 * bit.astype(probs.real.dtype)
            total = total + jnp.sum(probs * signs.reshape(1, -1))
    return jnp.real(total)


def _jax_apply_local_statevector_instruction(
    amplitudes: Any,
    global_indices: Any,
    instruction: Any,
    *,
    plan: Any,
    complex_bytes: int,
) -> Any:
    matrix, diagonal = _parameterized_gate_matrix_as_jax(
        instruction, complex_bytes=complex_bytes
    )
    wires = tuple(int(wire) for wire in instruction.wires)
    return jax_apply_local_statevector_gate(
        amplitudes,
        global_indices,
        matrix,
        wires,
        n_wires=int(plan.n_wires),
        sharded_wires=tuple(int(wire) for wire in plan.sharded_wires),
        diagonal=diagonal,
        gate_name=str(instruction.name),
    )


def _jax_apply_all_to_all_statevector_instruction(
    amplitudes: Any,
    global_indices: Any,
    instruction: Any,
    *,
    plan: Any,
    complex_bytes: int,
) -> Any:
    jax, jnp = _require_jax()
    matrix, diagonal = _parameterized_gate_matrix_as_jax(
        instruction, complex_bytes=complex_bytes
    )
    wires = tuple(int(wire) for wire in instruction.wires)
    sharded_wires = tuple(int(wire) for wire in plan.sharded_wires)
    sharded_set = set(sharded_wires)
    touched = tuple(wire for wire in wires if wire in sharded_set)
    if diagonal or not touched:
        return _jax_apply_local_statevector_instruction(
            amplitudes,
            global_indices,
            instruction,
            plan=plan,
            complex_bytes=complex_bytes,
        )
    local_wires = tuple(
        wire for wire in range(int(plan.n_wires)) if wire not in sharded_set
    )
    local_gate_wires = tuple(wire for wire in wires if wire not in sharded_set)
    basis_out = _jax_basis_indices_for_wires(
        global_indices, n_wires=int(plan.n_wires), wires=wires
    )
    updated = jnp.zeros_like(amplitudes)
    for delta_code in range(2 ** len(touched)):
        rank_mask = jax_rank_mask_for_touched_delta(
            sharded_wires,
            touched,
            delta_code,
        )
        if rank_mask:
            perm = tuple(
                (rank, rank ^ rank_mask) for rank in range(int(plan.world_size))
            )
            source_amplitudes = jax.lax.ppermute(
                amplitudes, axis_name="fq_rank", perm=perm
            )
        else:
            source_amplitudes = amplitudes
        for local_input_basis in range(2 ** len(local_gate_wires)):
            source_positions = jax_local_positions_for_gate_input(
                global_indices,
                n_wires=int(plan.n_wires),
                local_wires=local_wires,
                local_gate_wires=local_gate_wires,
                local_input_basis=local_input_basis,
            )
            source_values = jnp.take(source_amplitudes, source_positions, axis=1)
            basis_in = jax_gate_basis_in_for_delta_and_local_input(
                global_indices,
                n_wires=int(plan.n_wires),
                wires=wires,
                touched_sharded_wires=touched,
                local_gate_wires=local_gate_wires,
                delta_code=delta_code,
                local_input_basis=local_input_basis,
            )
            if matrix.ndim == 2:
                coeff = matrix[basis_out, basis_in].reshape(1, -1)
            else:
                coeff = matrix[:, basis_out, basis_in]
            updated = updated + source_values * coeff
    return updated


def _jax_apply_pair_exchange_statevector_instruction(
    amplitudes: Any,
    global_indices: Any,
    instruction: Any,
    *,
    plan: Any,
    complex_bytes: int,
) -> Any:
    jax, _ = _require_jax()
    matrix, diagonal = _parameterized_gate_matrix_as_jax(
        instruction, complex_bytes=complex_bytes
    )
    wires = tuple(int(wire) for wire in instruction.wires)
    sharded_wires = tuple(int(wire) for wire in plan.sharded_wires)
    touched = tuple(wire for wire in wires if wire in set(sharded_wires))
    if diagonal or not touched:
        return _jax_apply_local_statevector_instruction(
            amplitudes,
            global_indices,
            instruction,
            plan=plan,
            complex_bytes=complex_bytes,
        )
    if len(wires) != 1 or len(touched) != 1:
        return _jax_apply_all_to_all_statevector_instruction(
            amplitudes,
            global_indices,
            instruction,
            plan=plan,
            complex_bytes=complex_bytes,
        )
    wire = int(touched[0])
    bit_index = sharded_wires.index(wire)
    rank_mask = 1 << (len(sharded_wires) - bit_index - 1)
    perm = tuple((rank, rank ^ rank_mask) for rank in range(int(plan.world_size)))
    partner = jax.lax.ppermute(amplitudes, axis_name="fq_rank", perm=perm)
    return jax_combine_pair_exchanged_statevector(
        amplitudes,
        partner,
        global_indices,
        matrix,
        n_wires=int(plan.n_wires),
        wire=wire,
    )


def _jax_pmap_statevector_parameter_loss(
    circuit_builder: Callable[[Any], Any],
    parameter_array: Any,
    *,
    plan: Any,
    complex_bytes: int,
    observable: str,
    observable_wires: Sequence[int] | None,
) -> Any:
    jax, jnp = _require_jax()
    import numpy as np

    from .kernel import _JAXParameterProxy, _set_active_jax_compute_dtype

    compute_dtype = "complex128" if int(complex_bytes) == 16 else "complex64"
    local_size = int(plan.shards[0].local_amplitudes)
    if any(int(shard.local_amplitudes) != local_size for shard in plan.shards):
        raise RuntimeError(
            "JAX pmap statevector backward requires equal local shard sizes."
        )
    global_indices_host = np.asarray(_jax_global_indices_by_rank_for_plan(plan))
    pmap_devices, _local_devices, local_rank_indices = _jax_pmap_device_assignment(
        int(plan.world_size)
    )
    global_indices_by_rank = global_indices_host[list(local_rank_indices)]

    def _rank_loss(global_indices: Any, params: Any) -> Any:
        previous = _set_active_jax_compute_dtype(compute_dtype)
        try:
            local_circuit = circuit_builder(_JAXParameterProxy(params))
            local_instructions = tuple(local_circuit.to_ir().instructions)
            amplitudes = jnp.zeros(
                (int(plan.bsz), int(local_size)),
                dtype=_jax_complex_dtype(complex_bytes),
            )
            initial = jnp.where(
                global_indices == 0,
                jnp.asarray(1.0 + 0.0j, dtype=amplitudes.dtype),
                jnp.asarray(0.0 + 0.0j, dtype=amplitudes.dtype),
            )
            amplitudes = amplitudes.at[:, :].set(initial.reshape(1, -1))
            for instruction in local_instructions:
                amplitudes = _jax_apply_pair_exchange_statevector_instruction(
                    amplitudes,
                    global_indices,
                    instruction,
                    plan=plan,
                    complex_bytes=complex_bytes,
                )
            local_value = _jax_sharded_statevector_rank_loss_from_local_amplitudes(
                amplitudes,
                global_indices,
                n_wires=int(plan.n_wires),
                observable=observable,
                observable_wires=observable_wires,
            )
            return jax.lax.psum(local_value, axis_name="fq_rank")
        finally:
            _set_active_jax_compute_dtype(previous)

    replicated_parameters = jnp.broadcast_to(
        parameter_array,
        (len(local_rank_indices),) + tuple(int(dim) for dim in parameter_array.shape),
    )
    totals = jax.pmap(
        _rank_loss, in_axes=(0, 0), axis_name="fq_rank", devices=pmap_devices
    )(
        global_indices_by_rank,
        replicated_parameters,
    )
    return totals[0]


def _statevector_shard_map_backward_blockers(plan: Any) -> tuple[str, ...]:
    blockers: list[str] = []
    if int(plan.world_size) <= 1:
        blockers.append("world_size_is_one")
    shard_sizes = {int(shard.local_amplitudes) for shard in plan.shards}
    if len(shard_sizes) != 1:
        blockers.append("shard_map_statevector_equal_shard_size_required")
    if str(plan.distribution) != "qubit_address_sharded":
        blockers.append(
            f"shard_map_statevector_qubit_address_sharding_required:{plan.distribution}"
        )
    unsupported = tuple(
        {
            str(gate_plan.communication)
            for gate_plan in plan.gate_plans
            if str(gate_plan.communication) not in {"local", "pair_exchange"}
        }
    )
    if unsupported:
        blockers.append(
            "shard_map_statevector_multi_sharded_wire_transport_pending:"
            + ",".join(sorted(unsupported))
        )
    return tuple(blockers)


def _statevector_pmap_backward_blockers(plan: Any) -> tuple[str, ...]:
    blockers: list[str] = []
    if int(plan.world_size) <= 1:
        blockers.append("world_size_is_one")
    shard_sizes = {int(shard.local_amplitudes) for shard in plan.shards}
    if len(shard_sizes) != 1:
        blockers.append("pmap_statevector_equal_shard_size_required")
    if str(plan.distribution) != "qubit_address_sharded":
        blockers.append(
            f"pmap_statevector_qubit_address_sharding_required:{plan.distribution}"
        )
    unsupported = tuple(
        {
            str(gate_plan.communication)
            for gate_plan in plan.gate_plans
            if str(gate_plan.communication)
            not in {"local", "pair_exchange", "all_to_all"}
        }
    )
    if unsupported:
        blockers.append(
            "pmap_statevector_cross_rank_transport_pending:"
            + ",".join(sorted(unsupported))
        )
    return tuple(blockers)


def _jax_shard_map_statevector_parameter_loss(
    circuit_builder: Callable[[Any], Any],
    parameter_array: Any,
    *,
    plan: Any,
    complex_bytes: int,
    observable: str,
    observable_wires: Sequence[int] | None,
) -> Any:
    jax, jnp = _require_jax()
    import numpy as np
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    from .kernel import _JAXParameterProxy, _set_active_jax_compute_dtype

    compute_dtype = "complex128" if int(complex_bytes) == 16 else "complex64"
    local_size = int(plan.shards[0].local_amplitudes)
    if any(int(shard.local_amplitudes) != local_size for shard in plan.shards):
        raise RuntimeError(
            "JAX shard_map statevector backward requires equal local shard sizes."
        )
    blockers = _statevector_shard_map_backward_blockers(plan)
    if blockers:
        raise RuntimeError(
            "JAX shard_map statevector backward is not production-ready for this workload: "
            f"{tuple(blockers)}. FlagQuantum refuses to run a replicated, local, or numerically unstable fallback."
        )
    local_devices = tuple(jax.local_devices())
    if len(local_devices) < int(plan.world_size):
        raise RuntimeError(
            f"JAX shard_map statevector backward currently requires {int(plan.world_size)} local JAX devices "
            f"to construct a single-process mesh, but only {len(local_devices)} are visible. Use "
            "backward_backend='pmap' after JAX multi-process initialization for multi-node execution."
        )
    mesh = Mesh(
        np.asarray(local_devices[: int(plan.world_size)], dtype=object), ("fq_rank",)
    )
    global_indices_by_rank = _jax_global_indices_by_rank_for_plan(plan)

    def _rank_loss(global_indices: Any, params: Any) -> Any:
        previous = _set_active_jax_compute_dtype(compute_dtype)
        try:
            local_circuit = circuit_builder(_JAXParameterProxy(params))
            local_instructions = tuple(local_circuit.to_ir().instructions)
            amplitudes = jnp.zeros(
                (int(plan.bsz), int(local_size)),
                dtype=_jax_complex_dtype(complex_bytes),
            )
            initial = jnp.where(
                global_indices == 0,
                jnp.asarray(1.0 + 0.0j, dtype=amplitudes.dtype),
                jnp.asarray(0.0 + 0.0j, dtype=amplitudes.dtype),
            )
            amplitudes = amplitudes.at[:, :].set(initial.reshape(1, -1))
            for instruction in local_instructions:
                amplitudes = _jax_apply_pair_exchange_statevector_instruction(
                    amplitudes,
                    global_indices,
                    instruction,
                    plan=plan,
                    complex_bytes=complex_bytes,
                )
            local_value = _jax_sharded_statevector_rank_loss_from_local_amplitudes(
                amplitudes,
                global_indices,
                n_wires=int(plan.n_wires),
                observable=observable,
                observable_wires=observable_wires,
            )
            return jax.lax.psum(local_value, axis_name="fq_rank")
        finally:
            _set_active_jax_compute_dtype(previous)

    mapped_loss = jax.shard_map(
        _rank_loss,
        mesh=mesh,
        in_specs=(P("fq_rank"), P()),
        out_specs=P(),
        axis_names={"fq_rank"},
        check_vma=False,
    )
    return mapped_loss(global_indices_by_rank, parameter_array)
