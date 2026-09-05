"""Parameterized tensor-network construction and sliced reverse mode."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from ....simulation.jax_tensor_network import (
    jax_tensor_network_loss_from_output as _jax_tn_loss_from_output,
)
from ...distributed.backend_policy import DistributedBackendPolicy
from .array_conversions import (
    _jax_nodes_from_torch_nodes,
    _jax_parameter_array_from_input,
    _torch_parameters_for_static_build,
)
from .backend_dispatch import plan_jax_distributed_quantum_backend
from .common import node_count as _node_count
from .runtime_environment import (
    _jax_complex_dtype,
    _jax_real_dtype,
    _require_jax,
    _require_torch,
    _resolve_collective_backend,
    _resolve_local_world_size,
    _resolve_policy,
    _resolve_world_size,
    _torch_complex_dtype,
)
from .tensor_network_contraction import (
    _jax_contract_tensor_slices_by_backend,
    _pauli_ops_from_term,
)
from .tensor_network_execution import _resolve_tn_compute_backend
from .tensor_network_planning import _tn_tasks
from .tensor_network_records import (
    JAXSlicedTensorNetworkGradientResult,
    JAXSlicedTensorNetworkParameterGradientResult,
    JAXTensorNetworkNode,
)


def _jax_parameterized_tn_state_nodes(
    circuit: Any,
    n_wires: int,
    *,
    bsz: int,
    complex_bytes: int,
    batch_label: int,
    start_label: int,
    conjugate: bool,
) -> tuple[list[JAXTensorNetworkNode], list[int], int]:
    _, jnp = _require_jax()
    from ....simulation.jax_gate_primitives import _jax_instruction_matrix

    next_label = int(start_label)
    current_labels: list[int] = []
    nodes: list[JAXTensorNetworkNode] = []
    zero = jnp.zeros((int(bsz), 2), dtype=_jax_complex_dtype(complex_bytes))
    zero = zero.at[:, 0].set(1.0 + 0.0j)
    for wire in range(int(n_wires)):
        label = next_label
        next_label += 1
        current_labels.append(label)
        nodes.append(
            JAXTensorNetworkNode(
                tensor=zero, labels=(int(batch_label), label), name=f"init_{wire}"
            )
        )
    for index, instruction in enumerate(circuit.to_ir()):
        if instruction.metadata.get("is_channel"):
            raise NotImplementedError(
                "JAX parameterized sliced TN currently supports unitary circuit instructions."
            )
        wires = tuple(int(wire) for wire in instruction.wires)
        matrix = _jax_instruction_matrix(instruction)
        if conjugate:
            matrix = jnp.conj(matrix)
        if len(tuple(matrix.shape)) != 2:
            raise NotImplementedError(
                "JAX parameterized sliced TN expects unbatched gate matrices."
            )
        tensor = matrix.reshape((2,) * (2 * len(wires)))
        input_labels = tuple(current_labels[wire] for wire in wires)
        output_labels = tuple(range(next_label, next_label + len(wires)))
        next_label += len(wires)
        for wire, label in zip(wires, output_labels):
            current_labels[wire] = label
        nodes.append(
            JAXTensorNetworkNode(
                tensor=tensor,
                labels=output_labels + input_labels,
                name=f"{index}:{instruction.name}",
                metadata={"wires": wires},
            )
        )
    return nodes, current_labels, next_label


def _jax_pauli_matrix_for_parameterized_tn(name: str, *, complex_bytes: int) -> Any:
    _, jnp = _require_jax()
    dtype = _jax_complex_dtype(complex_bytes)
    normalized = str(name).lower()
    if normalized == "i":
        return jnp.eye(2, dtype=dtype)
    if normalized == "x":
        return jnp.asarray([[0, 1], [1, 0]], dtype=dtype)
    if normalized == "y":
        return jnp.asarray([[0, -1j], [1j, 0]], dtype=dtype)
    if normalized == "z":
        return jnp.asarray([[1, 0], [0, -1]], dtype=dtype)
    raise ValueError(f"Unsupported Pauli operator {name!r}.")


def _jax_parameterized_tn_expectation_nodes(
    circuit: Any,
    n_wires: int,
    *,
    bsz: int,
    complex_bytes: int,
    ops: Mapping[int, str],
) -> tuple[tuple[JAXTensorNetworkNode, ...], tuple[int, ...]]:
    _, jnp = _require_jax()
    ket_nodes, ket_outputs, next_label = _jax_parameterized_tn_state_nodes(
        circuit,
        n_wires,
        bsz=bsz,
        complex_bytes=complex_bytes,
        batch_label=0,
        start_label=1,
        conjugate=False,
    )
    max_label = max((label for node in ket_nodes for label in node.labels), default=0)
    offset = max(int(next_label), int(max_label) + 1)
    bra_nodes, bra_outputs, _ = _jax_parameterized_tn_state_nodes(
        circuit,
        n_wires,
        bsz=bsz,
        complex_bytes=complex_bytes,
        batch_label=offset,
        start_label=offset + 1,
        conjugate=True,
    )
    nodes: list[JAXTensorNetworkNode] = list(ket_nodes) + list(bra_nodes)
    batch_identity = jnp.eye(int(bsz), dtype=_jax_complex_dtype(complex_bytes))
    nodes.append(
        JAXTensorNetworkNode(
            tensor=batch_identity,
            labels=(0, offset),
            name="batch_identity",
        )
    )
    for wire in range(int(n_wires)):
        matrix = _jax_pauli_matrix_for_parameterized_tn(
            ops.get(int(wire), "i"), complex_bytes=complex_bytes
        )
        nodes.append(
            JAXTensorNetworkNode(
                tensor=matrix,
                labels=(int(bra_outputs[wire]), int(ket_outputs[wire])),
                name=f"obs_{wire}",
                metadata={"wire": int(wire)},
            )
        )
    return tuple(nodes), (0,)


def _static_expectation_plan_for_parameterized_tn(
    circuit: Any,
    *,
    bsz: int,
    complex_bytes: int,
    observable: str,
    observable_wires: Sequence[int] | None,
    hamiltonian_terms: Sequence[tuple[float, Sequence[tuple[int, str]]]],
) -> Any:
    from ....simulation.tensor import (
        build_tensor_network,
        build_tensor_network_expectation,
    )

    torch_dtype = _torch_complex_dtype(complex_bytes)
    base = build_tensor_network(circuit, bsz=bsz, device="cpu", dtype=torch_dtype)
    normalized = str(observable)
    if normalized in {"z", "z_sum"}:
        wires = (
            tuple(range(base.n_wires))
            if observable_wires is None
            else tuple(int(wire) for wire in observable_wires)
        )
        wire = int(wires[0]) if wires else 0
        return build_tensor_network_expectation(base, z=(wire,))
    if normalized == "state_norm":
        return build_tensor_network_expectation(base)
    if normalized == "hamiltonian":
        first_ops: Sequence[tuple[int, str]] = ()
        if hamiltonian_terms:
            _coefficient, first_ops = hamiltonian_terms[0]
        op_map = _pauli_ops_from_term(first_ops)
        return build_tensor_network_expectation(
            base,
            x=tuple(wire for wire, name in op_map.items() if name == "x"),
            y=tuple(wire for wire, name in op_map.items() if name == "y"),
            z=tuple(wire for wire, name in op_map.items() if name == "z"),
        )
    raise ValueError(
        "JAX parameterized sliced TN supports observable='z_sum', 'z', 'state_norm', or 'hamiltonian'."
    )


def _jax_parameterized_tn_scalar_from_nodes(
    nodes: Sequence[JAXTensorNetworkNode],
    output_labels: Sequence[int],
    tasks: Sequence[tuple[int, tuple[tuple[int, int], ...]]],
    *,
    world_size: int,
    compute_backend: str,
    collective_backend: str,
) -> Any:
    _, jnp = _require_jax()
    _partials, reduced, _compute_execution, _collective_execution = (
        _jax_contract_tensor_slices_by_backend(
            nodes,
            output_labels,
            tasks,
            world_size=world_size,
            compute_backend=compute_backend,
            collective_backend=collective_backend,
        )
    )
    return jnp.real(jnp.sum(reduced))


def _jax_parameterized_tn_observable_loss(
    circuit: Any,
    *,
    n_wires: int,
    bsz: int,
    complex_bytes: int,
    tasks: Sequence[tuple[int, tuple[tuple[int, int], ...]]],
    output_labels: Sequence[int],
    world_size: int,
    compute_backend: str,
    collective_backend: str,
    observable: str,
    observable_wires: Sequence[int] | None,
    hamiltonian_terms: Sequence[tuple[float, Sequence[tuple[int, str]]]],
) -> Any:
    _, jnp = _require_jax()
    normalized = str(observable)
    if normalized in {"z", "z_sum"}:
        wires = (
            tuple(range(int(n_wires)))
            if observable_wires is None
            else tuple(int(wire) for wire in observable_wires)
        )
        total = jnp.zeros((), dtype=_jax_real_dtype(complex_bytes))
        for wire in wires:
            nodes, _ = _jax_parameterized_tn_expectation_nodes(
                circuit,
                n_wires,
                bsz=bsz,
                complex_bytes=complex_bytes,
                ops={int(wire): "z"},
            )
            total = total + _jax_parameterized_tn_scalar_from_nodes(
                nodes,
                output_labels,
                tasks,
                world_size=world_size,
                compute_backend=compute_backend,
                collective_backend=collective_backend,
            )
        return total
    if normalized == "state_norm":
        nodes, _ = _jax_parameterized_tn_expectation_nodes(
            circuit,
            n_wires,
            bsz=bsz,
            complex_bytes=complex_bytes,
            ops={},
        )
        return _jax_parameterized_tn_scalar_from_nodes(
            nodes,
            output_labels,
            tasks,
            world_size=world_size,
            compute_backend=compute_backend,
            collective_backend=collective_backend,
        )
    if normalized == "hamiltonian":
        total = jnp.zeros((), dtype=_jax_real_dtype(complex_bytes))
        for coefficient, term_ops in hamiltonian_terms:
            nodes, _ = _jax_parameterized_tn_expectation_nodes(
                circuit,
                n_wires,
                bsz=bsz,
                complex_bytes=complex_bytes,
                ops=_pauli_ops_from_term(term_ops),
            )
            total = total + jnp.asarray(
                coefficient, dtype=_jax_real_dtype(complex_bytes)
            ) * _jax_parameterized_tn_scalar_from_nodes(
                nodes,
                output_labels,
                tasks,
                world_size=world_size,
                compute_backend=compute_backend,
                collective_backend=collective_backend,
            )
        return total
    raise ValueError(
        "JAX parameterized sliced TN supports observable='z_sum', 'z', 'state_norm', or 'hamiltonian'."
    )


def jax_sliced_tensor_network_value_and_grad(
    circuit_or_ir: Any,
    *,
    world_size: int | None = None,
    local_world_size: int | None = None,
    bsz: int = 1,
    complex_bytes: int | None = None,
    dtype: Any | None = None,
    max_intermediate_size: int | None = None,
    sliced_labels: Sequence[int] | None = None,
    observable: str = "z_sum",
    observable_wires: Sequence[int] | None = None,
    distributed_backend_policy: DistributedBackendPolicy | None = None,
    distributed_profile: str | None = None,
    jax_backend: str | None = None,
    torch_backend: str | None = None,
    compute_backend: str = "auto",
    collective_backend: str = "local_simulated",
    jit: bool = False,
) -> JAXSlicedTensorNetworkGradientResult:
    """Reverse-mode value/gradient for sliced JAX tensor-network contraction.

    The current gradient target is the tensor-network node tensors. Parameter
    gate pullbacks are intentionally reported as a remaining blocker instead of
    being implied by node-level VJP coverage.
    """

    from ....simulation.tensor import build_tensor_network

    torch = _require_torch()
    jax, _ = _require_jax()
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
    torch_dtype = _torch_complex_dtype(complex_bytes)
    jax_dtype = _jax_complex_dtype(complex_bytes)
    plan = build_tensor_network(circuit_or_ir, bsz=bsz, device="cpu", dtype=torch_dtype)
    slicing = plan.slicing_plan(
        max_intermediate_size=max_intermediate_size, sliced_labels=sliced_labels
    )
    tasks = _tn_tasks(slicing.sliced_labels, slicing.slice_shape, resolved_world_size)
    active_ranks = tuple(sorted({int(rank) for rank, _ in tasks}))
    if resolved_world_size > 1 and (int(slicing.n_slices) < 2 or len(active_ranks) < 2):
        raise RuntimeError(
            "JAX sliced tensor-network reverse mode requires at least two slice tasks assigned to multiple ranks."
        )
    base_nodes = _jax_nodes_from_torch_nodes(plan.nodes, dtype=jax_dtype, device=None)
    specs = tuple((node.labels, node.name, node.metadata) for node in base_nodes)
    tensors = tuple(node.tensor for node in base_nodes)

    def _loss(node_tensors: tuple[Any, ...]) -> Any:
        nodes = tuple(
            JAXTensorNetworkNode(
                tensor=tensor,
                labels=tuple(labels),
                name=str(name),
                metadata=metadata,
            )
            for tensor, (labels, name, metadata) in zip(node_tensors, specs)
        )
        _partials, reduced, _compute_execution, _collective_execution = (
            _jax_contract_tensor_slices_by_backend(
                nodes,
                plan.output_labels,
                tasks,
                world_size=resolved_world_size,
                compute_backend=resolved_compute_backend,
                collective_backend=resolved_collective_backend,
            )
        )
        return _jax_tn_loss_from_output(
            reduced,
            n_wires=plan.n_wires,
            bsz=plan.bsz,
            observable=observable,
            observable_wires=observable_wires,
        )

    value_and_grad = jax.value_and_grad(_loss)
    if jit:
        value_and_grad = jax.jit(value_and_grad)
    value, gradients = value_and_grad(tensors)
    jax_plan = plan_jax_distributed_quantum_backend(
        circuit_or_ir,
        mode="tensor_network",
        world_size=resolved_world_size,
        local_world_size=resolved_local_world_size,
        bsz=plan.bsz,
        complex_bytes=complex_bytes,
        max_intermediate_size=max_intermediate_size,
        sliced_labels=slicing.sliced_labels,
        distributed_backend_policy=policy,
    )
    _partials, _reduced, compute_execution, collective_execution = (
        _jax_contract_tensor_slices_by_backend(
            base_nodes,
            plan.output_labels,
            tasks,
            world_size=resolved_world_size,
            compute_backend=resolved_compute_backend,
            collective_backend=resolved_collective_backend,
        )
    )
    return JAXSlicedTensorNetworkGradientResult(
        value=value,
        node_gradients=tuple(gradients),
        slicing=slicing,
        jax_plan=jax_plan,
        backend_policy=policy,
        n_wires=plan.n_wires,
        bsz=plan.bsz,
        complex_bytes=complex_bytes,
        local_world_size=resolved_local_world_size,
        node_count=_node_count(resolved_world_size, resolved_local_world_size),
        compute_backend=resolved_compute_backend,
        compute_execution=compute_execution,
        collective_backend=resolved_collective_backend,
        collective_execution=collective_execution,
        observable=str(observable),
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
    parameters. It contracts observable tensor networks directly, instead of
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
