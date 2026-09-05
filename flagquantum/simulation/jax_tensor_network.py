"""Dependency-light local JAX tensor-network contraction."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .jax_gate_primitives import (
    _jax_complex_dtype,
    _jax_instruction_matrix,
    _jax_pauli_matrix,
    _jax_real_dtype,
)

_JAX_EINSUM_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def jax_tensor_network_statevector_from_circuit(
    circuit: Any,
    n_wires: int,
    parameters: Any,
    *,
    matmul_precision: str | None = "highest",
) -> Any:
    del parameters
    nodes, current_labels, _next_label = jax_tensor_network_nodes_from_circuit(
        circuit,
        n_wires,
        start_label=0,
        conjugate=False,
    )
    out = jax_contract_nodes_greedy(nodes, tuple(current_labels), matmul_precision)
    return out.reshape(-1)


def jax_tensor_network_nodes_from_circuit(
    circuit: Any,
    n_wires: int,
    *,
    start_label: int = 0,
    conjugate: bool = False,
) -> tuple[list[tuple[Any, tuple[int, ...]]], list[int], int]:
    import jax.numpy as jnp

    next_label = int(start_label)
    current_labels = []
    nodes: list[tuple[Any, tuple[int, ...]]] = []
    zero = jnp.asarray([1.0 + 0.0j, 0.0 + 0.0j], dtype=_jax_complex_dtype())
    for _wire in range(int(n_wires)):
        label = next_label
        next_label += 1
        current_labels.append(label)
        nodes.append((zero, (label,)))
    for instruction in circuit.to_ir():
        if instruction.metadata.get("is_channel"):
            raise NotImplementedError(
                "JAX tensor_network mode currently supports unitary circuit instructions."
            )
        wires = tuple(int(wire) for wire in instruction.wires)
        matrix = _jax_instruction_matrix(instruction).reshape((2,) * (2 * len(wires)))
        if conjugate:
            matrix = jnp.conj(matrix)
        input_labels = tuple(current_labels[wire] for wire in wires)
        output_labels = tuple(range(next_label, next_label + len(wires)))
        next_label += len(wires)
        for wire, label in zip(wires, output_labels):
            current_labels[wire] = label
        nodes.append((matrix, output_labels + input_labels))
    return nodes, current_labels, next_label


def jax_tensor_network_expectation_product_ops(
    circuit: Any,
    n_wires: int,
    ops: dict[int, Any],
    matmul_precision: str | None,
) -> Any:
    import jax.numpy as jnp

    ket_nodes, ket_labels, next_label = jax_tensor_network_nodes_from_circuit(
        circuit,
        n_wires,
        start_label=0,
        conjugate=False,
    )
    bra_nodes, bra_labels, _next_label = jax_tensor_network_nodes_from_circuit(
        circuit,
        n_wires,
        start_label=next_label,
        conjugate=True,
    )
    identity = _jax_pauli_matrix("i")
    op_nodes = []
    for wire in range(int(n_wires)):
        op = ops.get(int(wire), identity)
        op_nodes.append((op, (int(bra_labels[wire]), int(ket_labels[wire]))))
    value = jax_contract_nodes_greedy(
        bra_nodes + ket_nodes + op_nodes, tuple(), matmul_precision
    )
    return jnp.real(value.reshape(()))


def jax_tensor_network_z_values(
    circuit: Any,
    n_wires: int,
    wires: Iterable[int],
    matmul_precision: str | None,
) -> Any:
    import jax.numpy as jnp

    z_op = _jax_pauli_matrix("z")
    values = []
    for wire in wires:
        values.append(
            jax_tensor_network_expectation_product_ops(
                circuit,
                n_wires,
                {int(wire): z_op},
                matmul_precision,
            )
        )
    return jnp.stack(values) if values else jnp.zeros((0,), dtype=_jax_real_dtype())


def jax_tensor_network_z_sum(
    circuit: Any,
    n_wires: int,
    wires: Iterable[int],
    matmul_precision: str | None,
) -> Any:
    import jax.numpy as jnp

    values = jax_tensor_network_z_values(circuit, n_wires, wires, matmul_precision)
    return jnp.sum(values)


def jax_tensor_network_pauli_string_expectation(
    circuit: Any,
    n_wires: int,
    ops: tuple[tuple[int, str], ...],
    matmul_precision: str | None,
) -> Any:
    op_map = {}
    for wire, name in ops:
        normalized = str(name).lower()
        if normalized != "i":
            op_map[int(wire)] = _jax_pauli_matrix(normalized)
    return jax_tensor_network_expectation_product_ops(
        circuit, n_wires, op_map, matmul_precision
    )


def jax_tensor_network_hamiltonian_expectation(
    circuit: Any,
    n_wires: int,
    terms: tuple[tuple[float, tuple[tuple[int, str], ...]], ...],
    matmul_precision: str | None,
) -> Any:
    import jax.numpy as jnp

    total = jnp.zeros((), dtype=_jax_real_dtype())
    for coefficient, ops in terms:
        total = total + jnp.asarray(
            coefficient, dtype=_jax_real_dtype()
        ) * jax_tensor_network_pauli_string_expectation(
            circuit,
            n_wires,
            ops,
            matmul_precision,
        )
    return total


def jax_contract_nodes_greedy(
    nodes: list[tuple[Any, tuple[int, ...]]],
    output_labels: tuple[int, ...],
    matmul_precision: str | None,
) -> Any:
    active = list(nodes)
    final_outputs = set(output_labels)
    while len(active) > 1:
        counts: dict[int, int] = {}
        for _tensor, labels in active:
            for label in labels:
                counts[label] = counts.get(label, 0) + 1
        pair = None
        for left_index in range(len(active)):
            for right_index in range(left_index + 1, len(active)):
                shared = set(active[left_index][1]) & set(active[right_index][1])
                if shared:
                    pair = (left_index, right_index)
                    break
            if pair is not None:
                break
        if pair is None:
            pair = (0, 1)
        left_index, right_index = pair
        left_tensor, left_labels = active[left_index]
        right_tensor, right_labels = active[right_index]
        pair_counts = dict(counts)
        for label in left_labels + right_labels:
            pair_counts[label] -= 1
        pair_outputs = tuple(
            label
            for label in tuple(dict.fromkeys(left_labels + right_labels))
            if label in final_outputs or pair_counts.get(label, 0) > 0
        )
        tensor = _jax_einsum_by_labels(
            left_tensor,
            left_labels,
            right_tensor,
            right_labels,
            pair_outputs,
            matmul_precision,
        )
        for index in sorted((left_index, right_index), reverse=True):
            active.pop(index)
        active.append((tensor, pair_outputs))
    final_tensor, final_labels = active[0]
    if final_labels != tuple(output_labels):
        final_tensor = _jax_einsum_reorder(
            final_tensor, final_labels, output_labels, matmul_precision
        )
    return final_tensor


def _jax_einsum_by_labels(
    left_tensor: Any,
    left_labels: tuple[int, ...],
    right_tensor: Any,
    right_labels: tuple[int, ...],
    output_labels: tuple[int, ...],
    matmul_precision: str | None,
) -> Any:
    import jax.numpy as jnp

    labels = tuple(dict.fromkeys(left_labels + right_labels + output_labels))
    if len(labels) > len(_JAX_EINSUM_CHARS):
        raise ValueError(
            "JAX tensor_network pair contraction exceeded local einsum label capacity."
        )
    mapping = {label: _JAX_EINSUM_CHARS[index] for index, label in enumerate(labels)}
    equation = (
        "".join(mapping[label] for label in left_labels)
        + ","
        + "".join(mapping[label] for label in right_labels)
        + "->"
        + "".join(mapping[label] for label in output_labels)
    )
    return jnp.einsum(equation, left_tensor, right_tensor, precision=matmul_precision)


def _jax_einsum_reorder(
    tensor: Any,
    labels: tuple[int, ...],
    output_labels: tuple[int, ...],
    matmul_precision: str | None,
) -> Any:
    import jax.numpy as jnp

    all_labels = tuple(dict.fromkeys(labels + output_labels))
    if len(all_labels) > len(_JAX_EINSUM_CHARS):
        raise ValueError(
            "JAX tensor_network final contraction exceeded local einsum label capacity."
        )
    mapping = {
        label: _JAX_EINSUM_CHARS[index] for index, label in enumerate(all_labels)
    }
    equation = (
        "".join(mapping[label] for label in labels)
        + "->"
        + "".join(mapping[label] for label in output_labels)
    )
    return jnp.einsum(equation, tensor, precision=matmul_precision)
