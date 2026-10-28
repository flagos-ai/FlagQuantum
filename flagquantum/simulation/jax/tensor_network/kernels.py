"""Dependency-light local JAX tensor-network contraction."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ..primitives import (
    _jax_pauli_matrix,
    _jax_real_dtype,
    jax_circuit_tensor_nodes,
    jax_contract_tensor_nodes,
)

_JAX_EINSUM_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def jax_tensor_network_expectation_product_ops(
    circuit: Any,
    n_qubits: int,
    ops: dict[int, Any],
    matmul_precision: str | None,
) -> Any:
    import jax.numpy as jnp

    ket_nodes, ket_labels, next_label = jax_circuit_tensor_nodes(
        circuit,
        n_qubits,
        start_label=0,
        conjugate=False,
        matmul_precision=matmul_precision,
    )
    bra_nodes, bra_labels, _next_label = jax_circuit_tensor_nodes(
        circuit,
        n_qubits,
        start_label=next_label,
        conjugate=True,
        matmul_precision=matmul_precision,
    )
    identity = _jax_pauli_matrix("i")
    nodes = bra_nodes + ket_nodes
    for qubit in range(int(n_qubits)):
        op = ops.get(int(qubit), identity)
        nodes.append((op, (int(bra_labels[qubit]), int(ket_labels[qubit]))))
    value = jax_contract_tensor_nodes(nodes, (), matmul_precision)
    return jnp.real(value.reshape(()))


def jax_tensor_network_z_values(
    circuit: Any,
    n_qubits: int,
    qubits: Iterable[int],
    matmul_precision: str | None,
) -> Any:
    import jax.numpy as jnp

    z_op = _jax_pauli_matrix("z")
    values = []
    for qubit in qubits:
        values.append(
            jax_tensor_network_expectation_product_ops(
                circuit,
                n_qubits,
                {int(qubit): z_op},
                matmul_precision,
            )
        )
    return jnp.stack(values) if values else jnp.zeros((0,), dtype=_jax_real_dtype())


def jax_tensor_network_z_sum(
    circuit: Any,
    n_qubits: int,
    qubits: Iterable[int],
    matmul_precision: str | None,
) -> Any:
    import jax.numpy as jnp

    selected = tuple(int(qubit) for qubit in qubits)
    if not selected:
        return jnp.zeros((), dtype=_jax_real_dtype())
    if any(qubit < 0 or qubit >= n_qubits for qubit in selected):
        return jnp.sum(
            jax_tensor_network_z_values(circuit, n_qubits, selected, matmul_precision)
        )
    z = _jax_pauli_matrix("z")
    if n_qubits == 1:
        return jax_tensor_network_expectation_product_ops(
            circuit, n_qubits, {0: len(selected) * z}, matmul_precision
        )
    ket, ket_labels, next_label = jax_circuit_tensor_nodes(
        circuit, n_qubits, matmul_precision=matmul_precision
    )
    bra, bra_labels, next_label = jax_circuit_tensor_nodes(
        circuit,
        n_qubits,
        start_label=next_label,
        conjugate=True,
        matmul_precision=matmul_precision,
    )
    nodes = bra + ket
    identity = _jax_pauli_matrix("i")
    zero = jnp.zeros_like(identity)
    bonds = tuple(range(next_label, next_label + n_qubits - 1))
    for qubit in range(n_qubits):
        local = selected.count(qubit) * z
        physical = (bra_labels[qubit], ket_labels[qubit])
        if qubit == 0:
            tensor = jnp.stack((local, identity))
            labels: tuple[int, ...] = (bonds[0],) + physical
        elif qubit == n_qubits - 1:
            tensor = jnp.stack((identity, local))
            labels = (bonds[-1],) + physical
        else:
            tensor = jnp.stack(
                (jnp.stack((identity, zero)), jnp.stack((local, identity)))
            )
            labels = (bonds[qubit - 1], bonds[qubit]) + physical
        nodes.append((tensor, labels))
    # Bond dimension two represents sum_i Z_i without one network per term.
    return jnp.real(jax_contract_tensor_nodes(nodes, (), matmul_precision).reshape(()))


def jax_tensor_network_pauli_string_expectation(
    circuit: Any,
    n_qubits: int,
    ops: tuple[tuple[int, str], ...],
    matmul_precision: str | None,
) -> Any:
    op_map = {}
    for qubit, name in ops:
        normalized = str(name).lower()
        if normalized != "i":
            op_map[int(qubit)] = _jax_pauli_matrix(normalized)
    return jax_tensor_network_expectation_product_ops(
        circuit, n_qubits, op_map, matmul_precision
    )


def jax_tensor_network_hamiltonian_expectation(
    circuit: Any,
    n_qubits: int,
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
            n_qubits,
            ops,
            matmul_precision,
        )
    return total


def jax_tensor_network_loss_from_output(
    output: Any,
    *,
    n_qubits: int,
    bsz: int,
    observable: str,
    observable_qubits: Iterable[int] | None,
) -> Any:
    """Evaluate a supported loss from a contracted statevector-shaped output."""

    import jax.numpy as jnp

    state = output.reshape(int(bsz), 2 ** int(n_qubits))
    if str(observable) == "state_norm":
        return jnp.real(jnp.sum(jnp.conj(state) * state))
    if str(observable) != "z_sum":
        raise ValueError(
            "JAX sliced TN reverse mode currently supports "
            "observable='z_sum' or 'state_norm'."
        )
    qubits = (
        tuple(range(int(n_qubits)))
        if observable_qubits is None
        else tuple(int(qubit) for qubit in observable_qubits)
    )
    probabilities = jnp.abs(state) ** 2
    indices = jnp.arange(2 ** int(n_qubits))
    total = jnp.zeros((), dtype=probabilities.real.dtype)
    for qubit in qubits:
        bits = (indices >> (int(n_qubits) - 1 - int(qubit))) & 1
        weights = 1.0 - 2.0 * bits.astype(probabilities.real.dtype)
        total = total + jnp.sum(probabilities * weights.reshape(1, -1))
    return total


def jax_slice_tensor_by_labels(
    tensor: Any,
    labels: Iterable[int],
    assignments: Mapping[int, int],
) -> tuple[Any, tuple[int, ...]]:
    """Select assigned tensor-network labels and return the remaining labels."""

    import jax.numpy as jnp

    remaining = [int(label) for label in labels]
    for label, value in assignments.items():
        normalized = int(label)
        if normalized not in remaining:
            continue
        axis = remaining.index(normalized)
        tensor = jnp.take(tensor, int(value), axis=axis)
        remaining.pop(axis)
    return tensor, tuple(remaining)


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
