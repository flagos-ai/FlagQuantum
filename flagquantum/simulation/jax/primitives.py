"""Shared JAX numerical primitives exposed through the PyTorch interface."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextvars import ContextVar
from typing import Any

TorchCircuitBuilder = Callable[..., Any]


_ACTIVE_JAX_COMPUTE_DTYPE: ContextVar[str] = ContextVar(
    "flagquantum_jax_compute_dtype", default="complex64"
)
_ACTIVE_JAX_BATCH_INDEX: ContextVar[Any] = ContextVar(
    "flagquantum_jax_batch_index", default=None
)


def _set_active_jax_compute_dtype(compute_dtype: str) -> str:
    previous = _ACTIVE_JAX_COMPUTE_DTYPE.get()
    _ACTIVE_JAX_COMPUTE_DTYPE.set(str(compute_dtype))
    return previous


def _jax_complex_dtype() -> Any:
    import jax.numpy as jnp

    return (
        jnp.complex128
        if _ACTIVE_JAX_COMPUTE_DTYPE.get() == "complex128"
        else jnp.complex64
    )


def _jax_real_dtype() -> Any:
    import jax.numpy as jnp

    return (
        jnp.float64 if _ACTIVE_JAX_COMPUTE_DTYPE.get() == "complex128" else jnp.float32
    )


def _jax_scalar_param(value: Any) -> Any:
    import jax.numpy as jnp

    tensor = jnp.asarray(value)
    index = _ACTIVE_JAX_BATCH_INDEX.get()
    if index is not None and tensor.ndim == 1:
        tensor = tensor[0 if tensor.shape[0] == 1 else index]
    return tensor.reshape(())


def jax_evaluate_declared_batch(circuit: Any, evaluate: Callable[[], Any]) -> Any:
    """Vectorize only the explicit Circuit batch and its declared angle rows."""
    import jax
    import jax.numpy as jnp

    from ...errors import ValidationError

    size = int(circuit.bsz)
    if size == 1:
        return evaluate()
    for instruction in circuit.to_ir():
        for name, value in instruction.params.items():
            shape = jnp.shape(value)
            if len(shape) > 1 or (shape and shape[0] not in (1, size)):
                raise ValidationError(
                    f"Gate {instruction.name!r} parameter {name!r} has shape {shape}; "
                    f"expected a scalar, a vector of length 1, or a vector of length "
                    f"{size} for a batch of {size}"
                )

    def entry(index: Any) -> Any:
        token = _ACTIVE_JAX_BATCH_INDEX.set(index)
        try:
            return evaluate()
        finally:
            _ACTIVE_JAX_BATCH_INDEX.reset(token)

    return jax.vmap(entry)(jnp.arange(size))


def _jax_apply_matrix(
    state: Any,
    matrix: Any,
    wires: tuple[int, ...],
    n_wires: int,
    matmul_precision: str | None = "highest",
) -> Any:
    import jax.numpy as jnp

    k = len(wires)
    if k == 1:
        wire = wires[0]
        paired = state.reshape(2**wire, 2, 2 ** (n_wires - wire - 1))
        zero, one = paired[:, 0, :], paired[:, 1, :]
        return jnp.stack(
            (
                matrix[0, 0] * zero + matrix[0, 1] * one,
                matrix[1, 0] * zero + matrix[1, 1] * one,
            ),
            axis=1,
        ).reshape(-1)
    dim = 2**k
    rest = tuple(wire for wire in range(n_wires) if wire not in wires)
    perm = tuple(wires) + rest
    inv_perm = [0] * len(perm)
    for index, axis in enumerate(perm):
        inv_perm[axis] = index
    tensor = state.reshape((2,) * n_wires).transpose(perm)
    flat = tensor.reshape(dim, -1)
    out = jnp.matmul(matrix, flat, precision=matmul_precision)
    return out.reshape((2,) * n_wires).transpose(tuple(inv_perm)).reshape(-1)


def _jax_z_values(state: Any, n_wires: int, wires: Iterable[int]) -> Any:
    import jax.numpy as jnp

    probs = jnp.abs(state) ** 2
    indices = jnp.arange(2**n_wires)
    values = []
    for wire in wires:
        bits = (indices >> (n_wires - 1 - int(wire))) & 1
        weights = 1.0 - 2.0 * bits.astype(probs.dtype)
        values.append(jnp.sum(probs * weights))
    return jnp.stack(values)


def _jax_z_sum(state: Any, n_wires: int, wires: Iterable[int]) -> Any:
    import jax.numpy as jnp

    selected = tuple(wires)
    if not selected:
        return jnp.sum(_jax_z_values(state, n_wires, selected))
    probabilities = jnp.abs(state) ** 2
    indices = jnp.arange(2**n_wires)
    diagonal = jnp.zeros_like(probabilities)
    for wire in selected:
        bits = (indices >> (n_wires - 1 - int(wire))) & 1
        diagonal = diagonal + 1 - 2 * bits.astype(probabilities.dtype)
    return jnp.sum(probabilities * diagonal)


def _jax_pauli_string_expectation(
    state: Any,
    n_wires: int,
    ops: tuple[tuple[int, str], ...],
    matmul_precision: str | None = "highest",
) -> Any:
    import jax.numpy as jnp

    if not ops:
        return jnp.ones((), dtype=_jax_real_dtype())
    transformed = state
    for wire, name in ops:
        transformed = _jax_apply_matrix(
            transformed,
            _jax_pauli_matrix(name),
            (int(wire),),
            n_wires,
            matmul_precision,
        )
    return jnp.real(jnp.sum(jnp.conj(state) * transformed))


def _jax_hamiltonian_expectation(
    state: Any,
    n_wires: int,
    terms: tuple[tuple[float, tuple[tuple[int, str], ...]], ...],
    matmul_precision: str | None = "highest",
) -> Any:
    import jax.numpy as jnp

    total = jnp.zeros((), dtype=_jax_real_dtype())
    for coefficient, ops in terms:
        total = total + jnp.asarray(
            coefficient, dtype=_jax_real_dtype()
        ) * _jax_pauli_string_expectation(
            state,
            n_wires,
            ops,
            matmul_precision,
        )
    return total


def _jax_pauli_matrix(name: str, *, dtype: Any | None = None) -> Any:
    normalized = str(name).lower()
    if normalized == "x":
        matrix = _jax_x()
    elif normalized == "y":
        matrix = _jax_y()
    elif normalized == "z":
        matrix = _jax_z()
    elif normalized == "i":
        import jax.numpy as jnp

        matrix = jnp.eye(2, dtype=_jax_complex_dtype())
    else:
        raise ValueError(f"Unsupported Pauli operator {name!r}.")
    return matrix if dtype is None else matrix.astype(dtype)


def _jax_rx(theta: Any) -> Any:
    import jax.numpy as jnp

    c = jnp.cos(theta / 2)
    s = jnp.sin(theta / 2)
    return jnp.array([[c, -1j * s], [-1j * s, c]], dtype=_jax_complex_dtype())


def _jax_ry(theta: Any) -> Any:
    import jax.numpy as jnp

    c = jnp.cos(theta / 2)
    s = jnp.sin(theta / 2)
    return jnp.array([[c, -s], [s, c]], dtype=_jax_complex_dtype())


def _jax_rz(theta: Any) -> Any:
    import jax.numpy as jnp

    return _jax_diag2(jnp.exp(-0.5j * theta), jnp.exp(0.5j * theta))


def _jax_phase(theta: Any) -> Any:
    import jax.numpy as jnp

    return _jax_diag2(1.0 + 0.0j, jnp.exp(1j * theta))


def _jax_u2(phi: Any, lbd: Any) -> Any:
    import jax.numpy as jnp

    scale = 1.0 / jnp.sqrt(2.0)
    return jnp.stack(
        (
            jnp.stack((scale + 0.0j, -scale * jnp.exp(1j * lbd))),
            jnp.stack((scale * jnp.exp(1j * phi), scale * jnp.exp(1j * (phi + lbd)))),
        )
    ).astype(_jax_complex_dtype())


def _jax_u3(theta: Any, phi: Any, lbd: Any) -> Any:
    import jax.numpy as jnp

    c = jnp.cos(theta / 2)
    s = jnp.sin(theta / 2)
    return jnp.stack(
        (
            jnp.stack((c + 0.0j, -jnp.exp(1j * lbd) * s)),
            jnp.stack((jnp.exp(1j * phi) * s, jnp.exp(1j * (phi + lbd)) * c)),
        )
    ).astype(_jax_complex_dtype())


def _jax_h() -> Any:
    import jax.numpy as jnp

    return jnp.asarray([[1, 1], [1, -1]], dtype=_jax_complex_dtype()) / jnp.sqrt(2.0)


def _jax_x() -> Any:
    import jax.numpy as jnp

    return jnp.asarray([[0, 1], [1, 0]], dtype=_jax_complex_dtype())


def _jax_y() -> Any:
    import jax.numpy as jnp

    return jnp.asarray([[0, -1j], [1j, 0]], dtype=_jax_complex_dtype())


def _jax_z() -> Any:
    import jax.numpy as jnp

    return jnp.asarray([[1, 0], [0, -1]], dtype=_jax_complex_dtype())


def _jax_s() -> Any:

    return _jax_diag2(1.0 + 0.0j, 1j)


def _jax_sdg() -> Any:

    return _jax_diag2(1.0 + 0.0j, -1j)


def _jax_t() -> Any:
    import jax.numpy as jnp

    return _jax_diag2(1.0 + 0.0j, jnp.exp(0.25j * jnp.pi))


def _jax_tdg() -> Any:
    import jax.numpy as jnp

    return _jax_diag2(1.0 + 0.0j, jnp.exp(-0.25j * jnp.pi))


def _jax_sx() -> Any:
    import jax.numpy as jnp

    return jnp.asarray(
        [[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]],
        dtype=_jax_complex_dtype(),
    )


def _jax_sxdg() -> Any:
    import jax.numpy as jnp

    return jnp.asarray(
        [[0.5 - 0.5j, 0.5 + 0.5j], [0.5 + 0.5j, 0.5 - 0.5j]],
        dtype=_jax_complex_dtype(),
    )


def _jax_cx() -> Any:
    import jax.numpy as jnp

    return jnp.asarray(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=_jax_complex_dtype(),
    )


def _jax_cz() -> Any:
    import jax.numpy as jnp

    return jnp.asarray(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]],
        dtype=_jax_complex_dtype(),
    )


def _jax_cy() -> Any:
    import jax.numpy as jnp

    return jnp.asarray(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]],
        dtype=_jax_complex_dtype(),
    )


def _jax_swap() -> Any:
    import jax.numpy as jnp

    return jnp.asarray(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=_jax_complex_dtype(),
    )


def _jax_controlled(target: Any) -> Any:
    import jax.numpy as jnp

    zero = jnp.zeros((), dtype=_jax_complex_dtype())
    one = jnp.ones((), dtype=_jax_complex_dtype())
    return jnp.stack(
        (
            jnp.stack((one, zero, zero, zero)),
            jnp.stack((zero, one, zero, zero)),
            jnp.stack((zero, zero, target[0, 0], target[0, 1])),
            jnp.stack((zero, zero, target[1, 0], target[1, 1])),
        )
    )


def _jax_cphase(theta: Any) -> Any:
    import jax.numpy as jnp

    return _jax_diag4(1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, jnp.exp(1j * theta))


def _jax_rxx(theta: Any) -> Any:
    import jax.numpy as jnp

    c = jnp.cos(theta / 2)
    v = -1j * jnp.sin(theta / 2)
    z = jnp.zeros((), dtype=_jax_complex_dtype())
    return jnp.stack(
        (
            jnp.stack((c + 0.0j, z, z, v)),
            jnp.stack((z, c + 0.0j, v, z)),
            jnp.stack((z, v, c + 0.0j, z)),
            jnp.stack((v, z, z, c + 0.0j)),
        )
    )


def _jax_ryy(theta: Any) -> Any:
    import jax.numpy as jnp

    c = jnp.cos(theta / 2)
    s = jnp.sin(theta / 2)
    z = jnp.zeros((), dtype=_jax_complex_dtype())
    return jnp.stack(
        (
            jnp.stack((c + 0.0j, z, z, 1j * s)),
            jnp.stack((z, c + 0.0j, -1j * s, z)),
            jnp.stack((z, -1j * s, c + 0.0j, z)),
            jnp.stack((1j * s, z, z, c + 0.0j)),
        )
    )


def _jax_rzz(theta: Any) -> Any:
    import jax.numpy as jnp

    exp_neg = jnp.exp(-0.5j * theta)
    exp_pos = jnp.exp(0.5j * theta)
    return _jax_diag4(exp_neg, exp_pos, exp_pos, exp_neg)


def _jax_diag2(a: Any, d: Any) -> Any:
    import jax.numpy as jnp

    zero = jnp.zeros((), dtype=_jax_complex_dtype())
    return jnp.stack((jnp.stack((a, zero)), jnp.stack((zero, d)))).astype(
        _jax_complex_dtype()
    )


def _jax_diag4(a: Any, b: Any, c: Any, d: Any) -> Any:
    import jax.numpy as jnp

    zero = jnp.zeros((), dtype=_jax_complex_dtype())
    return jnp.stack(
        (
            jnp.stack((a, zero, zero, zero)),
            jnp.stack((zero, b, zero, zero)),
            jnp.stack((zero, zero, c, zero)),
            jnp.stack((zero, zero, zero, d)),
        )
    ).astype(_jax_complex_dtype())


def _jax_instruction_matrix(instruction: Any) -> Any:
    name = instruction.name
    if name == "rx":
        return _jax_rx(_jax_scalar_param(instruction.params["theta"]))
    if name == "ry":
        return _jax_ry(_jax_scalar_param(instruction.params["theta"]))
    if name == "rz":
        return _jax_rz(_jax_scalar_param(instruction.params["theta"]))
    if name in {"phase", "u1"}:
        return _jax_phase(_jax_scalar_param(instruction.params["theta"]))
    if name == "u2":
        return _jax_u2(
            _jax_scalar_param(instruction.params["phi"]),
            _jax_scalar_param(instruction.params["lbd"]),
        )
    if name == "u3":
        return _jax_u3(
            _jax_scalar_param(instruction.params["theta"]),
            _jax_scalar_param(instruction.params["phi"]),
            _jax_scalar_param(instruction.params["lbd"]),
        )
    if name == "h":
        return _jax_h()
    if name == "x":
        return _jax_x()
    if name == "y":
        return _jax_y()
    if name == "z":
        return _jax_z()
    if name == "s":
        return _jax_s()
    if name == "sdg":
        return _jax_sdg()
    if name == "t":
        return _jax_t()
    if name == "tdg":
        return _jax_tdg()
    if name == "sx":
        return _jax_sx()
    if name == "sxdg":
        return _jax_sxdg()
    if name == "cx":
        return _jax_cx()
    if name == "cz":
        return _jax_cz()
    if name == "cy":
        return _jax_cy()
    if name == "swap":
        return _jax_swap()
    if name == "crx":
        return _jax_controlled(_jax_rx(_jax_scalar_param(instruction.params["theta"])))
    if name == "cry":
        return _jax_controlled(_jax_ry(_jax_scalar_param(instruction.params["theta"])))
    if name == "crz":
        return _jax_controlled(_jax_rz(_jax_scalar_param(instruction.params["theta"])))
    if name == "cphase":
        return _jax_cphase(_jax_scalar_param(instruction.params["theta"]))
    if name == "rxx":
        return _jax_rxx(_jax_scalar_param(instruction.params["theta"]))
    if name == "ryy":
        return _jax_ryy(_jax_scalar_param(instruction.params["theta"]))
    if name == "rzz":
        return _jax_rzz(_jax_scalar_param(instruction.params["theta"]))
    raise NotImplementedError(f"JAX quantum kernel does not support gate {name!r} yet.")


def jax_circuit_tensor_nodes(
    circuit: Any,
    n_qubits: int,
    *,
    start_label: int = 0,
    conjugate: bool = False,
    matmul_precision: str | None = "highest",
) -> tuple[list[tuple[Any, tuple[int, ...]]], list[int], int]:
    import jax.numpy as jnp

    next_label = int(start_label)
    current_labels = []
    nodes: list[tuple[Any, tuple[int, ...]]] = []
    producers: list[int] = []
    zero = jnp.asarray([1.0 + 0.0j, 0.0 + 0.0j], dtype=_jax_complex_dtype())
    for _qubit in range(int(n_qubits)):
        label = next_label
        next_label += 1
        current_labels.append(label)
        nodes.append((zero, (label,)))
        producers.append(len(nodes) - 1)
    for instruction in circuit.to_ir():
        if instruction.metadata.get("is_channel"):
            raise NotImplementedError(
                "JAX tensor_network mode currently supports unitary circuit instructions."
            )
        qubits = tuple(int(qubit) for qubit in instruction.wires)
        matrix = _jax_instruction_matrix(instruction).reshape((2,) * (2 * len(qubits)))
        if conjugate:
            matrix = jnp.conj(matrix)
        if len(qubits) == 1:
            producer = producers[qubits[0]]
            previous, labels = nodes[producer]
            if len(labels) <= 2:
                nodes[producer] = (
                    jnp.matmul(matrix, previous, precision=matmul_precision),
                    labels,
                )
                continue
        input_labels = tuple(current_labels[qubit] for qubit in qubits)
        output_labels = tuple(range(next_label, next_label + len(qubits)))
        next_label += len(qubits)
        for qubit, label in zip(qubits, output_labels, strict=True):
            current_labels[qubit] = label
        nodes.append((matrix, output_labels + input_labels))
        for qubit in qubits:
            producers[qubit] = len(nodes) - 1
    return nodes, current_labels, next_label


def jax_contract_tensor_nodes(
    nodes: list[tuple[Any, tuple[int, ...]]],
    output_labels: tuple[int, ...],
    matmul_precision: str | None,
    *,
    path: Any = "greedy",
) -> Any:
    import jax.numpy as jnp

    operands: list[Any] = []
    for tensor, labels in nodes:
        operands.extend((tensor, labels))
    operands.append(tuple(output_labels))
    return jnp.einsum(*operands, optimize=path, precision=matmul_precision)
