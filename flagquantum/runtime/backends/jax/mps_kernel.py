"""Canonical JAX kernels exposed through the PyTorch interface."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from typing import Any

TorchCircuitBuilder = Callable[..., Any]


_ACTIVE_JAX_COMPUTE_DTYPE: ContextVar[str] = ContextVar(
    "flagquantum_jax_compute_dtype", default="complex64"
)


from ....simulation.jax_gate_primitives import (  # noqa: E402
    _jax_complex_dtype,
    _jax_cx,
    _jax_instruction_matrix,
    _jax_pauli_matrix,
    _jax_real_dtype,
)
from ....simulation.jax_mps import (  # noqa: E402
    jax_mps_apply_one as _jax_mps_apply_one,
)
from ....simulation.jax_mps import (  # noqa: E402
    jax_mps_apply_two_remote as _jax_mps_apply_two_remote,
)
from ....simulation.jax_mps import (  # noqa: E402
    jax_mps_pauli_string_expectation as _jax_mps_pauli_string_expectation,
)
from ....simulation.jax_mps import (  # noqa: E402
    jax_mps_split_pair as _jax_mps_split_pair,
)
from ....simulation.jax_mps import (  # noqa: E402
    jax_mps_to_statevector as _jax_mps_to_statevector,
)
from ....simulation.jax_mps import (  # noqa: E402
    jax_mps_zz_z_chain_expectation_padded_scan as _jax_mps_zz_z_chain_expectation_padded_scan,
)
from ....simulation.jax_mps import (  # noqa: E402
    parse_zz_z_chain_hamiltonian as _parse_zz_z_chain_hamiltonian,
)


def _jax_mps_from_circuit(
    circuit: Any,
    n_wires: int,
    parameters: Any,
    *,
    max_bond: int | None = None,
    cutoff: float = 0.0,
    matmul_precision: str | None = "highest",
) -> Any:
    import jax.numpy as jnp

    del parameters
    structured = _jax_mps_from_nearest_cx_layers(
        circuit,
        int(n_wires),
        max_bond=max_bond,
        cutoff=cutoff,
        matmul_precision=matmul_precision,
    )
    if structured is not None:
        return structured

    tensors = []
    for _ in range(int(n_wires)):
        tensor = jnp.zeros((1, 2, 1), dtype=_jax_complex_dtype())
        tensor = tensor.at[0, 0, 0].set(1.0 + 0.0j)
        tensors.append(tensor)
    pending_one_qubit: list[Any | None] = [None for _ in range(int(n_wires))]

    def flush_wire(wire: int) -> None:
        matrix = pending_one_qubit[int(wire)]
        if matrix is not None:
            tensors[int(wire)] = _jax_mps_apply_one(
                tensors[int(wire)], matrix, matmul_precision
            )
            pending_one_qubit[int(wire)] = None

    def flush_all_one_qubit() -> None:
        for wire in range(int(n_wires)):
            flush_wire(wire)

    for instruction in circuit.to_ir():
        if instruction.metadata.get("is_channel"):
            raise NotImplementedError(
                "JAX MPS mode currently supports unitary circuit instructions."
            )
        matrix = _jax_instruction_matrix(instruction)
        wires = tuple(int(wire) for wire in instruction.wires)
        if len(wires) == 1:
            wire = wires[0]
            pending = pending_one_qubit[wire]
            pending_one_qubit[wire] = (
                matrix
                if pending is None
                else jnp.matmul(matrix, pending, precision=matmul_precision)
            )
        elif len(wires) == 2:
            flush_all_one_qubit()
            tensors = _jax_mps_apply_two_remote(
                tensors,
                matrix,
                wires,
                max_bond=max_bond,
                cutoff=cutoff,
                matmul_precision=matmul_precision,
            )
        else:
            flush_all_one_qubit()
            raise NotImplementedError(
                "JAX MPS mode does not support gates acting on more than two wires without dense fallback."
            )
    flush_all_one_qubit()
    return tensors


def _jax_mps_from_nearest_cx_layers(
    circuit: Any,
    n_wires: int,
    *,
    max_bond: int | None,
    cutoff: float,
    matmul_precision: str | None,
) -> Any | None:
    if os.environ.get("FQ_DISABLE_JAX_MPS_CX_SCAN", "").lower() in {"1", "true", "yes"}:
        return None
    parsed = _parse_nearest_cx_layers(circuit, int(n_wires), matmul_precision)
    if parsed is None or max_bond is None or float(cutoff) > 0.0:
        return None

    import jax

    local_layers, two_qubit_matrix = parsed
    if not local_layers:
        return None
    layer_bond_limit = max(1, 2 ** len(local_layers))
    if int(max_bond) > layer_bond_limit:
        return None
    bond_dim = max(1, int(max_bond))
    tensors = _jax_mps_initial_padded_stack(int(n_wires), bond_dim)

    del jax
    for layer_index, local_matrices in enumerate(local_layers):
        tensors = _jax_mps_apply_local_stack(tensors, local_matrices, matmul_precision)
        if int(n_wires) > 1:
            tensors = _jax_mps_apply_adjacent_chain_scan(
                tensors,
                two_qubit_matrix,
                max_bond=bond_dim,
                layer_index=int(layer_index),
                matmul_precision=matmul_precision,
            )
            tensors = _jax_mps_project_open_boundaries(tensors)
    return [tensors[index] for index in range(int(n_wires))]


def _parse_nearest_cx_layers(
    circuit: Any,
    n_wires: int,
    matmul_precision: str | None,
) -> tuple[list[Any], Any] | None:
    import jax.numpy as jnp

    instructions = list(circuit.to_ir())
    if not instructions:
        return None
    layers = []
    index = 0
    while index < len(instructions):
        local = [_jax_pauli_matrix("i") for _ in range(int(n_wires))]
        saw_local = False
        while index < len(instructions) and len(tuple(instructions[index].wires)) == 1:
            instruction = instructions[index]
            if instruction.metadata.get("is_channel"):
                return None
            wire = int(tuple(instruction.wires)[0])
            if wire < 0 or wire >= int(n_wires):
                return None
            matrix = _jax_instruction_matrix(instruction)
            local[wire] = jnp.matmul(matrix, local[wire], precision=matmul_precision)
            saw_local = True
            index += 1
        if not saw_local:
            return None
        for wire in range(int(n_wires) - 1):
            if index >= len(instructions):
                return None
            instruction = instructions[index]
            wires = tuple(int(item) for item in instruction.wires)
            if (
                instruction.metadata.get("is_channel")
                or str(instruction.name).lower() != "cx"
                or wires != (wire, wire + 1)
            ):
                return None
            index += 1
        layers.append(jnp.stack(local))
    return layers, _jax_cx()


def _jax_mps_initial_padded_stack(n_wires: int, bond_dim: int) -> Any:
    import jax.numpy as jnp

    tensor = jnp.zeros((int(bond_dim), 2, int(bond_dim)), dtype=_jax_complex_dtype())
    tensor = tensor.at[0, 0, 0].set(1.0 + 0.0j)
    return jnp.broadcast_to(
        tensor, (int(n_wires), int(bond_dim), 2, int(bond_dim))
    ).copy()


def _jax_mps_project_open_boundaries(tensors: Any) -> Any:
    import jax.numpy as jnp

    left_mask = (
        jnp.zeros((tensors.shape[1], 1, 1), dtype=tensors.dtype)
        .at[0, 0, 0]
        .set(1.0 + 0.0j)
    )
    right_mask = (
        jnp.zeros((1, 1, tensors.shape[3]), dtype=tensors.dtype)
        .at[0, 0, 0]
        .set(1.0 + 0.0j)
    )
    tensors = tensors.at[0].set(tensors[0] * left_mask)
    tensors = tensors.at[-1].set(tensors[-1] * right_mask)
    return tensors


def _jax_mps_apply_local_stack(
    tensors: Any, matrices: Any, matmul_precision: str | None
) -> Any:
    import jax.numpy as jnp

    return jnp.einsum("wpq,wlqr->wlpr", matrices, tensors, precision=matmul_precision)


def _jax_mps_apply_adjacent_chain_scan(
    tensors: Any,
    matrix: Any,
    *,
    max_bond: int,
    layer_index: int,
    matmul_precision: str | None,
) -> Any:
    import jax
    import jax.numpy as jnp

    n_wires = int(tensors.shape[0])
    if n_wires <= 1:
        return tensors

    def rank_before(bond: int) -> int:
        if int(layer_index) <= 0:
            return 1
        return min(
            int(max_bond),
            2 ** min(int(layer_index), int(bond) + 1, n_wires - 1 - int(bond)),
        )

    def rank_after(bond: int) -> int:
        return min(
            int(max_bond),
            2 ** min(int(layer_index) + 1, int(bond) + 1, n_wires - 1 - int(bond)),
        )

    def spec(wire: int) -> tuple[int, int, int, int]:
        left_active = 1 if int(wire) == 0 else rank_after(int(wire) - 1)
        shared_active = rank_before(int(wire))
        right_active = 1 if int(wire) == n_wires - 2 else rank_before(int(wire) + 1)
        out_rank = rank_after(int(wire))
        return left_active, shared_active, right_active, out_rank

    def apply_one(
        left_tensor: Any, right_tensor: Any, current_spec: tuple[int, int, int, int]
    ) -> tuple[Any, Any]:
        left_active, shared_active, right_active, out_rank = current_spec
        return _jax_mps_apply_two_adjacent_active(
            left_tensor,
            right_tensor,
            matrix,
            max_bond=int(max_bond),
            left_active=left_active,
            shared_active=shared_active,
            right_active=right_active,
            out_rank=out_rank,
            matmul_precision=matmul_precision,
        )

    pieces = []
    carry = tensors[0]
    wire = 0
    while wire < n_wires - 1:
        current_spec = spec(wire)
        run_end = wire + 1
        if current_spec[0] == current_spec[3]:
            while run_end < n_wires - 1 and spec(run_end) == current_spec:
                run_end += 1
        if run_end - wire > 1:

            def step(left_tensor: Any, right_tensor: Any) -> tuple[Any, Any]:
                next_left, next_right = apply_one(
                    left_tensor, right_tensor, current_spec
                )
                return next_right, next_left

            carry, completed = jax.lax.scan(
                step, carry, tensors[wire + 1 : run_end + 1]
            )
            pieces.append(completed)
            wire = run_end
            continue

        next_left, carry = apply_one(carry, tensors[wire + 1], current_spec)
        pieces.append(next_left[None, :, :, :])
        wire += 1

    pieces.append(carry[None, :, :, :])
    return jnp.concatenate(pieces, axis=0)


def _jax_mps_apply_two_adjacent_active(
    left: Any,
    right: Any,
    matrix: Any,
    *,
    max_bond: int,
    left_active: int,
    shared_active: int,
    right_active: int,
    out_rank: int,
    matmul_precision: str | None,
) -> tuple[Any, Any]:
    import jax.numpy as jnp

    left = left[: int(left_active), :, : int(shared_active)]
    right = right[: int(shared_active), :, : int(right_active)]
    theta = jnp.einsum("lsm,mtr->lstr", left, right, precision=matmul_precision)
    theta = theta.reshape(int(left_active), 4, int(right_active))
    flat = theta.transpose(1, 0, 2).reshape(4, -1)
    applied = jnp.matmul(matrix, flat, precision=matmul_precision)
    theta = applied.reshape(4, int(left_active), int(right_active)).transpose(1, 0, 2)
    theta = theta.reshape(int(left_active), 2, 2, int(right_active))
    next_left, next_right = _jax_mps_split_pair(
        theta, max_bond=int(out_rank), cutoff=0.0
    )
    next_left = jnp.pad(
        next_left,
        (
            (0, int(max_bond) - int(left_active)),
            (0, 0),
            (0, int(max_bond) - int(out_rank)),
        ),
    )
    next_right = jnp.pad(
        next_right,
        (
            (0, int(max_bond) - int(out_rank)),
            (0, 0),
            (0, int(max_bond) - int(right_active)),
        ),
    )
    return next_left, next_right


def _jax_mps_statevector_from_circuit(
    circuit: Any,
    n_wires: int,
    parameters: Any,
    *,
    max_bond: int | None = None,
    cutoff: float = 0.0,
    matmul_precision: str | None = "highest",
) -> Any:
    tensors = _jax_mps_from_circuit(
        circuit,
        n_wires,
        parameters,
        max_bond=max_bond,
        cutoff=cutoff,
        matmul_precision=matmul_precision,
    )
    return _jax_mps_to_statevector(tensors, matmul_precision)


def _jax_mps_zz_z_chain_expectation(
    tensors: Sequence[Any],
    terms: tuple[tuple[float, tuple[tuple[int, str], ...]], ...],
    matmul_precision: str | None,
) -> Any | None:
    parsed = _parse_zz_z_chain_hamiltonian(terms, len(tensors))
    if parsed is None:
        return None

    return _jax_mps_zz_z_chain_expectation_padded_scan(
        tensors, parsed, matmul_precision
    )


def _jax_mps_hamiltonian_expectation(
    tensors: Sequence[Any],
    terms: tuple[tuple[float, tuple[tuple[int, str], ...]], ...],
    matmul_precision: str | None,
) -> Any:
    import jax.numpy as jnp

    fast_value = _jax_mps_zz_z_chain_expectation(tensors, terms, matmul_precision)
    if fast_value is not None:
        return fast_value

    total = jnp.zeros((), dtype=_jax_real_dtype())
    for coefficient, ops in terms:
        total = total + jnp.asarray(
            coefficient, dtype=_jax_real_dtype()
        ) * _jax_mps_pauli_string_expectation(
            tensors,
            ops,
            matmul_precision,
        )
    return total
