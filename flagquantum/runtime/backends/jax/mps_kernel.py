"""Runtime-side circuit lowering for local JAX MPS execution."""

from __future__ import annotations

import os
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

TorchCircuitBuilder = Callable[..., Any]


_ACTIVE_JAX_COMPUTE_DTYPE: ContextVar[str] = ContextVar(
    "flagquantum_jax_compute_dtype", default="complex64"
)


from ....simulation.jax_gate_primitives import (  # noqa: E402
    _jax_cx,
    _jax_instruction_matrix,
    _jax_pauli_matrix,
)
from ....simulation.jax_mps import (  # noqa: E402
    jax_mps_apply_adjacent_chain_scan as _jax_mps_apply_adjacent_chain_scan,
)
from ....simulation.jax_mps import (  # noqa: E402
    jax_mps_apply_local_stack as _jax_mps_apply_local_stack,
)
from ....simulation.jax_mps import (  # noqa: E402
    jax_mps_apply_one as _jax_mps_apply_one,
)
from ....simulation.jax_mps import (  # noqa: E402
    jax_mps_apply_two_remote as _jax_mps_apply_two_remote,
)
from ....simulation.jax_mps import (  # noqa: E402
    jax_mps_initial_open_boundary_tensors as _jax_mps_initial_open_boundary_tensors,
)
from ....simulation.jax_mps import (  # noqa: E402
    jax_mps_initial_padded_stack as _jax_mps_initial_padded_stack,
)
from ....simulation.jax_mps import (  # noqa: E402
    jax_mps_project_open_boundaries as _jax_mps_project_open_boundaries,
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

    tensors = _jax_mps_initial_open_boundary_tensors(int(n_wires))
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

    local_layers, two_qubit_matrix = parsed
    if not local_layers:
        return None
    layer_bond_limit = max(1, 2 ** len(local_layers))
    if int(max_bond) > layer_bond_limit:
        return None
    bond_dim = max(1, int(max_bond))
    tensors = _jax_mps_initial_padded_stack(int(n_wires), bond_dim)

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
