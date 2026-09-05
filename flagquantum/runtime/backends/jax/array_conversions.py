"""Shared gate, shard, MPS split, and tensor conversion helpers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ....simulation.jax_statevector import (
    jax_basis_indices_for_wires as _jax_basis_indices_for_wires,
)
from .runtime_environment import (
    _jax_complex_dtype,
    _jax_real_dtype,
    _jnp_device_put,
    _require_jax,
    _require_torch,
)
from .statevector_records import JAXStatevectorShardState
from .tensor_network_records import JAXTensorNetworkNode


def _gate_matrix_as_jax(
    instruction: Any, *, torch_dtype: Any, jax_dtype: Any
) -> tuple[Any, bool]:
    from ..statevector.local_execution import _instruction_matrix

    torch = _require_torch()
    _, jnp = _require_jax()
    matrix_torch = _instruction_matrix(
        instruction, device=torch.device("cpu"), dtype=torch_dtype
    )
    diagonal = bool(
        torch.count_nonzero(
            matrix_torch - torch.diag(torch.diagonal(matrix_torch))
        ).item()
        == 0
    )
    matrix = jnp.asarray(matrix_torch.detach().cpu().numpy(), dtype=jax_dtype)
    return matrix, diagonal


def _parameterized_gate_matrix_as_jax(
    instruction: Any, *, complex_bytes: int
) -> tuple[Any, bool]:
    from ....simulation.jax_gate_primitives import _jax_instruction_matrix

    _, jnp = _require_jax()
    matrix = _jax_instruction_matrix(instruction).astype(
        _jax_complex_dtype(complex_bytes)
    )
    diagonal_gates = {
        "z",
        "s",
        "sdg",
        "t",
        "tdg",
        "rz",
        "phase",
        "u1",
        "cz",
        "crz",
        "cphase",
        "rzz",
    }
    return matrix, str(instruction.name).lower() in diagonal_gates


def _jax_parameter_array_from_input(parameters: Any, *, complex_bytes: int) -> Any:
    import numpy as np

    jax, jnp = _require_jax()
    torch = _require_torch()
    real_dtype = _jax_real_dtype(complex_bytes)
    if torch.is_tensor(parameters):
        tensor = parameters.detach().contiguous()
        try:
            import jax.dlpack

            return jnp.asarray(jax.dlpack.from_dlpack(tensor), dtype=real_dtype)
        except Exception:
            return jnp.asarray(np.asarray(tensor.cpu()).copy(), dtype=real_dtype)
    return jnp.asarray(parameters, dtype=real_dtype)


def _torch_parameters_for_static_build(parameters: Any, *, complex_bytes: int) -> Any:
    import numpy as np

    torch = _require_torch()
    if torch.is_tensor(parameters):
        return parameters.detach().cpu()
    dtype = torch.float64 if int(complex_bytes) == 16 else torch.float32
    return torch.as_tensor(
        np.asarray(parameters).copy(), dtype=dtype, device=torch.device("cpu")
    )


def _apply_gate_to_jax_shards(
    shards: Sequence[JAXStatevectorShardState],
    matrix: Any,
    wires: Sequence[int],
    *,
    plan: Any,
    diagonal: bool,
) -> tuple[JAXStatevectorShardState, ...]:
    from ..statevector.local_execution import _basis_offset, _wire_mask

    _, jnp = _require_jax()
    shards = tuple(shards)
    if not shards:
        return ()
    wires = tuple(int(wire) for wire in wires)
    width = len(wires)
    gate_dim = 2**width
    if tuple(matrix.shape[-2:]) != (gate_dim, gate_dim):
        raise ValueError(
            f"Gate on {width} wires requires matrix shape {(gate_dim, gate_dim)}."
        )

    if diagonal:
        diagonal_values = jnp.diagonal(matrix)
        updated_shards = []
        for shard in shards:
            basis_indices = _jax_basis_indices_for_wires(
                shard.global_indices,
                n_wires=plan.n_wires,
                wires=wires,
            )
            factors = diagonal_values[basis_indices].reshape(1, -1)
            updated_shards.append(
                JAXStatevectorShardState(
                    rank=shard.rank,
                    shard=shard.shard,
                    amplitudes=shard.amplitudes * factors,
                    global_indices=shard.global_indices,
                    global_indices_tuple=shard.global_indices_tuple,
                )
            )
        return tuple(updated_shards)

    masks = tuple(_wire_mask(plan.n_wires, wire) for wire in wires)
    offsets = tuple(
        _basis_offset(plan.n_wires, wires, basis) for basis in range(gate_dim)
    )
    positions_by_global: dict[int, tuple[int, int]] = {}
    for shard_index, shard in enumerate(shards):
        for local_index, global_index in enumerate(shard.global_indices_tuple):
            positions_by_global[int(global_index)] = (shard_index, local_index)

    out_amplitudes = [shard.amplitudes for shard in shards]
    processed_bases: set[int] = set()
    for shard in shards:
        for global_index in shard.global_indices_tuple:
            base = int(global_index)
            for mask in masks:
                base &= ~mask
            if base in processed_bases:
                continue
            processed_bases.add(base)
            group_indices = tuple(base | offset for offset in offsets)
            missing = tuple(
                index for index in group_indices if index not in positions_by_global
            )
            if missing:
                raise ValueError(
                    f"Shard plan does not cover gate basis group; missing amplitudes {missing}."
                )
            locations = tuple(positions_by_global[index] for index in group_indices)
            vector = jnp.stack(
                [
                    shards[shard_index].amplitudes[:, local_index]
                    for shard_index, local_index in locations
                ],
                axis=-1,
            )
            updated = vector @ jnp.swapaxes(matrix, -2, -1)
            for basis_index, (shard_index, local_index) in enumerate(locations):
                out_amplitudes[shard_index] = (
                    out_amplitudes[shard_index]
                    .at[:, local_index]
                    .set(updated[:, basis_index])
                )

    return tuple(
        JAXStatevectorShardState(
            rank=shard.rank,
            shard=shard.shard,
            amplitudes=out_amplitudes[shard_index],
            global_indices=shard.global_indices,
            global_indices_tuple=shard.global_indices_tuple,
        )
        for shard_index, shard in enumerate(shards)
    )


def _jax_split_record(
    split_info: Mapping[str, Any],
    *,
    bond: int,
    max_bond: int | None,
    cutoff: float,
) -> dict[str, Any] | None:
    if split_info.get("method") != "svd":
        return None
    return {
        "bond": int(bond),
        "kept_rank": int(split_info["rank"]),
        "original_rank": int(split_info["original_rank"]),
        "discarded_weight": float(split_info["discarded_weight"]),
        "max_bond": max_bond,
        "cutoff": float(cutoff),
        "source": "jax_sharded_mps_two_site",
    }


def _jax_nodes_from_torch_nodes(
    nodes: Sequence[Any], *, dtype: Any, device: Any | None
) -> tuple[JAXTensorNetworkNode, ...]:
    import numpy as np

    _, jnp = _require_jax()
    out = []
    for node in nodes:
        tensor = jnp.asarray(np.asarray(node.tensor.detach().cpu()).copy(), dtype=dtype)
        tensor = _jnp_device_put(tensor, device)
        out.append(
            JAXTensorNetworkNode(
                tensor=tensor,
                labels=tuple(int(label) for label in node.labels),
                name=str(node.name),
                metadata=node.metadata,
            )
        )
    return tuple(out)
