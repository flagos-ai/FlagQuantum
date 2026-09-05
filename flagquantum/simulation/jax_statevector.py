"""Dependency-light local JAX statevector numerics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def jax_sharded_statevector_rank_loss(
    amplitudes: Any,
    global_indices: Any,
    *,
    n_wires: int,
    observable: str,
    observable_wires: Sequence[int] | None,
) -> Any:
    """Evaluate a rank-local norm or Z-family observable contribution."""

    import jax.numpy as jnp

    normalized = str(observable)
    if normalized == "state_norm":
        return jnp.real(jnp.sum(jnp.abs(amplitudes) ** 2))
    if normalized not in {"z", "z_sum"}:
        raise ValueError(
            "JAX pmap sharded statevector gradients currently support "
            "observable='z_sum', 'z', or 'state_norm'."
        )
    wires = (
        tuple(range(int(n_wires)))
        if observable_wires is None
        else tuple(int(wire) for wire in observable_wires)
    )
    probabilities = jnp.abs(amplitudes) ** 2
    total = jnp.zeros((), dtype=probabilities.real.dtype)
    for wire in wires:
        bit = (global_indices >> (int(n_wires) - 1 - int(wire))) & 1
        signs = 1.0 - 2.0 * bit.astype(probabilities.real.dtype)
        total = total + jnp.sum(probabilities * signs.reshape(1, -1))
    return jnp.real(total)
