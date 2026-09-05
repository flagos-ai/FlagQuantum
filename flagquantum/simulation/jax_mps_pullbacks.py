"""Local JAX MPS parameter and boundary pullback kernels."""

from __future__ import annotations

from typing import Any


def jax_mps_owner_local_vjp(
    parameter: Any,
    cotangent: Any,
    *,
    gate_kind: str,
) -> Any:
    import jax
    import jax.numpy as jnp

    if gate_kind == "one_site_ry":

        def local_score(value: Any) -> Any:
            tensor = jnp.stack((jnp.cos(value / 2.0), jnp.sin(value / 2.0)))
            return jnp.real(
                tensor[0] * jnp.conj(tensor[0]) - tensor[1] * jnp.conj(tensor[1])
            )

    elif gate_kind == "same_shard_two_site_rxx":

        def local_score(value: Any) -> Any:
            tensor = jnp.stack(
                (
                    jnp.cos(value / 2.0),
                    jnp.asarray(0.0j),
                    jnp.asarray(0.0j),
                    -1j * jnp.sin(value / 2.0),
                )
            )
            return jnp.real(
                tensor[0] * jnp.conj(tensor[0])
                + tensor[1] * jnp.conj(tensor[1])
                - tensor[2] * jnp.conj(tensor[2])
                - tensor[3] * jnp.conj(tensor[3])
            )

    else:
        raise ValueError(f"unsupported owner-local MPS gate kind: {gate_kind}")

    _, pullback = jax.vjp(local_score, parameter)
    return pullback(jnp.asarray(cotangent, dtype=parameter.dtype))[0]


def jax_mps_boundary_rxx_pullback(
    parameter: Any,
    left_site: Any,
    right_site: Any,
) -> tuple[Any, Any, Any, Any]:
    import jax
    import jax.numpy as jnp

    def boundary_score(value: Any, left: Any, right: Any) -> Any:
        product = jnp.einsum("lpm,mqr->lpqr", left, right, precision="highest")
        flat = product.reshape(4)
        cosine = jnp.cos(value / 2.0)
        sine = jnp.sin(value / 2.0)
        gate = jnp.asarray(
            (
                (cosine, 0.0, 0.0, -1j * sine),
                (0.0, cosine, -1j * sine, 0.0),
                (0.0, -1j * sine, cosine, 0.0),
                (-1j * sine, 0.0, 0.0, cosine),
            ),
            dtype=left.dtype,
        )
        evolved = jnp.matmul(gate, flat, precision="highest")
        probabilities = jnp.real(evolved * jnp.conj(evolved))
        return probabilities[0] + probabilities[1] - probabilities[2] - probabilities[3]

    value, pullback = jax.vjp(boundary_score, parameter, left_site, right_site)
    parameter_gradient, left_adjoint, right_adjoint = pullback(
        jnp.asarray(1.0, dtype=value.dtype)
    )
    return value, parameter_gradient, left_adjoint, right_adjoint
