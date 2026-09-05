"""Independent JAX distributed initialization and transport facade."""

from __future__ import annotations

from .runtime_environment import initialize_jax_distributed

__all__ = ("initialize_jax_distributed",)
