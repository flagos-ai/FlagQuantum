"""Narrow runtime services consumed by the compiler planner.

This module is the only supported compiler-to-runtime seam. It deliberately
exposes planning and capability queries, never execution implementations.
"""

from __future__ import annotations

from typing import Any

from .configuration import backend_execution_options


def plan_jax_statevector_training(*args: Any, **kwargs: Any) -> Any:
    """Plan sharded JAX statevector training without exposing its implementation."""

    from .executors.jax import plan_jax_sharded_statevector_training

    return plan_jax_sharded_statevector_training(*args, **kwargs)


def plan_jax_mps_training(*args: Any, **kwargs: Any) -> Any:
    """Plan sharded JAX MPS training without exposing its implementation."""

    from .executors.jax import plan_jax_sharded_mps_training

    return plan_jax_sharded_mps_training(*args, **kwargs)


__all__ = (
    "backend_execution_options",
    "plan_jax_mps_training",
    "plan_jax_statevector_training",
)
