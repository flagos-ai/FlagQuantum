"""Independent statevector facade over the compatibility implementation."""

from __future__ import annotations

from typing import Any

_EXPORTS = {
    "JAXStatevectorShardState",
    "JAXShardedStatevectorResult",
    "JAXShardedStatevectorParameterGradientResult",
    "JAXShardedStatevectorTrainingPlan",
    "run_jax_sharded_statevector",
    "jax_sharded_statevector_parameter_value_and_grad",
    "plan_jax_sharded_statevector_training",
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    from ._loader import resolve

    return resolve(name, domain="statevector")


def __dir__() -> list[str]:
    return sorted(_EXPORTS)


__all__ = sorted(_EXPORTS)
