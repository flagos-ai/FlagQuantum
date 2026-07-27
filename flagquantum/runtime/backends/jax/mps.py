"""Independent MPS facade over the compatibility implementation."""

from __future__ import annotations

from typing import Any

_EXPORTS = {
    "JAXMPSRankShardState",
    "JAXShardedMPSResult",
    "JAXShardedMPSParameterGradientResult",
    "JAXShardedMPSTrainingPlan",
    "run_jax_sharded_mps",
    "jax_sharded_mps_parameter_value_and_grad",
    "plan_jax_sharded_mps_training",
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    from ._loader import resolve

    return resolve(name, domain="mps")


def __dir__() -> list[str]:
    return sorted(_EXPORTS)


__all__ = sorted(_EXPORTS)
