"""Independent tensor-network facade over the compatibility implementation."""

from __future__ import annotations

from typing import Any

_EXPORTS = {
    "JAXTensorNetworkNode",
    "JAXTNSliceRankState",
    "JAXShardedTensorNetworkResult",
    "JAXSlicedTensorNetworkGradientResult",
    "JAXSlicedTensorNetworkParameterGradientResult",
    "run_jax_sharded_tensor_network",
    "jax_sliced_tensor_network_value_and_grad",
    "jax_sliced_tensor_network_parameter_value_and_grad",
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    from ._loader import resolve

    return resolve(name, domain="tensor_network")


def __dir__() -> list[str]:
    return sorted(_EXPORTS)


__all__ = sorted(_EXPORTS)
