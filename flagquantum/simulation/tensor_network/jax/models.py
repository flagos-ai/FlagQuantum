"""Numerical records used by JAX tensor-network simulation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from flagquantum._import_compat import alias_module as _alias_module


@dataclass(frozen=True)
class JAXTensorNetworkNode:
    """One JAX tensor and its integer contraction labels."""

    tensor: Any
    labels: tuple[int, ...]
    name: str = ""
    metadata: Mapping[str, Any] | None = None


_alias_module("flagquantum.simulation.jax.tensor_network.models", __name__)
