"""Numerical records used by JAX tensor-network simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class JAXTensorNetworkNode:
    """One JAX tensor and its integer contraction labels."""

    tensor: Any
    labels: tuple[int, ...]
    name: str = ""
    metadata: Mapping[str, Any] | None = None
