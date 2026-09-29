"""Data model for the FlagQuantum kernel catalog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

KernelDomain = Literal["statevector", "gradient", "mps", "numerics"]
KernelProvider = Literal["pytorch", "triton", "flagtree"]
KernelDirection = Literal["forward", "backward", "jvp", "vjp", "jacobian"]
KernelMaturity = Literal["experimental", "provisional", "stable"]


@dataclass(frozen=True, slots=True)
class KernelSemantic:
    """A provider-neutral mathematical operation owned by FlagQuantum."""

    catalog_id: str
    semantic_id: str
    domain: KernelDomain
    summary: str
    reference: str
    workloads: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class KernelImplementation:
    """One executable implementation of a cataloged kernel semantic."""

    implementation_id: str
    semantic_id: str
    provider: KernelProvider
    module: str
    symbol: str
    devices: tuple[str, ...]
    dtypes: tuple[str, ...]
    layouts: tuple[str, ...]
    directions: tuple[KernelDirection, ...]
    addressing: tuple[str, ...]
    maturity: KernelMaturity
    internal_fallback: bool = False


__all__ = [
    "KernelDirection",
    "KernelDomain",
    "KernelImplementation",
    "KernelMaturity",
    "KernelProvider",
    "KernelSemantic",
]
