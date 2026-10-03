"""Data model for the FlagQuantum kernel catalog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

KernelDomain = Literal["statevector", "gradient", "mps", "numerics"]
KernelProvider = Literal["pytorch", "triton", "flagtree"]
KernelDirection = Literal["forward", "backward", "jvp", "vjp", "jacobian"]
KernelMaturity = Literal["experimental", "provisional", "stable"]

# The execution-device axis is closed on purpose. ``flagquantum.compute`` owns the
# platform registry, but the protected boundaries in ``architecture.toml`` forbid
# ``flagquantum.kernels`` from importing it, so the catalog declares the axis here
# and ``tests/unit/test_kernel_catalog.py`` asserts the two vocabularies agree.
# An open axis would make a misspelled device indistinguishable from a device that
# simply has no implementation, which is exactly the fail-open the axis prevents.
KernelDevice = Literal["cpu", "cuda", "flagos"]
KERNEL_DEVICES: frozenset[str] = frozenset(get_args(KernelDevice))


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
    devices: tuple[KernelDevice, ...]
    dtypes: tuple[str, ...]
    layouts: tuple[str, ...]
    directions: tuple[KernelDirection, ...]
    addressing: tuple[str, ...]
    maturity: KernelMaturity
    internal_fallback: bool = False


@dataclass(frozen=True, slots=True)
class KernelEvidence:
    """Repository evidence associated with one kernel implementation."""

    evidence_id: str
    implementation_id: str
    correctness_tests: tuple[str, ...]
    gradient_tests: tuple[str, ...] = ()
    capability_tests: tuple[str, ...] = ()
    benchmark_artifacts: tuple[str, ...] = ()
    required_lanes: tuple[str, ...] = ("gpu_scheduled",)


__all__ = [
    "KERNEL_DEVICES",
    "KernelDevice",
    "KernelDirection",
    "KernelDomain",
    "KernelEvidence",
    "KernelImplementation",
    "KernelMaturity",
    "KernelProvider",
    "KernelSemantic",
]
