"""Private QuantumIR pass infrastructure; not a supported public API."""

from .base import CompilerPass, PassDescriptor, PassResult
from .canonicalize import CanonicalizeAttributesPass
from .manager import PassManager, PipelineResult

__all__ = [
    "CanonicalizeAttributesPass",
    "CompilerPass",
    "PassDescriptor",
    "PassManager",
    "PassResult",
    "PipelineResult",
]
