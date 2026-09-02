"""Private QuantumIR pass infrastructure; not a supported public API."""

from .base import CompilerPass, PassDescriptor, PassResult
from .canonicalize import CanonicalizeAttributesPass
from .manager import PassManager, PipelineResult
from .static_canonicalization import (
    CancelSelfInverseOperationsPass,
    MergeAdjacentRotationsPass,
    RemoveIdentityOperationsPass,
    StaticCanonicalizationPass,
    phase2_batch_a_passes,
)

__all__ = [
    "CanonicalizeAttributesPass",
    "CancelSelfInverseOperationsPass",
    "CompilerPass",
    "PassDescriptor",
    "PassManager",
    "PassResult",
    "PipelineResult",
    "MergeAdjacentRotationsPass",
    "RemoveIdentityOperationsPass",
    "StaticCanonicalizationPass",
    "phase2_batch_a_passes",
]
