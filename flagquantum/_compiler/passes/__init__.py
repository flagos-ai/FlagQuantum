"""Private QuantumIR pass infrastructure; not a supported public API."""

from .base import CompilerPass, PassDescriptor, PassResult
from .canonicalize import CanonicalizeAttributesPass
from .manager import PassManager, PipelineResult
from .placement_routing import DirectedCouplingGraph, PlacementRoutingPass
from .static_canonicalization import (
    CancelSelfInverseOperationsPass,
    MergeAdjacentRotationsPass,
    RemoveIdentityOperationsPass,
    StaticCanonicalizationPass,
    phase2_batch_a_passes,
)
from .target_decomposition import (
    UNIVERSAL_RX_RY_RZ_CX_V1,
    DecomposeToTargetGateSetPass,
    TargetGateSetProfile,
)

__all__ = [
    "CanonicalizeAttributesPass",
    "CancelSelfInverseOperationsPass",
    "CompilerPass",
    "PassDescriptor",
    "PassManager",
    "PassResult",
    "PipelineResult",
    "DirectedCouplingGraph",
    "PlacementRoutingPass",
    "MergeAdjacentRotationsPass",
    "RemoveIdentityOperationsPass",
    "StaticCanonicalizationPass",
    "phase2_batch_a_passes",
    "DecomposeToTargetGateSetPass",
    "TargetGateSetProfile",
    "UNIVERSAL_RX_RY_RZ_CX_V1",
]
