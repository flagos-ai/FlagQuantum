"""Semantic catalog for FlagQuantum-owned computational kernels.

Importing this package is metadata-only: it does not import Triton, initialize
CUDA, or load implementation modules.
"""

from .evidence import EVIDENCE
from .implementations import IMPLEMENTATIONS
from .schema import KernelEvidence, KernelImplementation, KernelSemantic
from .semantics import SEMANTICS
from .validation import validate_catalog

__all__ = [
    "IMPLEMENTATIONS",
    "SEMANTICS",
    "EVIDENCE",
    "KernelEvidence",
    "KernelImplementation",
    "KernelSemantic",
    "validate_catalog",
]
