"""Semantic catalog for FlagQuantum-owned computational kernels.

Importing this package is metadata-only: it does not import Triton, initialize
CUDA, or load implementation modules.
"""

from .implementations import IMPLEMENTATIONS
from .schema import KernelImplementation, KernelSemantic
from .semantics import SEMANTICS
from .validation import validate_catalog

__all__ = [
    "IMPLEMENTATIONS",
    "SEMANTICS",
    "KernelImplementation",
    "KernelSemantic",
    "validate_catalog",
]
