"""Private structured-program compiler machinery.

This package is intentionally absent from :mod:`flagquantum.compiler` exports.
"""

from .capture import (
    CaptureDiagnostic,
    HybridCaptureError,
    capture_function,
    capture_source,
)
from .model import (
    BOOL,
    INDEX,
    QUANTUM_EFFECT,
    Block,
    HybridProgram,
    IRType,
    Operation,
    Region,
    SourceLocation,
    Value,
    ValueId,
    scalar_type,
    tensor_type,
)
from .verifier import HybridVerificationError, verify_program

__all__ = (
    "BOOL",
    "INDEX",
    "QUANTUM_EFFECT",
    "Block",
    "CaptureDiagnostic",
    "HybridProgram",
    "HybridCaptureError",
    "HybridVerificationError",
    "IRType",
    "Operation",
    "Region",
    "SourceLocation",
    "Value",
    "ValueId",
    "capture_function",
    "capture_source",
    "scalar_type",
    "tensor_type",
    "verify_program",
)
