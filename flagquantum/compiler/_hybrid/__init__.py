"""Private structured-program compiler machinery.

This package is intentionally absent from :mod:`flagquantum.compiler` exports.
"""

from .capture import (
    CaptureDiagnostic,
    HybridCaptureError,
    capture_function,
    capture_source,
)
from .dynamic_lowering import LoweredDynamicProgram, lower_dynamic_program
from .lowering import (
    CircuitStructureCache,
    LoweredHybridProgram,
    lower_trace,
    specialize_and_lower,
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
from .specialize import (
    SpecializationError,
    SpecializedTrace,
    TraceGate,
    specialize_program,
)
from .verifier import HybridVerificationError, verify_program

__all__ = (
    "BOOL",
    "INDEX",
    "QUANTUM_EFFECT",
    "Block",
    "CaptureDiagnostic",
    "CircuitStructureCache",
    "HybridProgram",
    "HybridCaptureError",
    "HybridVerificationError",
    "IRType",
    "LoweredHybridProgram",
    "LoweredDynamicProgram",
    "Operation",
    "Region",
    "SourceLocation",
    "SpecializationError",
    "SpecializedTrace",
    "TraceGate",
    "Value",
    "ValueId",
    "capture_function",
    "capture_source",
    "lower_trace",
    "lower_dynamic_program",
    "scalar_type",
    "tensor_type",
    "specialize_and_lower",
    "specialize_program",
    "verify_program",
)
