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
from .passes import (
    DEFAULT_HYBRID_PASSES,
    BoundedLoopUnrollPass,
    ConstantFoldPass,
    DeadConstantEliminationPass,
    HybridOptimizationResult,
    HybridProgramAnalysis,
    PassRecord,
    StructuredControlFlowSimplificationPass,
    analyze_program,
    run_pass_pipeline,
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
    "BoundedLoopUnrollPass",
    "CaptureDiagnostic",
    "CircuitStructureCache",
    "ConstantFoldPass",
    "DEFAULT_HYBRID_PASSES",
    "DeadConstantEliminationPass",
    "HybridProgram",
    "HybridOptimizationResult",
    "HybridProgramAnalysis",
    "HybridCaptureError",
    "HybridVerificationError",
    "IRType",
    "LoweredHybridProgram",
    "LoweredDynamicProgram",
    "Operation",
    "PassRecord",
    "Region",
    "SourceLocation",
    "SpecializationError",
    "SpecializedTrace",
    "StructuredControlFlowSimplificationPass",
    "TraceGate",
    "Value",
    "ValueId",
    "capture_function",
    "capture_source",
    "analyze_program",
    "lower_trace",
    "lower_dynamic_program",
    "run_pass_pipeline",
    "scalar_type",
    "tensor_type",
    "specialize_and_lower",
    "specialize_program",
    "verify_program",
)
