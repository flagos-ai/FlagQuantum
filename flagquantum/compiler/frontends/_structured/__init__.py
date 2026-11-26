"""Private structured-program compiler machinery.

This package is intentionally absent from :mod:`flagquantum.compiler` exports.
"""

from flagquantum.compiler.frontends._structured.diagnostics import SpecializationError
from flagquantum.compiler.frontends._structured.dynamic.program import (
    LoweredDynamicProgram,
    lower_dynamic_program,
)
from flagquantum.compiler.frontends._structured.frontend.diagnostics import (
    CaptureDiagnostic,
    HybridCaptureError,
)
from flagquantum.compiler.frontends._structured.frontend.source import (
    capture_function,
    capture_source,
)
from flagquantum.compiler.frontends._structured.lowering import (
    CircuitStructureCache,
    LoweredHybridProgram,
    lower_trace,
    specialize_and_lower,
)
from flagquantum.compiler.frontends._structured.model import (
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
from flagquantum.compiler.frontends._structured.passes import (
    DEFAULT_HYBRID_PASSES,
    BoundedLoopUnrollPass,
    ConstantFoldPass,
    DeadConstantEliminationPass,
    HybridOptimizationResult,
    HybridProgramAnalysis,
    PassOutcome,
    PassRecord,
    StructuredControlFlowSimplificationPass,
    analyze_program,
    run_pass_pipeline,
)
from flagquantum.compiler.frontends._structured.specialize import (
    SpecializedTrace,
    TraceGate,
    specialize_program,
)
from flagquantum.compiler.frontends._structured.verifier import (
    HybridVerificationError,
    verify_program,
)

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
    "PassOutcome",
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
