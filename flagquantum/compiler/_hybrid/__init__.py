"""Historical import facade for the relocated implementation."""

import sys as _alias_sys
from importlib import import_module as _import_module
from typing import Any as _Any

from flagquantum.compiler.frontends._structured import (
    BOOL as BOOL,
)
from flagquantum.compiler.frontends._structured import (
    DEFAULT_HYBRID_PASSES as DEFAULT_HYBRID_PASSES,
)
from flagquantum.compiler.frontends._structured import (
    INDEX as INDEX,
)
from flagquantum.compiler.frontends._structured import (
    QUANTUM_EFFECT as QUANTUM_EFFECT,
)
from flagquantum.compiler.frontends._structured import (
    Block as Block,
)
from flagquantum.compiler.frontends._structured import (
    BoundedLoopUnrollPass as BoundedLoopUnrollPass,
)
from flagquantum.compiler.frontends._structured import (
    CaptureDiagnostic as CaptureDiagnostic,
)
from flagquantum.compiler.frontends._structured import (
    CircuitStructureCache as CircuitStructureCache,
)
from flagquantum.compiler.frontends._structured import (
    ConstantFoldPass as ConstantFoldPass,
)
from flagquantum.compiler.frontends._structured import (
    DeadConstantEliminationPass as DeadConstantEliminationPass,
)
from flagquantum.compiler.frontends._structured import (
    HybridCaptureError as HybridCaptureError,
)
from flagquantum.compiler.frontends._structured import (
    HybridOptimizationResult as HybridOptimizationResult,
)
from flagquantum.compiler.frontends._structured import (
    HybridProgram as HybridProgram,
)
from flagquantum.compiler.frontends._structured import (
    HybridProgramAnalysis as HybridProgramAnalysis,
)
from flagquantum.compiler.frontends._structured import (
    HybridVerificationError as HybridVerificationError,
)
from flagquantum.compiler.frontends._structured import (
    IRType as IRType,
)
from flagquantum.compiler.frontends._structured import (
    LoweredDynamicProgram as LoweredDynamicProgram,
)
from flagquantum.compiler.frontends._structured import (
    LoweredHybridProgram as LoweredHybridProgram,
)
from flagquantum.compiler.frontends._structured import (
    Operation as Operation,
)
from flagquantum.compiler.frontends._structured import (
    PassOutcome as PassOutcome,
)
from flagquantum.compiler.frontends._structured import (
    PassRecord as PassRecord,
)
from flagquantum.compiler.frontends._structured import (
    Region as Region,
)
from flagquantum.compiler.frontends._structured import (
    SourceLocation as SourceLocation,
)
from flagquantum.compiler.frontends._structured import (
    SpecializationError as SpecializationError,
)
from flagquantum.compiler.frontends._structured import (
    SpecializedTrace as SpecializedTrace,
)
from flagquantum.compiler.frontends._structured import (
    StructuredControlFlowSimplificationPass as StructuredControlFlowSimplificationPass,
)
from flagquantum.compiler.frontends._structured import (
    TraceGate as TraceGate,
)
from flagquantum.compiler.frontends._structured import (
    Value as Value,
)
from flagquantum.compiler.frontends._structured import (
    ValueId as ValueId,
)
from flagquantum.compiler.frontends._structured import __all__ as __all__
from flagquantum.compiler.frontends._structured import (
    analyze_program as analyze_program,
)
from flagquantum.compiler.frontends._structured import (
    capture_function as capture_function,
)
from flagquantum.compiler.frontends._structured import (
    capture_source as capture_source,
)
from flagquantum.compiler.frontends._structured import (
    lower_dynamic_program as lower_dynamic_program,
)
from flagquantum.compiler.frontends._structured import (
    lower_trace as lower_trace,
)
from flagquantum.compiler.frontends._structured import (
    run_pass_pipeline as run_pass_pipeline,
)
from flagquantum.compiler.frontends._structured import (
    scalar_type as scalar_type,
)
from flagquantum.compiler.frontends._structured import (
    specialize_and_lower as specialize_and_lower,
)
from flagquantum.compiler.frontends._structured import (
    specialize_program as specialize_program,
)
from flagquantum.compiler.frontends._structured import (
    tensor_type as tensor_type,
)
from flagquantum.compiler.frontends._structured import (
    verify_program as verify_program,
)

_implementation = _import_module("flagquantum.compiler.frontends._structured")
__doc__ = _implementation.__doc__ or ""


def __getattr__(name: str) -> _Any:
    cached = _alias_sys.modules.get(f"{__name__}.{name}")
    if cached is not None:
        return cached
    return getattr(_implementation, name)
