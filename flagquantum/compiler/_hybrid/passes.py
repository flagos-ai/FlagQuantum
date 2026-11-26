"""Historical import facade for the relocated implementation."""

import sys as _sys
from importlib import import_module as _import_module

from flagquantum.compiler.frontends._structured.passes import (
    DEFAULT_HYBRID_PASSES as DEFAULT_HYBRID_PASSES,
)
from flagquantum.compiler.frontends._structured.passes import (
    BoundedLoopUnrollPass as BoundedLoopUnrollPass,
)
from flagquantum.compiler.frontends._structured.passes import (
    ConstantFoldPass as ConstantFoldPass,
)
from flagquantum.compiler.frontends._structured.passes import (
    DeadConstantEliminationPass as DeadConstantEliminationPass,
)
from flagquantum.compiler.frontends._structured.passes import (
    HybridOptimizationResult as HybridOptimizationResult,
)
from flagquantum.compiler.frontends._structured.passes import (
    HybridProgramAnalysis as HybridProgramAnalysis,
)
from flagquantum.compiler.frontends._structured.passes import (
    HybridProgramPass as HybridProgramPass,
)
from flagquantum.compiler.frontends._structured.passes import (
    PassOutcome as PassOutcome,
)
from flagquantum.compiler.frontends._structured.passes import (
    PassRecord as PassRecord,
)
from flagquantum.compiler.frontends._structured.passes import (
    StructuredControlFlowSimplificationPass as StructuredControlFlowSimplificationPass,
)
from flagquantum.compiler.frontends._structured.passes import __all__ as __all__
from flagquantum.compiler.frontends._structured.passes import (
    analyze_program as analyze_program,
)
from flagquantum.compiler.frontends._structured.passes import (
    run_pass_pipeline as run_pass_pipeline,
)

_implementation = _import_module("flagquantum.compiler.frontends._structured.passes")
_sys.modules[__name__] = _implementation
