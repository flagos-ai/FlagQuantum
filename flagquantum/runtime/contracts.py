"""Backend-neutral execution contracts for runtime consumers."""

from flagquantum.runtime.results import (
    ExecutionResult,
    MeasurementResult,
    normalize_execution_result,
)
from flagquantum.runtime.training.module import Module

from .distributed.protocols import (
    DistributedExecutionRecord,
    DistributedExecutionRequest,
    DistributedExecutor,
)
from .options import ExecutionOptions
from .policy import RuntimePolicy

__all__ = (
    "DistributedExecutionRecord",
    "DistributedExecutionRequest",
    "DistributedExecutor",
    "ExecutionOptions",
    "ExecutionResult",
    "MeasurementResult",
    "Module",
    "RuntimePolicy",
    "normalize_execution_result",
)
