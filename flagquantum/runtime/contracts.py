"""Backend-neutral execution contracts for runtime consumers."""

from .distributed.protocols import (
    DistributedExecutionRecord,
    DistributedExecutionRequest,
    DistributedExecutor,
)
from .module import Module
from .policy import RuntimePolicy
from .result import ExecutionResult, MeasurementResult, normalize_execution_result

__all__ = (
    "DistributedExecutionRecord",
    "DistributedExecutionRequest",
    "DistributedExecutor",
    "ExecutionResult",
    "MeasurementResult",
    "Module",
    "RuntimePolicy",
    "normalize_execution_result",
)
