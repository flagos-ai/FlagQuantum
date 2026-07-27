"""Backend-neutral execution contracts for runtime consumers."""

from .distributed.protocols import (
    DistributedExecutionRecord,
    DistributedExecutionRequest,
    DistributedExecutor,
)
from .module import Module, QuantumModule
from .policy import RuntimePolicy
from .result import ExecutionResult, normalize_execution_result

__all__ = (
    "DistributedExecutionRecord",
    "DistributedExecutionRequest",
    "DistributedExecutor",
    "ExecutionResult",
    "Module",
    "QuantumModule",
    "RuntimePolicy",
    "normalize_execution_result",
)
