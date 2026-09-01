"""Stable, backend-neutral exception categories for FlagQuantum users."""


class FlagQuantumError(Exception):
    """Base class for failures intentionally exposed by FlagQuantum."""

    category = "flagquantum"


class ValidationError(ValueError, FlagQuantumError):
    """A user-supplied program, option, or value is invalid."""

    category = "validation"


class SerializationError(ValueError, FlagQuantumError):
    """A versioned FlagQuantum payload cannot be read or written."""

    category = "serialization"


class PlanningError(ValueError, FlagQuantumError):
    """A requested execution plan is invalid, stale, or incompatible."""

    category = "planning"


class CompilationError(RuntimeError, FlagQuantumError):
    """Compilation failed after valid input was accepted."""

    category = "compilation"


class ExecutionError(RuntimeError, FlagQuantumError):
    """Execution or training failed after validation and planning."""

    category = "execution"


class CapabilityError(NotImplementedError, FlagQuantumError):
    """The requested semantic capability is not supported."""

    category = "capability"


__all__ = (
    "CapabilityError",
    "CompilationError",
    "ExecutionError",
    "FlagQuantumError",
    "PlanningError",
    "SerializationError",
    "ValidationError",
)
