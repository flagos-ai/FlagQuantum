"""Structured diagnostics for the private compiler implementation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DiagnosticSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    NOTE = "note"


class DiagnosticCode(str, Enum):
    UNKNOWN_OPERATION = "IRV001"
    UNKNOWN_ATTRIBUTE = "IRV002"
    MISSING_ATTRIBUTE = "IRV003"
    ATTRIBUTE_TYPE_MISMATCH = "IRV004"
    OPERAND_ARITY_MISMATCH = "IRV005"
    RESULT_ARITY_MISMATCH = "IRV006"
    REGION_COUNT_MISMATCH = "IRV007"
    USE_BEFORE_DEFINITION = "IRV008"
    DUPLICATE_DEFINITION = "IRV009"
    VALUE_TYPE_MISMATCH = "IRV010"
    LINEAR_VALUE_REUSED = "IRV011"
    USE_AFTER_RELEASE = "IRV012"
    OPERATION_AFTER_TERMINATOR = "IRV013"
    INVALID_VARIADIC_ARITY = "IRV014"


@dataclass(frozen=True)
class DiagnosticLocation:
    source: str
    line: int
    column: int = 0

    def __post_init__(self) -> None:
        source = str(self.source)
        line = int(self.line)
        column = int(self.column)
        if not source or line < 1 or column < 0:
            raise ValueError("invalid diagnostic location")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "line", line)
        object.__setattr__(self, "column", column)

    def to_dict(self) -> dict[str, object]:
        return {"source": self.source, "line": self.line, "column": self.column}


@dataclass(frozen=True)
class Diagnostic:
    code: DiagnosticCode
    message: str
    location: DiagnosticLocation | None = None
    notes: tuple[str, ...] = ()
    severity: DiagnosticSeverity = DiagnosticSeverity.ERROR

    def __post_init__(self) -> None:
        message = str(self.message).strip()
        notes = tuple(str(note).strip() for note in self.notes)
        if not message:
            raise ValueError("diagnostic message cannot be empty")
        if any(not note for note in notes):
            raise ValueError("diagnostic notes cannot be empty")
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "notes", notes)

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code.value,
            "severity": self.severity.value,
            "message": self.message,
            "location": self.location.to_dict() if self.location else None,
            "notes": list(self.notes),
        }


__all__ = [
    "Diagnostic",
    "DiagnosticCode",
    "DiagnosticLocation",
    "DiagnosticSeverity",
]
