from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from flagquantum._compiler.diagnostics import (
    Diagnostic,
    DiagnosticCode,
    DiagnosticLocation,
)

pytestmark = pytest.mark.unit


def test_diagnostic_is_immutable_and_machine_serializable() -> None:
    diagnostic = Diagnostic(
        DiagnosticCode.UNKNOWN_OPERATION,
        "unknown operation",
        DiagnosticLocation("fixture.fq", 4, 2),
        ("register the operation schema",),
    )

    assert diagnostic.to_dict() == {
        "code": "IRV001",
        "severity": "error",
        "message": "unknown operation",
        "location": {"source": "fixture.fq", "line": 4, "column": 2},
        "notes": ["register the operation schema"],
    }
    with pytest.raises(FrozenInstanceError):
        diagnostic.message = "changed"  # type: ignore[misc]


def test_diagnostic_rejects_empty_message_note_and_invalid_location() -> None:
    with pytest.raises(ValueError, match="message"):
        Diagnostic(DiagnosticCode.UNKNOWN_OPERATION, "")
    with pytest.raises(ValueError, match="notes"):
        Diagnostic(DiagnosticCode.UNKNOWN_OPERATION, "message", notes=("",))
    with pytest.raises(ValueError, match="location"):
        DiagnosticLocation("fixture.fq", 0)
