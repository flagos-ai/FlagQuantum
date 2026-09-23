from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from tools.check_interop_capability_gap_matrix import (
    MATRIX_PATH,
    build_report,
    load_toml,
    matrix_errors,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def _matrix() -> dict:
    return load_toml(MATRIX_PATH)


def test_capability_gap_matrix_is_current() -> None:
    assert matrix_errors(_matrix(), root=ROOT) == ()


def test_matrix_rejects_missing_framework_and_capability() -> None:
    matrix = deepcopy(_matrix())
    matrix["frameworks"].pop()
    matrix["frameworks"][0]["capabilities"].pop()

    errors = matrix_errors(matrix, root=ROOT)

    assert any("framework inventory or ordering drifted" in error for error in errors)
    assert any(
        "qiskit: capability inventory or ordering drifted" in error for error in errors
    )


def test_matrix_rejects_stale_contract_evidence() -> None:
    matrix = deepcopy(_matrix())
    matrix["frameworks"][1]["capabilities"][3]["evidence"] = [
        "semantics.removed_parameter_policy"
    ]

    errors = matrix_errors(matrix, root=ROOT)

    assert (
        "cirq.symbolic_parameters: evidence path drifted: "
        "semantics.removed_parameter_policy"
    ) in errors


def test_report_derives_exact_operation_gaps_from_adapter_contracts() -> None:
    report = build_report(_matrix(), root=ROOT)
    frameworks = {item["framework"]: item for item in report["frameworks"]}

    assert report["schema"] == "flagquantum_interop_capability_gap_report_v1"
    assert list(frameworks) == ["qiskit", "cirq", "pennylane", "cudaq", "braket"]
    for framework in frameworks.values():
        operations = framework["operations"]
        assert (
            operations["supported_count"] + len(operations["missing"])
            == operations["total_count"]
        )
        assert not set(operations["supported"]) & set(operations["missing"])
        assert framework["next_gap"]
    assert frameworks["cudaq"]["direction"] == "export_only"
    assert frameworks["cudaq"]["capabilities"]["circuit_conversion"]["status"] == (
        "partial"
    )
    assert frameworks["qiskit"]["capabilities"]["symbolic_parameters"]["status"] == (
        "partial"
    )
    assert frameworks["cirq"]["capabilities"]["symbolic_parameters"]["status"] == (
        "partial"
    )
    assert frameworks["cirq"]["capabilities"]["measurements"]["status"] == "partial"


def test_operation_coverage_status_cannot_overstate_contract() -> None:
    matrix = deepcopy(_matrix())
    matrix["frameworks"][0]["capabilities"][1]["status"] = "supported"

    errors = matrix_errors(matrix, root=ROOT)

    assert (
        "qiskit.operation_coverage: status must be 'partial' for its opcode partition"
        in errors
    )
