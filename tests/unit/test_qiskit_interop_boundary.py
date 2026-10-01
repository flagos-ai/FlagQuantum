from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from flagquantum.ecosystem.qiskit import (
    QiskitConversionIssue,
    QiskitConversionReport,
    QiskitDependencyError,
    conversion,
)
from tools.check_architecture import CONFIG, architecture_errors

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_qiskit_namespace_is_lazy_without_external_framework_imports() -> None:
    code = """
import sys
import flagquantum.ecosystem.qiskit
loaded = sorted(
    name for name in sys.modules
    if name == 'qiskit' or name.startswith('qiskit.')
    or name == 'qiskit_aer' or name.startswith('qiskit_aer.')
)
assert not loaded, loaded
assert 'flagquantum.ecosystem.qiskit.execution' not in sys.modules
assert 'flagquantum.runtime.dynamic' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_experimental_convenience_exports_delegate_to_interop() -> None:
    import flagquantum as fq
    import flagquantum.ecosystem.pennylane as pennylane
    import flagquantum.ecosystem.qiskit as qiskit

    assert fq.experimental.interop.qiskit is qiskit
    assert fq.experimental.interop.pennylane is pennylane


def test_missing_qiskit_fails_only_when_conversion_is_requested(monkeypatch) -> None:
    real_import = conversion.import_module

    def unavailable(name: str):
        if name == "qiskit" or name.startswith("qiskit."):
            raise ImportError(name)
        return real_import(name)

    monkeypatch.setattr(conversion, "import_module", unavailable)
    with pytest.raises(QiskitDependencyError, match=r"flagquantum\[qiskit\]"):
        conversion.from_qiskit(object())


def test_conversion_report_is_machine_readable_and_fail_closed() -> None:
    issue = QiskitConversionIssue(
        "unsupported_operation",
        "operation cannot be represented",
        "error",
        3,
        "opaque",
    )
    report = QiskitConversionReport("from_qiskit", "2.0", (issue,))

    assert not report.lossless
    assert report.blockers == (issue,)
    assert report.to_dict() == {
        "schema": "flagquantum_qiskit_conversion_report_v1",
        "direction": "from_qiskit",
        "framework_version": "2.0",
        "lossless": False,
        "issues": [
            {
                "code": "unsupported_operation",
                "message": "operation cannot be represented",
                "severity": "error",
                "operation_index": 3,
                "operation_name": "opaque",
            }
        ],
    }


def test_architecture_isolates_qiskit_imports_to_owned_edge_namespaces() -> None:
    prefixes = CONFIG["interop_boundaries"]["qiskit_import_allowed_prefixes"]
    assert prefixes == [
        "flagquantum/ecosystem/qiskit/",
    ]
    assert architecture_errors() == ()


def test_qiskit_aer_bridge_has_no_parallel_top_level_package() -> None:
    assert not (ROOT / "flagquantum_qiskit_aer").exists()
    assert (ROOT / "flagquantum/ecosystem/qiskit/aer.py").is_file()


def test_ci_proves_both_qiskit_optionality_and_real_compatibility() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "qiskit-optional:" in workflow
    assert 'OPTIONAL_VERSIONS: "2.0.* 2.5.*"' in workflow
    assert "for version in ${OPTIONAL_VERSIONS}" in workflow
    assert '"qiskit[qasm3-import]==${version}"' in workflow
    assert "python tools/check_qiskit_interop_contract.py" in workflow
    assert "tests/test_qiskit_interop.py" in workflow
    assert "tests/test_qiskit_interop_conformance.py" in workflow
    assert "tests/optional/test_qiskit_aer_backend.py" in workflow
    assert "-m qiskit -q" in workflow
    assert "external quantum frameworks are absent from core" in workflow


def test_custom_unitary_validation_returns_a_tensor_or_none() -> None:
    """Export may detach the validator's result unconditionally.

    `_validated_custom_unitary` builds the tensor itself, so its success value
    is a `torch.Tensor` and its refusal value is `None`; export reads the
    `None` as "already reported" and skips the instruction, then detaches,
    moves to CPU, and converts to numpy with no further test. That step used to
    sit behind `hasattr(matrix, "detach")`, a guard left from exporting the raw
    `instruction.matrix` field, and it was that guard -- not the type -- which
    kept a third return shape from reaching the detach. This pins the two
    shapes the unconditional step relies on, so a validator that starts
    returning, say, an ndarray or a qiskit matrix fails here rather than inside
    a caller that no longer checks.
    """

    accepted = (
        ([[1, 0], [0, 1]], 1),
        ([[0, 1], [1, 0]], 1),
        (torch.eye(2, dtype=torch.complex128), 1),
        (torch.eye(4, dtype=torch.complex64), 2),
    )
    for matrix, width in accepted:
        issues: list[QiskitConversionIssue] = []
        resolved = conversion._validated_custom_unitary(
            matrix,
            tuple(range(width)),
            issues=issues,
            operation_index=0,
            operation_name="unitary",
        )
        assert isinstance(resolved, torch.Tensor), (matrix, issues)
        assert issues == []

    refused = (
        ([[1, 1], [0, 1]], 1, "non_unitary_custom_matrix"),
        ([[1, 0, 0], [0, 1, 0]], 1, "invalid_custom_unitary_shape"),
        ([[float("nan"), 0], [0, 1]], 1, "invalid_custom_unitary_values"),
    )
    for matrix, width, code in refused:
        issues = []
        assert (
            conversion._validated_custom_unitary(
                matrix,
                tuple(range(width)),
                issues=issues,
                operation_index=0,
                operation_name="unitary",
            )
            is None
        )
        assert [issue.code for issue in issues] == [code]
