from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from flagquantum.interop.qiskit import (
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
import flagquantum.interop.qiskit
loaded = sorted(
    name for name in sys.modules
    if name == 'qiskit' or name.startswith('qiskit.')
    or name == 'qiskit_aer' or name.startswith('qiskit_aer.')
)
assert not loaded, loaded
assert 'flagquantum.interop.qiskit.execution' not in sys.modules
assert 'flagquantum.runtime.dynamic' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_experimental_convenience_exports_delegate_to_interop() -> None:
    import flagquantum as fq
    from flagquantum.interop.qiskit import from_qiskit, to_qiskit

    assert fq.experimental.from_qiskit is from_qiskit
    assert fq.experimental.to_qiskit is to_qiskit


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


def test_architecture_isolates_qiskit_imports_to_interop_namespace() -> None:
    prefixes = CONFIG["interop_boundaries"]["qiskit_import_allowed_prefixes"]
    assert prefixes == ["flagquantum/interop/qiskit/"]
    assert architecture_errors() == ()


def test_ci_proves_both_qiskit_optionality_and_real_compatibility() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "qiskit-optional:" in workflow
    assert 'qiskit-version: ["2.0.*", "2.5.*"]' in workflow
    assert "'qiskit[qasm3-import]==${{ matrix.qiskit-version }}'" in workflow
    assert "python tools/check_qiskit_interop_contract.py" in workflow
    assert "python -m pytest -m qiskit -q" in workflow
    assert "external quantum frameworks are absent from core" in workflow
