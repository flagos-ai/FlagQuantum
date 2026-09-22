from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from flagquantum.ecosystem.cirq import CirqDependencyError, conversion
from tools.check_architecture import CONFIG, architecture_errors

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_cirq_namespace_and_registry_are_lazy() -> None:
    code = """
import sys
import flagquantum.ecosystem.cirq
from flagquantum.ecosystem import available_adapters, get_adapter
assert available_adapters() == ('cirq', 'cudaq', 'pennylane', 'qiskit')
assert get_adapter('cirq').name == 'cirq'
assert not [name for name in sys.modules if name == 'cirq' or name.startswith('cirq.')]
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_missing_cirq_fails_only_when_conversion_is_requested(monkeypatch) -> None:
    monkeypatch.setattr(
        conversion,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError(name)),
    )
    with pytest.raises(CirqDependencyError, match=r"flagquantum\[cirq\]"):
        conversion.from_cirq(object())


def test_architecture_isolates_cirq_imports() -> None:
    assert CONFIG["interop_boundaries"]["cirq_import_allowed_prefixes"] == [
        "flagquantum/ecosystem/cirq/"
    ]
    assert architecture_errors() == ()


def test_ci_runs_both_cirq_versions_and_the_cirq_suite() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "cirq-optional:" in workflow
    assert 'cirq-version: ["1.6.1", "1.7.0"]' in workflow
    assert "python tools/check_cirq_interop_contract.py --verify-sdk" in workflow
    assert "python -m pytest -m cirq -q" in workflow
