from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from flagquantum.ecosystem.braket import BraketDependencyError, conversion
from tools.check_architecture import CONFIG, architecture_errors

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_braket_namespace_and_registry_are_lazy() -> None:
    code = """
import sys
import flagquantum.ecosystem.braket
from flagquantum.ecosystem import available_adapters, get_adapter
assert available_adapters() == ('braket', 'cirq', 'cudaq', 'pennylane', 'qiskit')
assert get_adapter('braket').name == 'braket'
assert not [name for name in sys.modules if name == 'braket' or name.startswith('braket.')]
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_missing_braket_fails_only_when_conversion_is_requested(monkeypatch) -> None:
    monkeypatch.setattr(
        conversion,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError(name)),
    )
    with pytest.raises(BraketDependencyError, match=r"flagquantum\[braket\]"):
        conversion.from_braket(object())


def test_architecture_isolates_braket_imports() -> None:
    assert CONFIG["interop_boundaries"]["braket_import_allowed_prefixes"] == [
        "flagquantum/ecosystem/braket/",
        "flagquantum/remote/qpu/",
    ]
    assert architecture_errors() == ()


def test_ci_runs_both_braket_versions_and_the_braket_suite() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "braket-optional:" in workflow
    assert 'braket-version: ["1.117.0", "1.127.1"]' in workflow
    assert "python tools/check_braket_interop_contract.py --verify-sdk" in workflow
    assert (
        "tests/test_amazon_braket_provider.py tests/test_braket_iqm_dynamic.py"
        in workflow
    )
    assert (
        "tests/test_braket_interop.py tests/test_braket_interop_conformance.py"
        in workflow
    )
    assert "-m braket -q" in workflow
