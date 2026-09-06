from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from flagquantum.ecosystem.pennylane import PennyLaneDependencyError, conversion
from tools.check_architecture import CONFIG, architecture_errors

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_pennylane_namespace_and_registry_are_lazy() -> None:
    code = """
import sys
import flagquantum.ecosystem.pennylane
from flagquantum.ecosystem import available_adapters, get_adapter
assert available_adapters() == ('pennylane', 'qiskit')
assert get_adapter('pennylane').name == 'pennylane'
assert not [name for name in sys.modules if name == 'pennylane' or name.startswith('pennylane.')]
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_missing_pennylane_fails_only_when_conversion_is_requested(monkeypatch) -> None:
    monkeypatch.setattr(
        conversion,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError(name)),
    )
    with pytest.raises(PennyLaneDependencyError, match=r"flagquantum\[pennylane\]"):
        conversion.from_pennylane(object())


def test_architecture_isolates_pennylane_imports() -> None:
    assert CONFIG["interop_boundaries"]["pennylane_import_allowed_prefixes"] == [
        "flagquantum/ecosystem/pennylane/"
    ]
    assert architecture_errors() == ()


def test_ci_has_minimum_and_latest_pennylane_lanes() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "pennylane-optional:" in workflow
    assert 'pennylane-version: ["0.44.1", "0.45.1"]' in workflow
    assert "python tools/check_pennylane_interop_contract.py" in workflow
    assert "python -m pytest -m pennylane -q" in workflow
