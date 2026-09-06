from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import flagquantum as fq
from flagquantum._compiler.importers import circuit_ir as internal_importer
from tools import public_api_snapshot

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH_FILES = (
    ROOT / "flagquantum/__init__.py",
    ROOT / "flagquantum/api.py",
    ROOT / "flagquantum/runtime/planner/__init__.py",
    ROOT / "flagquantum/runtime/execution.py",
    ROOT / "flagquantum/runtime/plan_execution.py",
)


def _import_targets(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    targets = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            targets.append(node.module or "")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "import_module"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            targets.append(str(node.args[0].value))
    return tuple(targets)


def test_default_path_source_has_no_internal_compiler_dependency() -> None:
    for path in DEFAULT_PATH_FILES:
        assert not any("_compiler" in target for target in _import_targets(path)), path


def test_normal_import_and_stable_discovery_do_not_load_or_expose_compiler() -> None:
    script = """
import sys
import flagquantum as fq
assert 'flagquantum._compiler' not in sys.modules
assert '_compiler' not in fq.__all__
assert '_compiler' not in dir(fq)
_ = fq.plan
_ = fq.run
assert 'flagquantum._compiler' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr


def test_default_plan_and_run_do_not_call_internal_importer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(_: object) -> object:
        raise AssertionError("default path invoked internal CircuitIR importer")

    monkeypatch.setattr(internal_importer, "import_circuit_ir", forbidden)
    circuit = fq.Circuit(1).h(0)

    plan = fq.plan(circuit)
    result = fq.run(plan)

    assert result.plan is plan
    assert result.state is not None


def test_public_api_snapshot_is_unchanged() -> None:
    assert public_api_snapshot.validate() == ()
