"""The OpenQASM import boundary, and the CI that enforces it.

Import sits in ``compiler`` and constructs a program. It must not reach for the
planner or an executor, and the gate spelling tables it shares with the emitter
must stay the only copy of those spellings. These tests check the boundary
directly and confirm the contract gate is wired into the required workflows.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tools.check_openqasm_import_contract import (
    IMPORTER,
    gate_tables_defined_once,
    ir_version_unchanged,
)
from tools.check_openqasm_import_contract import (
    ROOT as TOOL_ROOT,
)
from tools.pre_push import checks

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]


def test_importing_the_importer_loads_no_runtime_or_simulation_module() -> None:
    """Reading text back is not running it, and must not pull in the runtime.

    The subprocess is the assertion: within this test process the runtime is
    already imported by other tests, so only a fresh interpreter can show that
    the importer does not request it.
    """

    code = """
import sys
import flagquantum.compiler.openqasm_import
loaded = sorted(
    name for name in sys.modules
    if name.startswith('flagquantum.runtime')
    or name.startswith('flagquantum.simulation')
)
assert not loaded, loaded
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_the_tool_and_the_importer_resolve_the_same_root() -> None:
    assert TOOL_ROOT == ROOT
    assert IMPORTER == ROOT / "flagquantum" / "compiler" / "openqasm_import.py"


def test_emission_and_import_share_one_copy_of_each_gate_table() -> None:
    """A second table is how the two directions would silently diverge."""

    assert gate_tables_defined_once() == ()


def test_the_interchange_did_not_move_the_ir_version() -> None:
    import json

    contract = json.loads(
        (ROOT / "contracts" / "openqasm-import-v1-candidate.json").read_text(
            encoding="utf-8"
        )
    )
    assert ir_version_unchanged(contract) == ()


def test_the_contract_gate_is_a_required_pre_push_check() -> None:
    commands = {check.command for check in checks("python")}
    assert ("python", "tools/check_openqasm_import_contract.py") in commands


def test_ci_runs_the_contract_gate() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "python tools/check_openqasm_import_contract.py" in workflow
    assert "Verify OpenQASM import contract" in workflow
