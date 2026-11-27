"""Actual shared fixtures retain explicit optional and mandatory SDK absence."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_IMPORT_WITHOUT_SDK = """
import json
import os
import sys
from collections.abc import Sequence
from importlib import import_module
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType

module_name, root, entry, required, paths = sys.argv[1:]
sys.path[:] = json.loads(paths)
os.environ['FLAGQUANTUM_REQUIRE_FTQC_DEPENDENCIES'] = required
import pytest
assert not any(name == root or name.startswith(root + '.') for name in sys.modules)

class MissingSDK(MetaPathFinder):
    def find_spec(
        self, fullname: str, path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        if fullname == root or fullname.startswith(root + '.'):
            raise ModuleNotFoundError('Missing optional test dependency: ' + root, name=root)
        return None

sys.meta_path.insert(0, MissingSDK())
try:
    module = import_module(module_name)
    if entry:
        arguments = (3, 4) if entry == 'source_encoder' else ()
        getattr(module, entry)(*arguments)
except pytest.skip.Exception as error:
    result = {'outcome': 'explicit_skip', 'reason': str(error)}
except ModuleNotFoundError as error:
    result = {'outcome': 'missing', 'name': error.name, 'reason': str(error)}
else:
    raise AssertionError('The unavailable SDK was required by this fixture')
print(json.dumps(result))
"""


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize(
    "module,root,entry",
    [
        ("bit_view_call_checks", "numpy", None),
        ("classical_vector_fixtures", "numpy", None),
        ("matrix_register_fixture", "numpy", None),
        ("mean_cost_reference", "cirq", "source_encoder"),
        ("multi_register_fixture", "numpy", None),
        ("register_example_checks", "numpy", None),
        ("register_source_workspace", "cirq", None),
        ("rotation_lattice_fixtures", "numpy", None),
        ("test_controlled_unitary_synthesis", "numpy", None),
        ("test_register_su2_rotation", "numpy", None),
        ("test_cirq_decomposition_file", "numpy", None),
        ("test_program_classical_registers", "numpy", None),
        ("test_register_tensor", "numpy", None),
        (
            "test_global_phase_source_binding",
            "cirq",
            "test_original_external_scalar_conventions_and_static_refusal",
        ),
        (
            "test_foreign_phase_addition",
            "cirq",
            "test_private_phase_keys_do_not_alias_source_measurements",
        ),
    ],
)
def test_actual_fixture_dependency_absence(
    tmp_path: Path,
    module: str,
    root: str,
    entry: str | None,
    required: bool,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _IMPORT_WITHOUT_SDK,
            "tests.unit." + module,
            root,
            entry or "",
            "1" if required else "0",
            json.dumps(sys.path),
        ],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    if required:
        assert report["outcome"] == "missing" and report["name"] == root
    else:
        assert report["outcome"] == "explicit_skip"
        assert "Optional reference dependency is unavailable" in report["reason"]
    assert "Missing optional test dependency: " + root in report["reason"]
