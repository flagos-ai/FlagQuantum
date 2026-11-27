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
        entry_name, arguments = json.loads(entry)
        getattr(module, entry_name)(*arguments)
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
        ("test_register_bindings", "numpy", None),
        ("test_register_binding_creation", "numpy", None),
        ("test_register_component_calls", "numpy", None),
        ("test_register_rebinding_contract", "numpy", None),
        ("test_register_tensor_binding", "numpy", None),
        (
            "test_basic_rotation_source_binding",
            "cirq",
            ("test_original_radian_adapter_round_trips", ("x",)),
        ),
        ("test_complex_phase_component_library", "numpy", None),
        (
            "test_generalized_qsp_component_library",
            "cirq",
            ("_check", (None, None, None)),
        ),
        ("test_generalized_qsp_entangling_unitary", "cirq", None),
        ("test_generalized_qsp_shaped_unitary", "cirq", None),
        ("test_generalized_qsp_stored_unitary", "cirq", None),
        ("test_hamiltonian_gqsp", "numpy", None),
        ("test_finite_field_inverse", "numpy", None),
        ("test_finite_field_multiplication", "numpy", None),
        ("test_finite_field_metadata", "numpy", None),
        ("test_finite_field_polynomial_encoding", "numpy", None),
        ("test_field_polynomial_addition", "numpy", None),
        ("test_field_polynomial_adjoint", "numpy", None),
        ("test_field_polynomial_component", "numpy", None),
        ("test_field_polynomial_constant_addition", "numpy", None),
        ("test_field_polynomial_file_audit", "numpy", None),
        ("test_field_polynomial_foreign_factories", "numpy", None),
        ("test_field_polynomial_layout", "numpy", None),
        ("test_typed_field_addition", "numpy", None),
        ("test_classical_arctangent", "numpy", None),
        ("test_classical_arctangent_math", "numpy", None),
        ("test_classical_register_codec", "numpy", None),
        ("test_classical_domains", "numpy", None),
        ("test_classical_feedback_tensors", "numpy", None),
        ("test_classical_parity_tensors", "numpy", None),
        ("test_classical_symbolic_metadata", "numpy", None),
        ("test_classical_tensor_crosscheck", "numpy", None),
        ("test_classical_tensor_discard", "numpy", None),
        ("test_classical_tensor_supplier", "numpy", None),
        ("test_classical_truth_table", "numpy", None),
        ("test_classical_typed_values", "numpy", None),
        ("test_classical_vector_source_audit", "numpy", None),
        ("test_component_classical_vectors", "numpy", None),
        ("test_classical_controlled_file_audit", "numpy", None),
        ("test_classical_source_callbacks", "numpy", None),
        ("test_classical_field_codec", "numpy", None),
        ("test_classical_field_registers", "numpy", None),
        ("register_source_graph", "cirq", ("source_line_qubit_index", (None,))),
        ("test_complex_phase_oracle", "cirq", None),
        ("test_hamming_phasing_file", "numpy", None),
        ("test_sparse_chemistry_prepare", "numpy", None),
        (
            "test_sparse_rotation_preparation",
            "numpy",
            ("test_original_sparse_state_grid", (None, 4)),
        ),
        (
            "test_hamiltonian_gqsp_component_library",
            "cirq",
            ("_bound_check", (None, None, None)),
        ),
        ("test_classical_action_tensor", "numpy", None),
        ("test_component_add_calls", "numpy", None),
        ("test_component_call_boundaries", "numpy", None),
        ("test_component_classical_state", "numpy", None),
        ("test_component_control", "numpy", None),
        ("test_component_expansion", "numpy", None),
        ("test_component_hierarchy_adjoint", "numpy", None),
        ("test_component_phase_output_lookup", "numpy", None),
        ("test_component_phased_evaluation", "numpy", None),
        ("test_component_phased_instance_ids", "numpy", None),
        ("test_component_tensor_expansion", "numpy", None),
        ("test_component_tensor_nodes", "numpy", None),
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
    entry: str | tuple[str, tuple[object, ...]] | None,
    required: bool,
) -> None:
    invocation = (
        (entry, (3, 4) if entry == "source_encoder" else ())
        if isinstance(entry, str)
        else entry
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _IMPORT_WITHOUT_SDK,
            "tests.unit." + module,
            root,
            json.dumps(invocation) if invocation is not None else "",
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
