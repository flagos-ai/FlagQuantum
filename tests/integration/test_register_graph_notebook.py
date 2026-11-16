"""Execute the native register graph notebook in an independent Jupyter kernel."""

import json
import os
import sys
from pathlib import Path
from xml.etree import ElementTree

import pytest

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "asset,code_count,svg_count,png_count",
    [
        ("register_graphs", 13, 10, 0),
        ("register_musical_score", 3, 1, 1),
        ("resource_call_graphs", 3, 2, 0),
        ("register_addition", 20, 6, 3),
        ("binary_field_addition", 7, 2, 0),
        ("binary_field_add_constant", 7, 2, 0),
        ("binary_field_square", 7, 2, 0),
        ("binary_field_polynomial_addition", 7, 2, 0),
        ("binary_field_polynomial_add_constant", 7, 2, 0),
        ("field_polynomial_views", 6, 2, 0),
        ("binary_field_multiplication", 32, 14, 0),
        ("binary_field_inverse", 7, 2, 0),
        ("block_encoding_interface", 3, 0, 0),
        ("register_chebyshev", 12, 4, 0),
        ("register_lcu_block_encoding", 22, 8, 0),
        ("register_linear_combination", 6, 2, 0),
        ("register_phase_block_encoding", 6, 2, 0),
        ("register_product_block_encoding", 8, 2, 0),
        ("register_sparse_matrix_block_encoding", 9, 2, 0),
        ("register_sparse_chemistry", 8, 4, 0),
        ("register_hermitian_sparse_matrix", 7, 2, 0),
        ("register_tensor_product_block_encoding", 8, 2, 0),
        ("register_unitary_block_encoding", 7, 2, 0),
        ("register_allocate_free", 10, 4, 0),
        ("register_always", 6, 2, 0),
        ("register_partition", 10, 3, 0),
        ("register_split2_join2", 8, 2, 0),
        ("register_cast", 5, 1, 0),
        ("register_split_join", 8, 2, 0),
        ("register_double_factorization", 11, 4, 1),
        ("register_single_factorization", 8, 4, 0),
        ("register_chemistry_resources", 12, 1, 7),
        ("register_thc_chemistry", 18, 9, 2),
        ("register_grid_trotter", 14, 12, 0),
        ("grid_trotter_costs", 5, 7, 3),
        ("register_hubbard_trotter", 7, 10, 0),
        ("hubbard_qpe_costs", 7, 1, 3),
        ("register_ising_trotter", 4, 4, 0),
        ("register_trotter_product", 4, 2, 0),
        ("chemistry_authoring", 5, 3, 3),
        ("register_hubbard_qubitization", 9, 4, 0),
        ("register_hubbard_select", 16, 4, 1),
        ("register_first_quantization", 29, 14, 0),
        ("register_first_quantization_projectile", 9, 4, 0),
        ("register_multiplication", 34, 16, 0),
        ("register_subtraction", 16, 4, 0),
        ("register_negation", 7, 2, 0),
        ("register_permutation", 16, 4, 0),
        ("register_sorting", 20, 8, 0),
        ("comparison_gate_costs", 36, 0, 0),
        ("register_trigonometric", 7, 2, 0),
        ("register_cnot", 7, 2, 0),
        ("register_diagonal_gates", 12, 3, 0),
        ("register_global_phase", 7, 1, 0),
        ("register_gradient_givens", 8, 4, 0),
        ("register_elliptic_curve_addition", 6, 2, 0),
        ("register_elliptic_curve_cryptography", 12, 9, 0),
        ("register_factoring_via_modexp", 6, 2, 0),
        ("register_rsa_phase_estimation", 6, 4, 0),
        ("register_qroam_clean", 6, 2, 0),
        ("register_qrom", 5, 2, 0),
        ("register_hadamard", 9, 2, 1),
        ("register_identity", 8, 2, 0),
        ("register_measurement", 10, 4, 0),
        ("register_rotation", 26, 10, 2),
        ("register_states_effects", 44, 21, 1),
        ("register_su2_rotation", 8, 2, 0),
        ("register_swap", 21, 6, 1),
        ("register_t_gate", 7, 2, 1),
        ("register_toffoli", 11, 2, 5),
        ("register_y_gate", 10, 2, 2),
    ],
)
def test_register_graph_notebook_kernel_without_reference_imports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    asset: str,
    code_count: int,
    svg_count: int,
    png_count: int,
) -> None:
    nbformat = pytest.importorskip("nbformat")
    preprocessors = pytest.importorskip("nbconvert.preprocessors")
    kernel_name = "fq-native-register-graph"
    kernel = tmp_path / "kernels" / kernel_name
    kernel.mkdir(parents=True)
    startup = """
import builtins
original_import = builtins.__import__
def native_import(name, *args, **kwargs):
    if name.split('.')[0] in {'qualtran', 'quimb', 'cirq'}:
        raise AssertionError(f'Native graph kernel imported {name}')
    return original_import(name, *args, **kwargs)
builtins.__import__ = native_import
from ipykernel.kernelapp import IPKernelApp
IPKernelApp.launch_instance()
"""
    (kernel / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-c", startup, "-f", "{connection_file}"],
                "display_name": "Native register graph verification",
                "language": "python",
            }
        )
    )
    monkeypatch.setenv("JUPYTER_PATH", str(tmp_path))
    monkeypatch.setenv("JUPYTER_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("IPYTHONDIR", str(tmp_path / "ipython"))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(map(str, (ROOT, *sys.path))))
    path = ROOT / f"examples/notebooks/{asset}.ipynb"
    original = path.read_bytes()
    notebook = nbformat.reads(original.decode(), as_version=4)
    processor = preprocessors.ExecutePreprocessor(timeout=45, kernel_name=kernel_name)
    executed, _ = processor.preprocess(notebook, {"metadata": {"path": str(tmp_path)}})
    cells = [cell for cell in executed.cells if cell.cell_type == "code"]
    assert len(cells) == code_count
    assert [cell.execution_count for cell in cells] == list(range(1, code_count + 1))
    renders = [
        output.data["image/svg+xml"]
        for cell in cells
        for output in cell.outputs
        if output.output_type == "display_data" and "image/svg+xml" in output.data
    ]
    assert len(renders) == svg_count
    assert (
        sum(
            "image/png" in output.get("data", {})
            for cell in cells
            for output in cell.outputs
        )
        == png_count
    )
    for i, svg in enumerate(renders):
        ElementTree.fromstring(svg)
        (tmp_path / f"native-{i}.svg").write_text(svg)
    assert all(
        output.output_type != "error" for cell in cells for output in cell.outputs
    )
    assert path.read_bytes() == original
    if asset == "register_musical_score":
        data = json.loads((tmp_path / "musical_score_example.json").read_text())
        assert (data["max_x"], data["max_y"]) == (3, 6)
        html = (tmp_path / "musical_score_example.html").read_text()
        payload = html.split('type="application/json">', 1)[1].split("</script>", 1)[0]
        assert json.loads(payload)["sequential"] == data
    nbformat.write(executed, tmp_path / f"executed-{asset}.ipynb")
