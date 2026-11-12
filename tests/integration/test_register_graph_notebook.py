"""Execute the native register graph notebook in an independent Jupyter kernel."""

import json
import os
import sys
from pathlib import Path
from xml.etree import ElementTree

import pytest

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


def test_register_graph_notebook_kernel_without_reference_imports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
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
    path = ROOT / "examples/notebooks/register_graphs.ipynb"
    original = path.read_bytes()
    notebook = nbformat.reads(original.decode(), as_version=4)
    processor = preprocessors.ExecutePreprocessor(timeout=45, kernel_name=kernel_name)
    executed, _ = processor.preprocess(notebook, {"metadata": {"path": str(ROOT)}})
    cells = [cell for cell in executed.cells if cell.cell_type == "code"]
    assert len(cells) == 13
    assert [cell.execution_count for cell in cells] == list(range(1, 14))
    renders = [
        output.data["image/svg+xml"]
        for cell in cells
        for output in cell.outputs
        if output.output_type == "display_data" and "image/svg+xml" in output.data
    ]
    assert len(renders) == 10
    for i, svg in enumerate(renders):
        ElementTree.fromstring(svg)
        (tmp_path / f"native-{i}.svg").write_text(svg)
    assert all(
        output.output_type != "error" for cell in cells for output in cell.outputs
    )
    assert path.read_bytes() == original
    nbformat.write(executed, tmp_path / "executed-register-graphs.ipynb")
