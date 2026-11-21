"""Complete frozen notebook views and actual native process/kernel execution."""

import hashlib
import json
import os
import subprocess
import sys
import types
from importlib import import_module
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch

from examples.algorithms import component_tensor_report as native
from examples.algorithms.factorized_controlled_not import (
    controlled_not_chain,
    factorized_cnot_supply,
)
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.tensor_network.component_tensor_nodes import (
    component_tensor_nodes_with_labels,
)
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    contract_register_tensors,
)
from flagquantum.simulation.unitary import get_unitary
from flagquantum.testing.notebook_execution import execute_notebook_node
from tests.unit.test_component_report_notebook import execute_cells
from tests.unit.test_development_tools import REFERENCE
from tests.unit.test_notebook_execution import configure_kernel_environment
from tools.development import process_deadlines

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "examples/notebooks/component_tensor_report.ipynb"
ORIGINAL = REFERENCE / "dev_tools/tensor-report-card.ipynb"


def test_complete_source_blob_and_native_cell_structure() -> None:
    raw = ORIGINAL.read_bytes()
    assert (
        hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        == "49d08b1f9edad52672322b73315b31c32c51415a"
    )
    source = json.loads(raw)
    notebook = json.loads(NATIVE.read_text())
    assert len(notebook["cells"]) == len(source["cells"]) == 21
    assert [cell["cell_type"] for cell in notebook["cells"]] == [
        cell["cell_type"] for cell in source["cells"]
    ]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            compile("".join(cell["source"]), str(NATIVE), "exec")


@pytest.mark.parametrize("all_timeout", [False, True])
def test_all_original_cells_and_native_views(
    monkeypatch: pytest.MonkeyPatch, all_timeout: bool
) -> None:
    programs = tuple(
        (f"case_{index}", controlled_not_chain(index + 1)) for index in range(5)
    )
    rows: list[dict[str, object] | None] = [
        {
            "name": name,
            "cls": "ProgramRecord",
            "flat_dur": 5 - i,
            "tn_dur": i + 1,
            "width": width,
            "width_dur": (i + 1) / 10,
        }
        for i, ((name, _), width) in enumerate(
            zip(programs[:3], (0, 25, 26), strict=True)
        )
    ]
    rows.extend([{"name": "case_3", "cls": "ProgramRecord", "err": "no tensors"}, None])
    if all_timeout:
        rows = [None] * 5

    class Executor:
        def __init__(self, timeout: float, max_workers: int) -> None:
            assert timeout == 8.0 and max_workers == 4
            self.queue: list[dict[str, Any]] = []
            self.index = 0

        def submit(self, func: object, kwargs: dict[str, Any]) -> None:
            self.queue.append(kwargs)

        @property
        def work_to_be_done(self) -> int:
            return len(self.queue)

        def next_result(self) -> tuple[dict[str, Any], dict[str, object] | None]:
            kwargs = self.queue.pop(0)
            result = rows[self.index]
            self.index += 1
            return kwargs, result

        def __enter__(self) -> "Executor":
            return self

        def __exit__(self, *args: object) -> None:
            assert not self.queue

    skip = type("ApplyGateToLthQubit", (), {})
    examples = [
        types.SimpleNamespace(name=name, bloq_cls=type(record), make=lambda: None)
        for name, record in programs
    ] + [types.SimpleNamespace(bloq_cls=skip)]
    package = types.ModuleType("qualtran_dev_tools")
    package.__path__ = []
    finder = types.ModuleType("qualtran_dev_tools.bloq_finder")
    setattr(finder, "get_bloq_examples", lambda: examples)
    reporter = types.ModuleType("qualtran_dev_tools.tensor_report_card")
    setattr(reporter, "report_on_tensors", object())
    setattr(reporter, "ExecuteWithTimeout", Executor)
    skipped = types.ModuleType("qualtran.bloqs.multiplexers.apply_gate_to_lth_target")
    setattr(skipped, "ApplyGateToLthQubit", skip)
    for module in (package, finder, reporter, skipped):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(native, "selected_programs", lambda: programs)
    monkeypatch.setattr(process_deadlines, "ProcessDeadlineExecutor", Executor)
    if all_timeout:
        # Preserve the original all-timeout missing-column failure as evidence.
        with pytest.raises(KeyError, match="flat_dur"):
            execute_cells(ORIGINAL)
        namespace, views = execute_cells(NATIVE)
        assert len(namespace["df"]) == 5 and namespace["df"]["width"].isna().all()
        assert len(views[2]) == 5
    else:
        source, expected = execute_cells(ORIGINAL)
        namespace, actual = execute_cells(NATIVE)
        assert len(actual) == len(expected) == 6
        for left, right in zip(actual, expected, strict=True):
            pd.testing.assert_frame_equal(
                left, right.reindex(columns=namespace["df"].columns)
            )
        assert actual[0]["name"].tolist() == ["case_0", "case_1"]
        assert actual[1]["name"].tolist() == ["case_2"]
        assert actual[2]["name"].tolist() == ["case_3", "case_4"]
        assert source["records"] == namespace["records"]


@pytest.mark.parametrize("repetitions", [1, 2, 3])
def test_selected_graph_exact_operator_complex_binding_and_resources(
    repetitions: int,
) -> None:
    from functools import partial

    record = controlled_not_chain(repetitions)
    flat = native.flatten_calls(record)
    like = torch.empty((), dtype=torch.complex128)
    nodes, incoming, outgoing = component_tensor_nodes_with_labels(
        record,
        flat[1],
        partial(factorized_cnot_supply, like=like),
        like,
        "cnot_chain",
    )
    assert len(native.build_nodes(flat)) == len(nodes) == 2 * repetitions
    actual = contract_register_tensors(
        nodes, Signature(record.functions[0].registers or ()), incoming, outgoing
    )
    bound = bind_register_program(record, ((0,), (1,)), entry="cnot_chain")
    expected = get_unitary(bound.circuit, dtype=torch.complex128)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert bound.events == () and bound.outputs == ((0,), (1,))
    assert [gate.name for gate in bound.circuit.instructions] == ["cx"] * repetitions
    state = torch.complex(
        torch.arange(1, 9, dtype=torch.float64),
        torch.arange(8, 0, -1, dtype=torch.float64),
    )
    state /= state.norm()
    circuit = Circuit(3, inputs=state, dtype=torch.complex128)
    circuit.compose(bound.circuit, qubits=(0, 1))
    full = torch.kron(expected.contiguous(), torch.eye(2, dtype=torch.complex128))
    torch.testing.assert_close(circuit.state().flatten(), full @ state, rtol=0, atol=0)


def test_real_process_task_and_cleanup(tmp_path: Path) -> None:
    script = tmp_path / "tensor_report_spawn.py"
    script.write_text(
        "import multiprocessing\n"
        "import torch\n"
        "from examples.algorithms.component_tensor_report import selected_programs, report_component\n"
        "from tools.development.process_deadlines import ProcessDeadlineExecutor\n"
        "if __name__ == '__main__':\n"
        "    before = {child.pid for child in multiprocessing.active_children()}\n"
        "    with ProcessDeadlineExecutor(8.0, 4) as executor:\n"
        "        for name, program in selected_programs():\n"
        "            executor.submit(report_component, dict(name=name, cls_name='ProgramRecord', component=program))\n"
        "        rows = []\n"
        "        while executor.work_to_be_done:\n"
        "            _, row = executor.next_result()\n"
        "            assert isinstance(row, dict), row\n"
        "            rows.append(row)\n"
        "    assert len(rows) == 3\n"
        "    assert all('err' not in row and 0 < row['width'] <= 25 for row in rows), rows\n"
        "    assert {child.pid for child in multiprocessing.active_children()} == before\n"
    )
    completed = subprocess.run(
        [sys.executable, str(script)],
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), OMP_NUM_THREADS="1"),
        capture_output=True,
        text=True,
        timeout=35,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_real_native_notebook(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    nbformat = import_module("nbformat")
    kernel = configure_kernel_environment(tmp_path, monkeypatch)
    result, _ = execute_notebook_node(
        nbformat.read(NATIVE, as_version=4),
        timeout=60,
        record_timing=False,
        startup_lock=kernel / "lock",
    )
    outputs = [
        output
        for cell in result.cells
        if cell.cell_type == "code"
        for output in cell.outputs
    ]
    assert not any(output.output_type == "error" for output in outputs)
    text = "".join(output.text for output in outputs if output.output_type == "stream")
    assert "Measured primitive counts" in text and "Model primitive estimates" in text
    frames = [
        output.data.get("text/html", "")
        for output in outputs
        if output.output_type == "execute_result"
    ]
    assert len(frames) == 6
    assert all(f"cnot_chain_{i}" in frames[0] for i in (1, 2, 3))
    assert "Timeout" not in "".join(frames) and "NaN" not in frames[0].replace(
        "<td>NaN</td>", ""
    )
