"""Complete report notebook views, upstream counterexamples and real checks."""

import ast
import csv
import hashlib
import json
import sys
import types
from importlib import import_module
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch

from examples.algorithms.register_component_tutorial import WordSwap
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.testing.example_checks import CheckStatus
from flagquantum.testing.notebook_execution import execute_notebook_node
from tests.unit.test_development_tools import REFERENCE
from tests.unit.test_notebook_execution import configure_kernel_environment
from tools.development import component_report
from tools.ftqc_source_inputs import FTQC_SOURCE_INVENTORY

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "examples/notebooks/component_report.ipynb"
ORIGINAL = REFERENCE / "dev_tools/bloq-report-card.ipynb"


def execute_cells(path: Path) -> tuple[dict[str, Any], list[Any]]:
    namespace: dict[str, Any] = {}
    results = []
    for index, cell in enumerate(json.loads(path.read_text())["cells"]):
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        expression = None
        last = tree.body[-1]
        if isinstance(last, ast.Expr):
            expression = last.value
            tree.body.pop()
        filename = f"{path.name}:{index}"
        exec(compile(tree, filename, "exec"), namespace)
        if expression is not None:
            value = eval(
                compile(ast.Expression(expression), filename, "eval"), namespace
            )
            if value is not None:
                results.append(value)
    return namespace, results


def test_frozen_source_and_complete_native_cells() -> None:
    with (FTQC_SOURCE_INVENTORY).open(encoding="utf-8-sig") as stream:
        row = next(row for row in csv.DictReader(stream) if row["id"] == "Q-0040")
    data = ORIGINAL.read_bytes()
    assert (
        hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        == row["git_blob"]
    )
    original = json.loads(data)
    native = json.loads(NATIVE.read_text())
    assert [cell["cell_type"] for cell in native["cells"]] == [
        cell["cell_type"] for cell in original["cells"]
    ]
    for cell in native["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            compile("".join(cell["source"]), str(NATIVE), "exec")


def test_complete_source_views_and_explicit_grouping_counterexamples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packages = [
        "basic_gates",
        "arithmetic",
        "chemistry.sub",
        "factoring",
        "factoring.sub",
        "other",
    ]
    report = pd.DataFrame(
        [
            {
                "package": package,
                "component_cls": f"Class{index}",
                "name": f"example{index}",
            }
            | dict.fromkeys(component_report.CHECK_COLUMNS, status)
            for index, (package, status) in enumerate(
                zip(packages, CheckStatus, strict=True)
            )
        ]
    )
    package = types.ModuleType("qualtran_dev_tools")
    package.__path__ = []
    owner = types.ModuleType("qualtran_dev_tools.bloq_report_card")
    setattr(owner, "get_bloq_report_card", lambda: report.copy())
    setattr(owner, "show_bloq_report_card", component_report.show_report)
    setattr(owner, "summarize_results", component_report.summarize_results)
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, owner.__name__, owner)
    native_report = report.copy()
    native_report["package"] = [
        "core.ir.program",
        "compiler.registers",
        "algorithms.chemistry.sub",
        "algorithms.primitives",
        "algorithms.primitives.sub",
        "other",
    ]
    monkeypatch.setattr(
        component_report, "get_report", lambda *args, **kwargs: native_report.copy()
    )
    _, expected = execute_cells(ORIGINAL)
    namespace, actual = execute_cells(NATIVE)
    assert len(expected) == len(actual) == 10
    labels = dict(zip(packages, native_report["package"], strict=True))
    for index in (0, 2, 4, 6, 8):
        mapped = expected[index].data.copy()
        mapped["package"] = mapped["package"].map(labels)
        pd.testing.assert_frame_equal(
            actual[index].data,
            mapped,
        )
    for index in (1, 3, 5, 7, 9):
        pd.testing.assert_frame_equal(actual[index], expected[index])
    # Source styling excludes a factoring child that its own summary includes.
    assert expected[6].data["name"].tolist() == ["example3"]
    assert actual[6].data["name"].tolist() == ["example3"]
    # Source Other includes stripped chemistry names and factoring children.
    assert expected[8].data["name"].tolist() == ["example2", "example4", "example5"]
    assert actual[8].data["name"].tolist() == ["example2", "example4", "example5"]
    assert namespace["observed_counts"] == namespace["declared_counts"] == {"cx": 6}


def test_actual_owned_checks_counts_and_append_complex_state() -> None:
    namespace, results = execute_cells(NATIVE)
    report = namespace["report_card"]
    assert len(report) == 1 and report.iloc[0]["package"] == "core.ir.program"
    assert all(
        report.iloc[0][column] is CheckStatus.PASS
        for column in component_report.CHECK_COLUMNS
    )
    assert results[-1]["pass"].tolist() == [1] * 5
    assert namespace["declared_counts"] == {"cx": 6}
    assert dict(namespace["observed_counts"]) == {"cx": 6}
    bound = bind_register_program(
        namespace["make_program"](), ((0, 1), (2, 3)), entry="Swap"
    )
    assert bound.events == () and bound.outputs == ((0, 1), (2, 3))
    values = torch.arange(32, dtype=torch.float64)
    state = torch.complex(values + 1, values.flip(0) + 5)
    state /= state.norm()
    circuit = Circuit(5, inputs=state, dtype=torch.complex128)
    circuit.append(WordSwap(2).make_on((0, 1), (2, 3)))
    expected = state.reshape(4, 4, 2).transpose(0, 1).flatten()
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=0)
    assert len(circuit.to_ir().instructions) == 6
    namespace["declared_counts"]["cx"] = 5
    failed = namespace["check_component"](namespace["example"], "counts")
    assert failed.status is CheckStatus.FAIL


def test_real_native_report_notebook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    assert "Measured primitive counts {'cx': 6}" in text
    assert "Declared primitive counts {'cx': 6}" in text
    html = "\n".join(
        output.data.get("text/html", "")
        for output in outputs
        if output.output_type == "execute_result"
    )
    assert "two_bit_word_swap" in html and "pass" in html
