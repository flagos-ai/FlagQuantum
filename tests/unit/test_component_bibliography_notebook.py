"""Complete bibliography source notebook and native discovery execution."""

import csv
import hashlib
import json
import sys
import types
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest

from flagquantum.testing.notebook_execution import execute_notebook_node
from tests.unit.test_development_tools import REFERENCE
from tests.unit.test_notebook_execution import configure_kernel_environment
from tools.development import component_discovery, docstrings
from tools.ftqc_source_inputs import FTQC_SOURCE_INVENTORY

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "examples/notebooks/component_bibliography.ipynb"
ORIGINAL = REFERENCE / "dev_tools/bibliography.ipynb"


def execute_cells(path: Path) -> tuple[dict[str, Any], list[str]]:
    rendered: list[str] = []

    def display(value: Any) -> None:
        rendered.append(value.data)

    namespace: dict[str, Any] = {"display": display}
    for index, cell in enumerate(json.loads(path.read_text())["cells"]):
        if cell["cell_type"] == "code":
            exec(
                compile("".join(cell["source"]), f"{path.name}:{index}", "exec"),
                namespace,
            )
    return namespace, rendered


def test_frozen_source_and_clean_native_notebook() -> None:
    with (FTQC_SOURCE_INVENTORY).open(encoding="utf-8-sig") as stream:
        row = next(row for row in csv.DictReader(stream) if row["id"] == "Q-0037")
    data = ORIGINAL.read_bytes()
    assert (
        hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        == row["git_blob"]
    )
    source = json.loads(data)
    native = json.loads(NATIVE.read_text())
    assert [cell["cell_type"] for cell in native["cells"]] == [
        cell["cell_type"] for cell in source["cells"]
    ]
    for cell in native["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell["source"]), str(NATIVE), "exec")


def test_complete_source_display_failures_and_backreferences(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def class_name(cls: type) -> str:
        return f"{cls.__module__}.{cls.__qualname__}"

    classes = [
        type(name, (), {"_class_name_in_pkg_": classmethod(class_name)})
        for name in ("Alpha", "Beta", "Empty")
    ]
    classes[0].__doc__ = """First component.

    References:
        [Shared](https://example.org/shared). First note.

        Unparsed first reference.
    """
    classes[1].__doc__ = """Second component.

    References:
        [Shared](https://example.org/shared). Second note.

        [Unique](https://example.org/unique). A unique reference.

        Unparsed second reference.
    """
    package = types.ModuleType("qualtran_dev_tools")
    package.__path__ = []
    finder = types.ModuleType("qualtran_dev_tools.bloq_finder")
    setattr(finder, "get_bloq_classes", lambda: classes)
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, finder.__name__, finder)
    monkeypatch.setitem(sys.modules, "qualtran_dev_tools.parse_docstrings", docstrings)
    monkeypatch.setattr(
        component_discovery, "get_component_classes", lambda **kwargs: classes
    )
    expected, source_rendered = execute_cells(ORIGINAL)
    source_errors = capsys.readouterr().out
    actual, native_rendered = execute_cells(NATIVE)
    assert source_errors == capsys.readouterr().out
    assert "Unparsed first reference" in source_errors
    assert "Unparsed second reference" in source_errors
    assert actual["component_classes"] == expected["bloq_classes"] == classes
    for key in ("references", "backrefs", "titles"):
        assert actual[key] == expected[key]
    assert source_rendered == native_rendered
    assert len(source_rendered) == 4
    assert actual["backrefs"]["https://example.org/shared"] == classes[:2]
    assert "Empty" not in "\n".join(native_rendered)


def test_real_native_bibliography_notebook(
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
    displays = [
        output.data["text/markdown"]
        for output in outputs
        if output.output_type == "display_data"
    ]
    assert any("KXorInstance" in text for text in displays)
    assert any(
        "Quartic quantum speedups for planted inference" in text for text in displays
    )
