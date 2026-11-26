"""Complete source override-report behavior and native declaration ownership."""

import contextlib
import csv
import hashlib
import importlib.util
import io
import os
import runpy
import subprocess
import sys
import types
from pathlib import Path

import pytest

from flagquantum.compiler._component_trace.call_graph import (
    SymbolAllocator,
    symbolic_call_graph,
)
from tests.unit.test_development_tools import REFERENCE, load_source
from tools.ftqc_source_inputs import FTQC_SOURCE_INVENTORY

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]
SOURCE = REFERENCE / "dev_tools/bloq-method-overrides-report.py"


@pytest.fixture
def original(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    package = types.ModuleType("_development_reference")
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(
        sys.modules, package.__name__ + ".git_tools", load_source("git_tools")
    )
    finder = load_source("bloq_finder")
    namespace = types.ModuleType("qualtran_dev_tools")
    namespace.__path__ = []
    monkeypatch.setitem(sys.modules, namespace.__name__, namespace)
    monkeypatch.setitem(sys.modules, namespace.__name__ + ".bloq_finder", finder)
    spec = importlib.util.spec_from_file_location("_method_report_reference", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    setattr(module, "finder", finder)
    return module


def test_complete_original_blob() -> None:
    with (FTQC_SOURCE_INVENTORY).open(encoding="utf-8-sig") as stream:
        row = next(row for row in csv.DictReader(stream) if row["id"] == "Q-0038")
    data = SOURCE.read_bytes()
    assert (
        hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        == row["git_blob"]
    )


def fixture_class(original: types.ModuleType, variant: str) -> type:
    def declaration(self: object, ssa: object) -> object:
        return {}

    declaration.__annotations__ = {
        "ssa": "SympySymbolAllocator",
        "return": getattr(original, "Set")[
            getattr(original, "ForwardRef")("BloqCountT")
        ],
    }
    members: dict[str, object] = {"build_call_graph": declaration}
    if variant in {"call_graph", "bloq_counts"}:
        members[variant] = lambda self: None
    elif variant == "extra":
        declaration.__annotations__["extra"] = int
    elif variant == "missing_ssa":
        del declaration.__annotations__["ssa"]
    elif variant == "missing_return":
        del declaration.__annotations__["return"]
    elif variant == "allocator":
        declaration.__annotations__["ssa"] = object
    elif variant == "return":
        declaration.__annotations__["return"] = dict[str, int]
    return type("Case", (original.Bloq,), members)


@pytest.mark.parametrize(
    "variant",
    [
        "valid",
        "call_graph",
        "bloq_counts",
        "extra",
        "missing_ssa",
        "missing_return",
        "allocator",
        "return",
    ],
)
def test_original_all_diagnostics_and_failure_edges(
    original: types.ModuleType, capsys: pytest.CaptureFixture[str], variant: str
) -> None:
    cls = fixture_class(original, variant)
    if variant == "call_graph":
        with pytest.raises(ValueError, match="Case"):
            original._call_graph(cls)
    elif variant in {"missing_ssa", "missing_return"}:
        with pytest.raises(
            KeyError, match="ssa" if variant == "missing_ssa" else "return"
        ):
            original._call_graph(cls)
    else:
        original._call_graph(cls)
    messages = {
        "valid": "",
        "call_graph": f"{cls}.call_graph should not be overridden.\n",
        "bloq_counts": f"{cls}.bloq_counts should not be overriden.\n",
        "extra": f"{cls}.build_call_graph should have one argument named `ssa` and a return type annotation\n",
        "missing_ssa": f"{cls}.build_call_graph should have one argument named `ssa` and a return type annotation\n",
        "missing_return": f"{cls}.build_call_graph should have one argument named `ssa` and a return type annotation\n",
        "allocator": f"{cls}.build_call_graph `ssa: 'SympySymbolAllocator'`\n",
        "return": f"{cls}.build_call_graph -> 'BloqCountT'\n",
    }
    assert capsys.readouterr().out == messages[variant]


@pytest.mark.parametrize("entry", ["report_call_graph_methods", "main", "__main__"])
def test_complete_original_entry_points_and_failure_order(
    original: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    entry: str,
) -> None:
    classes = [
        fixture_class(original, value) for value in ("return", "call_graph", "return")
    ]
    monkeypatch.setattr(original, "get_bloq_classes", lambda: classes)
    monkeypatch.setattr(original.finder, "get_bloq_classes", lambda: classes)
    with pytest.raises(ValueError, match="Case"):
        if entry == "__main__":
            runpy.run_path(str(SOURCE), run_name="__main__")
        else:
            getattr(original, entry)()
    output = capsys.readouterr().out
    assert output.count("-> 'BloqCountT'") == 1
    assert output.count("should not be overridden") == 1


def test_complete_340_class_source_diagnostics(original: types.ModuleType) -> None:
    classes = original.finder.get_bloq_classes(REFERENCE / "qualtran/bloqs")
    assert len(classes) == 340
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        for cls in classes:
            original._call_graph(cls)
    lines = stream.getvalue().splitlines()
    expected = []
    allocator_warnings = {
        "qualtran.bloqs.arithmetic._shims.CHalf",
        "qualtran.bloqs.arithmetic._shims.CSub",
        "qualtran.bloqs.arithmetic._shims.Lt",
        "qualtran.bloqs.arithmetic._shims.MultiCToffoli",
        "qualtran.bloqs.arithmetic.trigonometric.arcsin.ArcSin",
        "qualtran.bloqs.arithmetic.trigonometric.arctan.ArcTan",
        "qualtran.bloqs.block_encoding.phase.Phase",
        "qualtran.bloqs.block_encoding.product.Product",
        "qualtran.bloqs.block_encoding.sparse_matrix.SparseMatrix",
        "qualtran.bloqs.block_encoding.sparse_matrix_hermitian.SparseMatrixHermitian",
        "qualtran.bloqs.block_encoding.tensor_product.TensorProduct",
        "qualtran.bloqs.block_encoding.unitary.Unitary",
        "qualtran.bloqs.mod_arithmetic.mod_multiplication.CModMulK",
        "qualtran.bloqs.mod_arithmetic.mod_multiplication.ModDbl",
        "qualtran.bloqs.qft.qft_phase_gradient.QFTPhaseGradient",
        "qualtran.bloqs.qft.qft_text_book.QFTTextBook",
        "qualtran.bloqs.rotations.phase_gradient.PhaseGradientUnitary",
        "qualtran.bloqs.rotations.programmable_ancilla_rotation.ZPowUsingProgrammedAncilla",
        "qualtran.bloqs.rotations.zpow_via_phase_gradient.ZPowConstViaPhaseGradient",
    }
    for cls in classes:
        if cls.__module__ + "." + cls.__name__ in allocator_warnings:
            expected.append(f"{cls}.build_call_graph `ssa: 'SympySymbolAllocator'`")
        expected.append(f"{cls}.build_call_graph -> 'BloqCountT'")
    assert len(lines) == len(expected) == 359
    assert lines == expected


@pytest.mark.parametrize(
    "variant",
    [
        "mapping",
        "iterable",
        "wrong_allocator",
        "wrong_return",
        "missing_allocator",
        "untyped",
    ],
)
def test_existing_strict_gate_checks_native_declaration_contract(
    tmp_path: Path, variant: str
) -> None:
    definitions = {
        "mapping": "def declare(component: str, symbols: SymbolAllocator) -> dict[str, int]:\n    return {}\n",
        "iterable": "def declare(component: str, symbols: SymbolAllocator) -> tuple[tuple[str, int], ...]:\n    return ()\n",
        "wrong_allocator": "def declare(component: str, symbols: str) -> dict[str, int]:\n    return {}\n",
        "wrong_return": "def declare(component: str, symbols: SymbolAllocator) -> int:\n    return 1\n",
        "missing_allocator": "def declare(component: str) -> dict[str, int]:\n    return {}\n",
        "untyped": "def declare(component, symbols):\n    return {}\n",
    }
    file = tmp_path / "declaration.py"
    file.write_text(
        "from flagquantum.compiler._component_trace.call_graph import CallDeclaration, SymbolAllocator\n"
        + definitions[variant]
        + "declaration: CallDeclaration[str] = declare\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--config-file=" + str(ROOT / "pyproject.toml"),
            "--strict",
            "--explicit-package-bases",
            "--follow-imports=silent",
            "--cache-dir=/dev/null",
            str(file),
        ],
        cwd=ROOT,
        env=dict(os.environ, PYTHONPATH=str(ROOT)),
        text=True,
        capture_output=True,
        timeout=120,
    )
    valid = variant in {"mapping", "iterable"}
    assert result.returncode == int(not valid), result.stdout + result.stderr
    if not valid:
        code = "[no-untyped-def]" if variant == "untyped" else "[assignment]"
        assert code in result.stdout


def test_declaration_object_cannot_replace_traversal_or_aggregation() -> None:
    class Declaration:
        def __call__(self, component: str, symbols: SymbolAllocator) -> dict[str, int]:
            if component == "root":
                return {"branch": 2}
            if component == "branch":
                return {"leaf": 3}
            return {}

        def call_graph(self) -> None:
            raise AssertionError("Traversal belongs to Compiler")

        def bloq_counts(self) -> None:
            raise AssertionError("Aggregation belongs to Compiler")

    graph = symbolic_call_graph("root", Declaration())
    assert graph.direct_calls == {("root", "branch"): 2, ("branch", "leaf"): 3}
    assert graph.leaf_counts == {"leaf": 6}
