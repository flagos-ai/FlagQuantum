"""Complete UI source export with the full reference catalog and native owners."""

import ast
import hashlib
import importlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch
from sympy import Symbol

from examples.algorithms import component_ui_export as example
from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.call_graph import SymbolAllocator
from flagquantum.core.ir.program.registers import RegisterSide
from flagquantum.errors import CapabilityError
from flagquantum.testing.component_examples import (
    ComponentDocumentation,
    ComponentExample,
)
from flagquantum.testing.notebook_execution import execute_notebook_node
from tests.unit.test_development_tools import REFERENCE
from tests.unit.test_notebook_execution import configure_kernel_environment
from tools.development import component_ui_export as native
from tools.development.library_call_graph import library_call_graph
from tools.development.notebook_autogen import NotebookSpec

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = REFERENCE / "dev_tools/ui-export.ipynb"
NOTEBOOK = ROOT / "examples/notebooks/component_ui_export.ipynb"


def original_cells() -> list[str]:
    raw = ORIGINAL.read_bytes()
    assert (
        hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        == "19959bb58f2218499ca2b927802ef79fa94cd88d"
    )
    return ["".join(cell["source"]) for cell in json.loads(raw)["cells"]]


def source_metadata() -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    for cell in original_cells()[1:3]:
        exec(compile(cell, str(ORIGINAL), "exec"), namespace)
    return namespace


def test_original_import_failure_and_native_cell_structure() -> None:
    with pytest.raises(ModuleNotFoundError, match="qualtran.bloqs.factoring"):
        exec(compile(original_cells()[0], str(ORIGINAL), "exec"), {})
    notebook = json.loads(NOTEBOOK.read_text())
    assert len(notebook["cells"]) == 6
    for cell in notebook["cells"]:
        assert cell["cell_type"] == "code"
        assert cell["outputs"] == [] and cell["execution_count"] is None
        compile("".join(cell["source"]), str(NOTEBOOK), "exec")


def test_all_scalar_encodings_hashes_and_wrapper_collision() -> None:
    from qualtran import Adjoint, Controlled, CtrlSpec
    from qualtran._infra.registers import Side
    from qualtran.bloqs.basic_gates import XGate

    source = source_metadata()
    values = [
        Symbol("theta"),
        complex(1, -2),
        RegisterSide.LEFT,
        RegisterSide.RIGHT,
        RegisterSide.THRU,
        object(),
    ]
    references = [
        Symbol("theta"),
        complex(1, -2),
        Side.LEFT,
        Side.RIGHT,
        Side.THRU,
        object(),
    ]
    assert json.dumps(values, cls=native.ComponentUIEncoder) == json.dumps(
        references, cls=source["BloqEncoder"]
    )
    attrs = {"z": complex(1, -2), "a": Symbol("theta")}
    pairs = [[key, attrs[key]] for key in sorted(attrs)]
    encoded = json.dumps(pairs, cls=source["BloqEncoder"])
    expected = (
        hashlib.md5(encoded.encode(), usedforsecurity=False).hexdigest() + ".json"
    )
    assert native.component_filename(attrs) == expected
    assert native.component_filename(dict(reversed(list(attrs.items())))) == expected
    for component, prefix in (
        (XGate(), ""),
        (Adjoint(XGate()), "Adjoint_"),
        (Controlled(XGate(), CtrlSpec()), "Controlled_"),
    ):
        assert native.component_filename(
            source["bloq_attrs"](component), prefix
        ) == source["bloq_filename"](component)
    # Source wrapper filenames omit control values and collide across polarities.
    assert source["bloq_filename"](Controlled(XGate(), CtrlSpec(cvs=0))) == source[
        "bloq_filename"
    ](Controlled(XGate(), CtrlSpec(cvs=1)))
    assert native.ComponentUIEncoder().encode(object()) == '"NOT_SERIALIZABLE"'


def test_full_catalog_export_protocol_with_supplied_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import matplotlib.pyplot as plt
    from qualtran import BloqExample, DecomposeNotImplementedError, DecomposeTypeError
    from qualtran.bloqs.cryptography.rsa.rsa_mod_exp import ModExp

    monkeypatch.syspath_prepend(str(REFERENCE / "openfermion-reference"))
    package = types.ModuleType("qualtran_dev_tools")
    package.__path__ = [str(REFERENCE / "dev_tools/qualtran_dev_tools")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    repository = importlib.import_module("qualtran_dev_tools.git_tools")
    monkeypatch.setattr(repository, "get_git_root", lambda: REFERENCE)
    original_graph = importlib.import_module("qualtran_dev_tools.all_call_graph")
    source_catalog = importlib.import_module("qualtran_dev_tools.notebook_specs")
    # The frozen notebook's obsolete import needs an explicit test-only route.
    obsolete = types.ModuleType("qualtran.bloqs.factoring.rsa")
    setattr(obsolete, "ModExp", ModExp)
    monkeypatch.setitem(sys.modules, obsolete.__name__, obsolete)
    made: dict[int, Any] = {}
    original_make = BloqExample.make

    def replay(factory: Any) -> Any:
        if id(factory) not in made:
            made[id(factory)] = original_make(factory)
        return made[id(factory)]

    monkeypatch.setattr(BloqExample, "make", replay)
    profiles: list[dict[str, Any]] = []
    builder = original_graph._build_call_graph

    def capture(**kwargs: Any) -> None:
        if not profiles:
            profiles.append(kwargs)
        builder(**kwargs)

    monkeypatch.setattr(original_graph, "_build_call_graph", capture)
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, Any] = {}
    try:
        for index, cell in enumerate(original_cells()):
            tree = ast.parse(cell)
            if index == 5:
                # Inject the diagram provider after its source definition, keeping
                # the original graph/node loop intact. Real all-node diagram
                # generation has a separate bounded external diagnostic.
                final_loop = tree.body.pop()
                exec(compile(tree, f"{ORIGINAL}:{index}", "exec"), namespace)
                namespace["bloq_score"] = lambda component: {
                    "supplied_diagram": type(component).__name__
                }
                tree.body = [final_loop]
            exec(compile(tree, f"{ORIGINAL}:{index}", "exec"), namespace)
    finally:
        plt.close("all")
    expected_graph = namespace["call_graph"]
    source_examples = namespace["examples"]
    assert len(expected_graph) == 3537 and len(expected_graph.edges) == 8244
    assert len(source_catalog.NB_BY_SECTION) == 11

    def declare(node: Any, allocator: SymbolAllocator) -> Any:
        try:
            return node.build_call_graph(allocator)
        except (DecomposeNotImplementedError, DecomposeTypeError) as error:
            raise CapabilityError(str(error)) from error

    examples = [
        ComponentExample(item.make, item.name, object) for item in source_examples
    ]
    adjacency = library_call_graph(
        examples,
        declare,
        generalizer=profiles[0]["generalizer"],
        keep=profiles[0]["keep"],
    )
    assert set(adjacency) == set(expected_graph.nodes)
    assert {
        (parent, child): count
        for parent, children in adjacency.items()
        for child, count in children.items()
    } == {
        (parent, child): data["n"]
        for parent, child, data in expected_graph.edges(data=True)
    }
    sections = []
    for title, specifications in source_catalog.NB_BY_SECTION:
        specs = []
        for specification in specifications:
            documents = [
                ComponentDocumentation(
                    component_type=document.bloq_cls,
                    examples=[
                        ComponentExample(
                            item.make, item.name, object, description=item.docstring
                        )
                        for item in document.examples
                    ],
                )
                for document in specification.bloq_specs
            ]
            specs.append(
                NotebookSpec(
                    title=specification.title,
                    module=specification.module,
                    component_specs=documents,
                )
            )
        sections.append((title, specs))
    native.export_component_ui(
        tmp_path / "native",
        sections,
        adjacency,
        name=namespace["bloq_name"],
        attributes=namespace["bloq_attrs"],
        filename=namespace["bloq_filename"],
        score=namespace["bloq_score"],
        encoder=namespace["BloqEncoder"],
    )
    expected = tmp_path / "ui_export"
    actual = tmp_path / "native"
    paths = sorted(
        path.relative_to(expected) for path in expected.rglob("*") if path.is_file()
    )
    assert paths == sorted(
        path.relative_to(actual) for path in actual.rglob("*") if path.is_file()
    )
    assert paths and Path("navigation.json") in paths
    for path in paths:
        assert (actual / path).read_bytes() == (expected / path).read_bytes(), path
    assert len(json.loads((actual / "navigation.json").read_text())) == 11


def test_native_owner_export_append_and_existing_file_semantics(tmp_path: Path) -> None:
    example.export_example(tmp_path)
    example_file = tmp_path / "WordSwap" / example.filename(example.make_swap())
    payload = json.loads(example_file.read_text())
    assert payload["attrs"] == {"width": 2}
    assert payload["msd"] == example.score(example.make_swap())
    assert payload["callees"] == [
        {
            "name": "ComponentCost",
            "filename": example.filename(
                next(iter(example.declaration(example.make_swap(), SymbolAllocator())))
            ),
        }
    ]
    assert len(list(tmp_path.rglob("*.json"))) == 3
    docs = tmp_path / "WordSwap/docs.txt"
    docs.write_text("authored documentation")
    example_file.write_text("retained example")
    navigation = tmp_path / "navigation.json"
    navigation.write_text("stale navigation")
    example.export_example(tmp_path)
    assert docs.read_text() == "authored documentation"
    assert example_file.read_text() == "retained example"
    assert json.loads(navigation.read_text())["Register components"]["Word exchange"]
    values = torch.arange(1, 33, dtype=torch.float64)
    state = torch.complex(values, values.flip(0))
    state /= state.norm()
    circuit = Circuit(5, inputs=state, dtype=torch.complex128)
    circuit.append(example.make_swap().make_on((0, 1), (2, 3)))
    expected = state.reshape(4, 4, 2).transpose(0, 1).flatten()
    torch.testing.assert_close(circuit.state().flatten(), expected, atol=0, rtol=0)
    assert len(circuit.to_ir().instructions) == 6


def test_score_errors_and_cancellation_metadata_and_partial_output(
    tmp_path: Path,
) -> None:
    def fail_score(component: object) -> object:
        raise ValueError("no score")

    def export(path: Path, score: Any, attributes: Any = lambda value: {}) -> None:
        native.export_component_ui(
            path,
            (),
            {"root": ("leaf",), "leaf": ()},
            name=lambda value: "Node",
            attributes=attributes,
            filename=lambda value: str(value) + ".json",
            score=score,
        )

    export(tmp_path, fail_score)
    assert json.loads((tmp_path / "Node/root.json").read_text())["msd"] is None

    def interrupt(component: object) -> object:
        raise KeyboardInterrupt("cancel export")

    with pytest.raises(KeyboardInterrupt, match="cancel export"):
        export(tmp_path / "interrupt", interrupt)
    assert (tmp_path / "interrupt/navigation.json").is_file()
    assert not (tmp_path / "interrupt/Node/root.json").exists()

    events = []

    def bad_attributes(component: object) -> Any:
        events.append("attributes")
        raise ValueError("metadata failed")

    def tracked_score(component: object) -> object:
        events.append("score")
        return None

    with pytest.raises(ValueError, match="metadata failed"):
        export(tmp_path / "metadata", tracked_score, bad_attributes)
    assert (tmp_path / "metadata/navigation.json").is_file()
    assert events == ["attributes"]
    # Cyclic metadata is still a JSON failure rather than an unknown scalar.
    cyclic: list[object] = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="Circular reference"):
        native.component_filename({"cycle": cyclic})


def test_real_native_ui_notebook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nbformat = importlib.import_module("nbformat")
    kernel = configure_kernel_environment(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    result, _ = execute_notebook_node(
        nbformat.read(NOTEBOOK, as_version=4),
        timeout=90,
        record_timing=False,
        startup_lock=kernel / "lock",
    )
    outputs = [output for cell in result.cells for output in cell.get("outputs", ())]
    assert not any(output.output_type == "error" for output in outputs)
    streams = "".join(
        output.text for output in outputs if output.output_type == "stream"
    )
    assert "Measured primitive counts {'cx': 6}" in streams
    assert "Model primitive estimates {'cx': 6}" in streams
    assert "Exported JSON files 3" in streams
    assert (tmp_path / "component_ui_export/navigation.json").is_file()


def test_actual_module_entry(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "examples.algorithms.component_ui_export"],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Measured primitive counts {'cx': 6}" in result.stdout
    assert "Model primitive estimates {'cx': 6}" in result.stdout
    assert len(list((tmp_path / "component_ui_export").rglob("*.json"))) == 3
