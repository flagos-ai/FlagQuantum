"""Complete notebook source behavior, native rendering and executable cells."""

import csv
import hashlib
import json
import subprocess
import sys
import types
from importlib import import_module
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pytest
from nbformat import NotebookNode

from examples.algorithms.register_component_tutorial import WordSwap
from flagquantum.testing.component_examples import (
    ComponentDocumentation,
    ComponentExample,
)
from tests.unit.test_development_tools import REFERENCE, load_source
from tools.development import notebook_autogen as native
from tools.development import notebook_cleaning as cleaning
from tools.ftqc_source_inputs import FTQC_SOURCE_INVENTORY

# nbformat exposes untyped optional I/O; validate nodes at this boundary.
nbformat = import_module("nbformat")

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture
def original(monkeypatch: pytest.MonkeyPatch) -> Any:
    package = types.ModuleType("_development_reference")
    package.__path__ = [str(REFERENCE / "dev_tools" / "qualtran_dev_tools")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    return load_source("jupyter_autogen")


def _swap_example() -> WordSwap:
    swap_example = WordSwap(3)
    return swap_example


def convert_example(example: Any) -> ComponentExample[object]:
    return ComponentExample(example._func, example.name, example.bloq_cls)


def convert_documentation(documentation: Any) -> ComponentDocumentation[object]:
    graph = documentation.call_graph_example
    return ComponentDocumentation(
        component_type=documentation.bloq_cls,
        examples=[convert_example(example) for example in documentation.examples],
        import_line=documentation.import_line,
        call_graph_example=None if graph is None else convert_example(graph),
    )


def source_cells(original: Any, documentation: Any) -> list[NotebookNode]:
    return [
        original._cell_to_nbnode(cell) for cell in original.get_cells(documentation)
    ]


def native_cells(original: Any, documentation: Any) -> list[NotebookNode]:
    return native.get_cells(
        convert_documentation(documentation),
        graphical_signature_code=lambda doc: original.get_graphical_signature_cells(
            documentation, "p"
        )[1].text,
        call_graph_code=lambda example: original.get_call_graph_cells(
            documentation, "p"
        )[1].text,
    )


def normalized(notebook: NotebookNode) -> dict[str, Any]:
    payload = json.loads(nbformat.writes(notebook))
    for cell in payload.get("cells", []):
        cell.pop("id", None)
    return dict(payload)


def test_source_blobs() -> None:
    remaining = {"Q-0059", "Q-0063", "Q-0064"}
    with (FTQC_SOURCE_INVENTORY).open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row["id"] not in remaining:
                continue
            raw = (REFERENCE / row["path"]).read_bytes()
            assert (
                hashlib.sha1(
                    b"blob " + str(len(raw)).encode() + b"\0" + raw
                ).hexdigest()
                == row["git_blob"]
            )
            remaining.remove(row["id"])
    assert not remaining


def test_all_unchanged_original_tests_use_native_consumer(
    original: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = types.ModuleType("_development_reference.jupyter_autogen")

    def spec(**kwargs: Any) -> Any:
        item = native.NotebookSpec(
            title=kwargs["title"],
            module=kwargs["module"],
            component_specs=[
                convert_documentation(doc) for doc in kwargs["bloq_specs"]
            ],
        )
        return types.SimpleNamespace(
            title=item.title, module=item.module, bloq_specs=item.component_specs
        )

    def cells(documentation: Any) -> list[Any]:
        result = native_cells(original, documentation)
        return [
            (
                original._MarkdownCell
                if cell.cell_type == "markdown"
                else original._PyCell
            )(cell.source, cell.metadata["cq.autogen"])
            for cell in result
        ]

    setattr(provider, "NotebookSpecV2", spec)
    setattr(
        provider,
        "_get_bloq_example_source_lines",
        lambda example: native.example_source_lines(convert_example(example)),
    )
    setattr(provider, "_MarkdownCell", original._MarkdownCell)
    setattr(provider, "_PyCell", original._PyCell)
    setattr(provider, "get_cells", cells)
    monkeypatch.setitem(sys.modules, provider.__name__, provider)
    tests = load_source("jupyter_autogen_test")
    tests.test_notebook_spec()
    tests.test_get_bloq_example_source_lines()
    tests.test_get_cells()


@pytest.mark.parametrize(
    "examples,graph", [(False, False), (True, False), (True, True)]
)
def test_complete_cell_sequence_and_example_corpus(
    original: Any, examples: bool, graph: bool
) -> None:
    from qualtran import BloqDocSpec
    from qualtran.bloqs.for_testing import TestAtom

    tests = load_source("jupyter_autogen_test")
    example = tests._my_bloq_example
    kwargs = {} if graph else {"call_graph_example": None}
    documentation = BloqDocSpec(
        bloq_cls=TestAtom, examples=[example] if examples else [], **kwargs
    )
    expected = source_cells(original, documentation)
    actual = native_cells(original, documentation)
    assert normalized(nbformat.v4.new_notebook(cells=actual)) == normalized(
        nbformat.v4.new_notebook(cells=expected)
    )
    assert native.example_source_lines(
        convert_example(example)
    ) == original._get_bloq_example_source_lines(example)


def test_complete_frozen_factory_source_extraction(original: Any) -> None:
    finder = load_source("bloq_finder")
    examples = finder.get_bloq_examples(REFERENCE / "qualtran" / "bloqs")
    assert len(examples) == 393
    for example in examples:
        try:
            expected = original._get_bloq_example_source_lines(example)
        except (ValueError, OSError, TypeError) as error:
            with pytest.raises(type(error)):
                native.example_source_lines(convert_example(example))
        else:
            assert (
                native.example_source_lines(convert_example(example)) == expected
            ), example.name


def test_spec_paths_titles_initialization_and_merge(
    original: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = types.ModuleType("native.example", "Module introduction.\n    Continued.")
    module.__file__ = str(tmp_path / "module.py")
    spec = native.NotebookSpec(title="Title", module=module, component_specs=[])
    expected_spec = original.NotebookSpecV2(title="Title", module=module, bloq_specs=[])
    assert (
        spec.directory == expected_spec.directory
        and spec.path_stem == expected_spec.path_stem
    )
    assert spec.path == expected_spec.path
    assert native._get_title_lines("Title", module) == original._get_title_lines(
        "Title", module
    )
    spec = native.NotebookSpec(
        title="Title",
        module=module,
        component_specs=[],
        directory=str(tmp_path),
        path_stem="native",
    )
    expected_spec = original.NotebookSpecV2(
        title="Title",
        module=module,
        bloq_specs=[],
        directory=str(tmp_path),
        path_stem="source",
    )
    # Equal source/native import text isolates the merge algorithm from native API code.
    native.render_notebook(
        spec,
        graphical_signature_code=lambda doc: "",
        call_graph_code=lambda example: "",
        imports=original._IMPORTS,
    )
    original.render_notebook(expected_spec)
    actual = nbformat.read(spec.path, as_version=4)
    expected = nbformat.read(expected_spec.path, as_version=4)
    assert normalized(actual) == normalized(expected)
    for notebook in (actual, expected):
        notebook.cells.append(
            nbformat.v4.new_markdown_cell("Authored content", id="manual")
        )
        notebook.cells.append(
            nbformat.v4.new_code_cell(
                "retired", metadata={"cq.autogen": "retired"}, id="retired"
            )
        )
        notebook.cells[0].id = "title-id"
    nbformat.write(actual, spec.path)
    nbformat.write(expected, expected_spec.path)
    native.render_notebook(
        spec,
        graphical_signature_code=lambda doc: "",
        call_graph_code=lambda example: "",
        imports=original._IMPORTS,
    )
    original.render_notebook(expected_spec)
    actual = nbformat.read(spec.path, as_version=4)
    expected = nbformat.read(expected_spec.path, as_version=4)
    assert normalized(actual) == normalized(expected)
    assert [cell.id for cell in actual.cells] == [
        "title-id",
        actual.cells[1].id,
        "manual",
        "retired",
    ]
    stamp = spec.path.stat().st_mtime_ns
    native.render_notebook(
        spec,
        graphical_signature_code=lambda doc: "",
        call_graph_code=lambda example: "",
        imports=original._IMPORTS,
    )
    assert spec.path.stat().st_mtime_ns == stamp
    for path in (spec.path, expected_spec.path):
        notebook = nbformat.read(path, as_version=4)
        notebook.cells.append(
            nbformat.v4.new_code_cell(
                "duplicate", metadata={"cq.autogen": "title_cell"}
            )
        )
        nbformat.write(notebook, path)
    before = spec.path.read_bytes()
    with pytest.raises(ValueError):
        native.render_notebook(
            spec,
            graphical_signature_code=lambda doc: "",
            call_graph_code=lambda example: "",
        )
    assert spec.path.read_bytes() == before
    with pytest.raises(ValueError):
        original.render_notebook(expected_spec)
    fresh, _ = native._init_notebook("native", overwrite=True, directory=str(tmp_path))
    assert fresh.cells == [] and not spec.path.exists()
    module.__doc__ = None
    assert native._get_title_lines("Title", module) == ["# Title"]


def test_runnable_native_cells_use_append_and_existing_drawer(tmp_path: Path) -> None:
    example: ComponentExample[object] = ComponentExample(
        _swap_example, "swap_example", WordSwap
    )
    documentation: ComponentDocumentation[object] = ComponentDocumentation(
        component_type=WordSwap,
        examples=[example],
        import_line="from examples.algorithms.register_component_tutorial import WordSwap",
    )
    module = types.ModuleType(
        "native.word_swap", "Execute the generated native example."
    )
    module.__file__ = str(tmp_path / "word_swap.py")
    spec = native.NotebookSpec(
        title="Word swap", module=module, component_specs=[documentation]
    )
    diagram = """from flagquantum.drawer import draw
circuit = fq.Circuit(6)
circuit.append(swap_example.make_on((0, 1, 2), (3, 4, 5)))
assert len(circuit.to_ir().instructions) == 9
signature = draw(circuit, format='text')
assert isinstance(signature, str)"""
    graph = """from flagquantum.compiler._component_trace.program_call_graph import program_component_call_graph
from flagquantum.drawer.call_graph import format_call_graph_debug_text
model = program_component_call_graph(swap_example.decompose(), entry='Swap')
assert model.leaf_counts['CNOT'] == 9
call_graph = format_call_graph_debug_text(model.root, model.direct_calls)
assert 'CNOT' in call_graph"""
    native.render_notebook(
        spec,
        graphical_signature_code=lambda doc: diagram,
        call_graph_code=lambda example: graph,
    )
    notebook = nbformat.read(spec.path, as_version=4)
    namespace: dict[str, Any] = {}
    for cell in notebook.cells:
        if cell.cell_type == "code":
            exec(compile(cell.source, str(spec.path), "exec"), namespace)
    assert (
        "WordSwap" in namespace and namespace["signature"] and namespace["call_graph"]
    )
    assert "qualtran" not in "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    operation = namespace["swap_example"]
    assert isinstance(operation, WordSwap)


def dirty_notebook() -> NotebookNode:
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "print('hello')",
                execution_count=7,
                outputs=[
                    nbformat.v4.new_output("stream", name="stdout", text="hello\n")
                ],
                metadata={"cq.autogen": "preserve", "transient": "remove"},
            )
        ],
        metadata={"transient": "remove"},
    )

    assert isinstance(notebook, NotebookNode)
    return notebook


def test_source_dirty_diff_defect_and_native_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = load_source("clean_notebooks")
    path = tmp_path / "dirty.ipynb"
    nbformat.write(dirty_notebook(), path)
    before = path.read_bytes()
    temporary: list[Path] = []

    def local_temporary(*args: Any, **kwargs: Any) -> Any:
        kwargs["dir"] = tmp_path
        output = NamedTemporaryFile(*args, **kwargs)
        temporary.append(Path(output.name))
        return output

    monkeypatch.setattr(original, "NamedTemporaryFile", local_temporary)
    try:
        with pytest.raises(subprocess.CalledProcessError) as error:
            original.clean_notebook(path, do_clean=True)
        assert error.value.returncode == 1 and path.read_bytes() == before
        expected = nbformat.read(temporary[0], as_version=4)
        assert cleaning.clean_notebook(path, do_clean=False)
        assert path.read_bytes() == before
        assert cleaning.clean_notebook(path, do_clean=True)
        actual = nbformat.read(path, as_version=4)
        assert normalized(actual) == normalized(expected)
        assert actual.cells[0].metadata == {"cq.autogen": "preserve"}
        assert actual.cells[0].outputs == [] and actual.cells[0].execution_count is None
        assert not cleaning.clean_notebook(path)
        assert not original.clean_notebook(path)
    finally:
        for item in temporary:
            item.unlink(missing_ok=True)
    assert set(tmp_path.iterdir()) == {path}


def test_tracked_discovery_batch_and_temp_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def git(*args: str) -> None:
        subprocess.run(
            ["git", *args], cwd=tmp_path, check=True, capture_output=True, timeout=15
        )

    git("init")
    first, second, untracked = (
        tmp_path / name for name in ("a.ipynb", "b.ipynb", "untracked.ipynb")
    )
    for path in (first, second, untracked):
        nbformat.write(dirty_notebook(), path)
    git("add", first.name, second.name)
    original = load_source("clean_notebooks")
    assert (
        cleaning.get_nb_rel_paths(tmp_path)
        == original.get_nb_rel_paths(tmp_path)
        == [Path("a.ipynb"), Path("b.ipynb")]
    )
    assert cleaning.clean_notebooks(tmp_path, do_clean=False) == 2
    assert cleaning.clean_notebooks(tmp_path) == 2
    assert cleaning.clean_notebooks(tmp_path) == 0
    assert nbformat.read(untracked, as_version=4).cells[0].outputs
    before = {path for path in tmp_path.iterdir()}

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("write failure")

    monkeypatch.setattr(nbformat, "write", fail)
    with pytest.raises(RuntimeError, match="write failure"):
        cleaning.clean_notebook(first)
    assert set(tmp_path.iterdir()) == before


def test_complete_frozen_documentation_cell_corpus(original: Any) -> None:
    finder = load_source("bloq_finder")
    specifications = finder.get_bloqdocspecs(REFERENCE / "qualtran" / "bloqs")
    assert len(specifications) == 234
    for specification in specifications:
        try:
            expected = source_cells(original, specification)
        except (ValueError, OSError, TypeError) as error:
            with pytest.raises(type(error)):
                native_cells(original, specification)
        else:
            actual = native_cells(original, specification)
            assert normalized(nbformat.v4.new_notebook(cells=actual)) == normalized(
                nbformat.v4.new_notebook(cells=expected)
            ), specification.bloq_cls.__name__


def test_diff_failure_preserves_notebook_and_cleans_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "dirty.ipynb"
    nbformat.write(dirty_notebook(), path)
    before = path.read_bytes()

    def failed_diff(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            args[0], returncode=2, stdout=b"", stderr=b"failure"
        )

    monkeypatch.setattr(subprocess, "run", failed_diff)
    with pytest.raises(subprocess.CalledProcessError):
        cleaning.clean_notebook(path)
    assert path.read_bytes() == before and set(tmp_path.iterdir()) == {path}
