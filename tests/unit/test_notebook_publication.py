"""Complete library publication order, source contents and native rendering."""

import csv
import hashlib
import importlib
import io
import json
import os
import runpy
import shutil
import subprocess
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import pytest

from tests.unit.test_development_tools import REFERENCE
from tools.development import notebook_publication as native
from tools.development.notebook_catalog import notebook_sections
from tools.ftqc_source_inputs import FTQC_SOURCE_INVENTORY

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def original(monkeypatch: pytest.MonkeyPatch) -> Any:
    package = types.ModuleType("qualtran_dev_tools")
    package.__path__ = [str(REFERENCE / "dev_tools/qualtran_dev_tools")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    repository = importlib.import_module("qualtran_dev_tools.git_tools")
    monkeypatch.setattr(repository, "get_git_root", lambda: REFERENCE)
    monkeypatch.delitem(sys.modules, "qualtran_dev_tools.notebook_specs", raising=False)
    return types.SimpleNamespace(
        **runpy.run_path(
            str(REFERENCE / "dev_tools/autogenerate-bloqs-notebooks-v2.py")
        )
    )


def test_frozen_source_and_concept_root_mapping(original: Any) -> None:
    records = json.loads(
        (ROOT / "tests/fixtures/notebook_publication_source_mapping.json").read_text()
    )
    assert [row["source_notebook"] for row in records] == [
        "qualtran/bloqs/" + name for name in original.CONCEPTS + original.ROOT_BLOQS
    ]
    assert [Path(row["native_notebook"]).name for row in records] == list(
        native.CONCEPTS + native.ROOT_COMPONENTS
    )
    with (FTQC_SOURCE_INVENTORY).open(encoding="utf-8-sig") as stream:
        inventory = {row["path"]: row for row in csv.DictReader(stream)}
    for path in [
        "dev_tools/autogenerate-bloqs-notebooks-v2.py",
        *[row["source_notebook"] for row in records],
    ]:
        data = (REFERENCE / path).read_bytes()
        assert (
            hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            == inventory[path]["git_blob"]
        )
    assert inventory["dev_tools/autogenerate-bloqs-notebooks-v2.py"]["id"] == "Q-0036"


def test_source_render_order_and_toc_equivalence(original: Any, tmp_path: Path) -> None:
    source_globals = original.render_notebooks.__globals__
    calls = []
    source_globals["render_notebook"] = lambda spec: calls.append(spec.path)
    original.render_notebooks()
    mapping = json.loads(
        (ROOT / "tests/fixtures/notebook_catalog_source_mapping.json").read_text()
    )
    assert len(calls) == len(mapping) == 110
    selected = notebook_sections(ROOT / "examples/notebooks")
    native_calls: list[Path] = []
    native.render_notebooks(selected, native_calls.append)
    assert [str(path.relative_to(ROOT)) for path in native_calls] == [
        row["native_notebook"] for row in mapping
    ]
    for caption in ("Empty", "Operators", "Two Words"):
        for depth in (0, 1, 2, 4):
            assert native.toc_section_lines(
                caption, ["one.ipynb", "two.ipynb"], depth
            ) == original._get_toc_section_lines(
                caption, ["one.ipynb", "two.ipynb"], depth
            )
            assert native.toc_section_lines(
                caption, [], depth
            ) == original._get_toc_section_lines(caption, [], depth)
    source_globals["GIT_ROOT"] = tmp_path
    (tmp_path / "docs/bloqs").mkdir(parents=True)
    original.write_toc()
    expected = (tmp_path / "docs/bloqs/index.rst").read_text()
    committed_index = REFERENCE / "docs/bloqs/index.rst"
    raw = committed_index.read_bytes()
    assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == (
        "db2da97b72b546f02a246d2bb3ef1df39e43d175"
    )
    assert expected == committed_index.read_text()
    original_rows = list(original._all_nbspecs())
    source_bloqs = source_globals["SOURCE_DIR"] / "bloqs"
    replacements = [
        (str(spec.path.relative_to(source_bloqs)), Path(row["native_notebook"]).name)
        for spec, row in zip(original_rows, mapping, strict=True)
    ]
    concepts = json.loads(
        (ROOT / "tests/fixtures/notebook_publication_source_mapping.json").read_text()
    )
    replacements.extend(
        (
            row["source_notebook"].removeprefix("qualtran/bloqs/"),
            Path(row["native_notebook"]).name,
        )
        for row in concepts
    )
    # Normalize only the explicitly audited native destinations and branding.
    lines = expected.split("\n")
    paths = dict(replacements)
    lines = [
        (
            "    " + paths[line.strip()]
            if line.startswith("    ") and line.strip() in paths
            else line
        )
        for line in lines
    ]
    expected = (
        "\n".join(lines)
        .replace(".. _bloqs_library:", ".. _component_library:")
        .replace("Bloqs Library\n=============", "Component Library\n=================")
        .replace(
            "``qualtran.bloqs`` contains implementations of quantum operations and subroutines.",
            "FlagQuantum implements quantum operations and subroutines in native domains.",
        )
        .replace(
            ".. Note: this file is autogenerated. See dev_tools/autogenerate-bloqs-notebooks-v2.py.",
            ".. Note: this file is autogenerated by tools.development.notebook_publication.",
        )
        .replace(":caption: Root Bloqs:", ":caption: Root Components:")
    )
    destination = tmp_path / "index.rst"
    native.write_toc(selected, ROOT / "examples/notebooks", destination)
    assert destination.read_text() == expected


def test_actual_full_native_publication_and_repeat_timestamps(tmp_path: Path) -> None:
    source_root = ROOT / "examples/notebooks"
    destination = tmp_path / "published"
    selected = notebook_sections(source_root)
    with redirect_stdout(io.StringIO()) as output:
        native.publish_notebooks(source_root, destination)
    expected = {path for _, group in selected for path in group}
    expected.update(
        source_root / name for name in native.CONCEPTS + native.ROOT_COMPONENTS
    )
    emitted = set(destination.rglob("*.ipynb"))
    assert len(expected) == len(emitted) == 120
    for path in expected:
        assert (
            destination / path.relative_to(source_root)
        ).read_bytes() == path.read_bytes()
    assert (destination / "index.rst").is_file()
    timestamps = {path: path.stat().st_mtime_ns for path in destination.iterdir()}
    with redirect_stdout(io.StringIO()):
        native.publish_notebooks(source_root, destination)
    assert {
        path: path.stat().st_mtime_ns for path in destination.iterdir()
    } == timestamps
    unlisted = native.report_unlisted_notebooks(
        selected, source_root.rglob("*.ipynb"), source_root
    )
    assert unlisted
    assert all(
        str(path.relative_to(source_root)) in output.getvalue() for path in unlisted
    )
    assert not set(unlisted) & expected


def test_renderer_failure_and_missing_path_preserve_original_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    first, second = source / "first.ipynb", source / "second.ipynb"
    first.write_text("first")
    selected = (("Entries", (first, second)),)
    calls: list[Path] = []

    def fail(path: Path) -> None:
        calls.append(path)
        raise ValueError("renderer failure")

    destination = tmp_path / "output"
    with pytest.raises(ValueError, match="renderer failure"):
        native.publish_notebooks(source, destination, sections=selected, render=fail)
    assert calls == [first] and not destination.exists()
    monkeypatch.setattr(native, "CONCEPTS", ())
    monkeypatch.setattr(native, "ROOT_COMPONENTS", ())
    with pytest.raises(FileNotFoundError):
        native.publish_notebooks(source, destination, sections=selected)
    assert (destination / first.name).read_text() == "first"
    assert not (destination / "index.rst").exists()
    with pytest.raises(ValueError, match="must differ"):
        native.publish_notebooks(source, source, sections=selected)


def test_existing_notebook_renderer_and_original_unlisted_report(
    original: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools.development.notebook_autogen import NotebookSpec, render_notebook

    source = tmp_path / "source"
    source.mkdir()
    notebook = importlib.import_module("nbformat")
    path = source / "operator.ipynb"
    notebook.write(
        notebook.v4.new_notebook(
            cells=[notebook.v4.new_markdown_cell("Authored narrative")]
        ),
        path,
    )
    module = types.ModuleType("operator_example", "Native module documentation")
    module.__file__ = str(source / "operator_example.py")
    spec = NotebookSpec(
        title="Operator example",
        module=module,
        component_specs=[],
        directory=str(source),
        path_stem="operator",
    )

    def render(path: Path) -> None:
        assert path == spec.path
        render_notebook(
            spec,
            graphical_signature_code=lambda record: "",
            call_graph_code=lambda example: "",
            imports="import flagquantum as fq",
        )

    monkeypatch.setattr(native, "CONCEPTS", ())
    monkeypatch.setattr(native, "ROOT_COMPONENTS", ())
    destination = tmp_path / "output"
    native.publish_notebooks(
        source, destination, sections=(("Example", (path,)),), render=render
    )
    actual = notebook.read(destination / path.name, as_version=4)
    assert actual.cells[0].source == "Authored narrative"
    assert any(cell.metadata.get("cq.autogen") == "title_cell" for cell in actual.cells)
    assert any(cell.source == "import flagquantum as fq" for cell in actual.cells)
    # The original reports unregistered metadata identities. Native reports the
    # corresponding authored notebook assets, with duplicate discoveries folded.
    component = types.SimpleNamespace(__name__="Unlisted")
    record = type("Document", (), {"bloq_cls": component})()
    original.check_all_bloqs_included.__globals__["get_bloqdocspecs"] = lambda: [
        record,
        record,
    ]
    with redirect_stdout(io.StringIO()) as captured:
        original.check_all_bloqs_included()
    assert captured.getvalue().count("Unlisted") == 1
    other = source / "unlisted.ipynb"
    with redirect_stdout(io.StringIO()) as captured:
        assert native.report_unlisted_notebooks(
            (("Example", (path,)),), [path, other, other], source
        ) == (other,)
    assert captured.getvalue().count("unlisted.ipynb") == 1


def test_actual_publication_module_command(tmp_path: Path) -> None:
    source = tmp_path / "examples/notebooks"
    source.mkdir(parents=True)
    notebook = '{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}\n'
    required = (
        {path.name for _, paths in notebook_sections(source) for path in paths}
        | set(native.CONCEPTS)
        | set(native.ROOT_COMPONENTS)
    )
    for name in required:
        (source / name).write_text(notebook)
    subprocess.run(
        ["git", "init"], cwd=tmp_path, capture_output=True, check=True, timeout=15
    )
    result = subprocess.run(
        [sys.executable, "-m", "tools.development.notebook_publication"],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(list((tmp_path / "docs/components").glob("*.ipynb"))) == 120
    assert (tmp_path / "docs/components/index.rst").exists()


def test_complete_native_library_site_from_authored_notebooks(tmp_path: Path) -> None:
    pytest.importorskip("sphinx")
    pytest.importorskip("myst_nb")
    pytest.importorskip("pydata_sphinx_theme")
    source = tmp_path / "docs"
    source.mkdir()
    for name in ("_static", "_templates"):
        shutil.copytree(ROOT / "docs" / name, source / name)
    shutil.copyfile(ROOT / "docs/conf.py", source / "conf.py")
    (tmp_path / "assets").mkdir()
    shutil.copyfile(
        ROOT / "assets/logo_flagquantum.png",
        tmp_path / "assets/logo_flagquantum.png",
    )
    native.publish_notebooks(ROOT / "examples/notebooks", source / "components")
    (source / "index.rst").write_text(
        "Component documentation\n=======================\n\n"
        ".. toctree::\n\n   components/index\n"
    )
    output = tmp_path / "html"
    result = subprocess.run(
        [sys.executable, "-m", "sphinx", "-b", "html", str(source), str(output)],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ERROR:" not in result.stderr
    published = tuple((source / "components").glob("*.ipynb"))
    assert len(published) == 120
    for path in published:
        assert (output / "components" / path.with_suffix(".html").name).is_file()
    index = (output / "components/index.html").read_text()
    assert "Component Library" in index
    assert "register_addition.html" in index
    assert "binary_field_inverse.html" in index
