"""Explicit native notebook registration and complete frozen catalog mapping."""

import ast
import csv
import hashlib
import importlib
import json
import sys
import types
from pathlib import Path

import pytest

from tests.unit.test_development_tools import REFERENCE
from tools.development import notebook_catalog as native
from tools.ftqc_source_inputs import FTQC_SOURCE_INVENTORY

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
MAPPING = ROOT / "tests/fixtures/notebook_catalog_source_mapping.json"


def test_native_registration_and_root_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(native, "get_git_root", lambda: tmp_path)
    implicit = native.notebook_sections()
    explicit = native.notebook_sections(tmp_path / "examples/notebooks")
    assert implicit == explicit
    assert [len(paths) for _, paths in explicit] == [
        15,
        13,
        14,
        7,
        5,
        3,
        18,
        10,
        5,
        6,
        14,
    ]
    paths = [path for _, items in explicit for path in items]
    assert len(paths) == len(set(paths)) == 110
    assert all(path.parent == tmp_path / "examples/notebooks" for path in paths)
    # Resolution leaves missing entries visible to the consumer's error report.
    assert all(not path.exists() for path in paths)
    assert native.notebook_sections(Path("relative"))[0][1][0] == Path(
        "relative/register_t_gate.ipynb"
    )


def test_native_assets_and_component_imports() -> None:
    records = json.loads(MAPPING.read_text())
    paths = [
        path
        for _, items in native.notebook_sections(ROOT / "examples/notebooks")
        for path in items
    ]
    assert [record["native_notebook"] for record in records] == [
        str(path.relative_to(ROOT)) for path in paths
    ]
    assert sum(len(record["documents"]) for record in records) == 237
    for path in paths:
        notebook = json.loads(path.read_text())
        assert notebook["nbformat"] == 4
        assert any(
            cell["cell_type"] == "markdown" and "".join(cell["source"]).startswith("# ")
            for cell in notebook["cells"]
        ), path
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert code_cells, path
        for cell in code_cells:
            tree = ast.parse("".join(cell["source"]), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    assert all(
                        alias.name.split(".")[0] not in {"qualtran", "kickmix"}
                        for alias in node.names
                    )
                elif isinstance(node, ast.ImportFrom) and node.module:
                    assert node.module.split(".")[0] not in {"qualtran", "kickmix"}


@pytest.mark.ftqc_reference
def test_complete_original_catalog_and_registered_notebook_parents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = types.ModuleType("qualtran_dev_tools")
    package.__path__ = [str(REFERENCE / "dev_tools/qualtran_dev_tools")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    repository = importlib.import_module("qualtran_dev_tools.git_tools")
    monkeypatch.setattr(repository, "get_git_root", lambda: REFERENCE)
    # Instantiate every unchanged original NotebookSpecV2, including all module,
    # directory and path_stem defaults and actual BloqDocSpec references.
    monkeypatch.delitem(sys.modules, "qualtran_dev_tools.notebook_specs", raising=False)
    source = importlib.import_module("qualtran_dev_tools.notebook_specs")
    rows: list[dict[str, object]] = []
    qualtran = importlib.import_module("qualtran")
    source_file = qualtran.__file__
    assert source_file is not None
    source_root = Path(source_file).parent
    for section, specs in source.NB_BY_SECTION:
        for spec in specs:
            rows.append(
                {
                    "section": section,
                    "title": spec.title,
                    "module": spec.module.__name__,
                    "notebook": "qualtran/" + str(spec.path.relative_to(source_root)),
                    "documents": [
                        item.bloq_cls.__module__ + "." + item.bloq_cls.__qualname__
                        for item in spec.bloq_specs
                    ],
                }
            )
    mapping = json.loads(MAPPING.read_text())
    assert len(rows) == len(mapping) == 110
    for original, record in zip(rows, mapping, strict=True):
        assert original == {
            key: value for key, value in record.items() if key != "native_notebook"
        }
    assert [section for section, _ in native.NOTEBOOKS_BY_SECTION] == [
        section for section, _ in source.NB_BY_SECTION
    ]
    with (FTQC_SOURCE_INVENTORY).open(encoding="utf-8-sig") as handle:
        inventory = {row["path"]: row for row in csv.DictReader(handle)}
    selected = inventory["dev_tools/qualtran_dev_tools/notebook_specs.py"]
    raw = (REFERENCE / selected["path"]).read_bytes()
    assert (
        hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        == selected["git_blob"]
    )
    assert selected["id"] == "Q-0080"
    for original in rows:
        path = str(original["notebook"])
        row = inventory[path]
        assert row["id"].startswith("Q-") and path.endswith(".ipynb")
        data = (REFERENCE / path).read_bytes()
        assert (
            hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            == row["git_blob"]
        )


@pytest.mark.ftqc_reference
def test_actual_native_imported_owners() -> None:
    imported = set()
    for _, paths in native.notebook_sections(ROOT / "examples/notebooks"):
        for path in paths:
            for cell in json.loads(path.read_text())["cells"]:
                if cell["cell_type"] != "code":
                    continue
                for node in ast.walk(ast.parse("".join(cell["source"]))):
                    if not isinstance(node, ast.ImportFrom) or not node.module:
                        continue
                    if node.module.startswith(("flagquantum.", "examples.algorithms.")):
                        module = importlib.import_module(node.module)
                        imported.add(node.module)
                        for alias in node.names:
                            assert alias.name == "*" or hasattr(module, alias.name), (
                                path,
                                node.module,
                                alias.name,
                            )
    assert len(imported) > 150
