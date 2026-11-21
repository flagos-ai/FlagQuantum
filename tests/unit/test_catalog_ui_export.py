"""Combined catalog publication, first-file behavior and frozen navigation gap."""

import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from examples.algorithms import catalog_ui_export as native
from tests.unit.test_chemistry_ui_export import originals as originals
from tests.unit.test_component_ui_export import source_metadata

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_fresh_all_groups_and_first_file_outputs(tmp_path: Path) -> None:
    expected_files: dict[Path, bytes] = {}
    expected_navigation: dict[str, object] = {}
    for index, export in enumerate(native.EXPORTERS):
        folder = tmp_path / str(index)
        export(folder)
        expected_navigation.update(json.loads((folder / "navigation.json").read_text()))
        for path in folder.rglob("*"):
            if path.is_file() and path.name != "navigation.json":
                expected_files.setdefault(path.relative_to(folder), path.read_bytes())
    combined = tmp_path / "combined"
    native.export_example(combined)
    assert json.loads((combined / "navigation.json").read_text()) == expected_navigation
    assert {p.relative_to(combined) for p in combined.rglob("*") if p.is_file()} == {
        *expected_files,
        Path("navigation.json"),
    }
    assert len(list(combined.rglob("*.json"))) == 354
    assert len(list(combined.rglob("docs.txt"))) == 233
    for path, expected in expected_files.items():
        assert (combined / path).read_bytes() == expected
        if path.suffix == ".json":
            data = json.loads(expected)
            assert data["msd"] is not None
            assert data["callees"] == []


def test_complete_navigation_and_original_composite_folder_gap(
    tmp_path: Path, originals: tuple[Any, dict[str, Any]]
) -> None:
    native.export_example(tmp_path)
    navigation = json.loads((tmp_path / "navigation.json").read_text())
    source = importlib.import_module("qualtran_dev_tools.notebook_specs")
    assert list(navigation) == [name for name, _ in source.NB_BY_SECTION]
    missing = []
    documents = factories = 0
    for section, specs in source.NB_BY_SECTION:
        assert list(navigation[section]) == [spec.title for spec in specs]
        for spec in specs:
            actual_docs = navigation[section][spec.title]
            assert [row["name"] for row in actual_docs] == [
                doc.bloq_cls.__name__ for doc in spec.bloq_specs
            ]
            for row, doc in zip(actual_docs, spec.bloq_specs, strict=True):
                documents += 1
                assert [e["name"] for e in row["examples"]] == [
                    e.name for e in doc.examples
                ]
                for example in row["examples"]:
                    factories += 1
                    if not (tmp_path / row["name"] / example["filename"]).is_file():
                        missing.append(
                            (row["name"], example["name"], example["filename"])
                        )
    assert sum(map(len, navigation.values())) == 110
    assert documents == 237 and factories == 359
    products = [
        doc.bloq_cls
        for _, specs in source.NB_BY_SECTION
        for spec in specs
        for doc in spec.bloq_specs
        if doc.bloq_cls.__name__ == "Product"
    ]
    assert len(products) == 2 and products[0].__module__ != products[1].__module__
    metadata = source_metadata()
    subtraction = importlib.import_module("qualtran.bloqs.arithmetic.subtraction")
    composite = subtraction._sub_symp_decomposition.make()
    assert type(composite).__name__ == "CompositeBloq"
    filename = metadata["bloq_filename"](composite)
    assert missing == [("Subtract", "sub_symp_decomposition", filename)]
    assert (tmp_path / "CompositeBloq" / filename).is_file()


def test_rerun_preserves_nodes_and_refreshes_complete_navigation(
    tmp_path: Path,
) -> None:
    native.export_example(tmp_path)
    node = next(
        path for path in tmp_path.rglob("*.json") if path.name != "navigation.json"
    )
    node.write_text("retained node")
    docs = next(tmp_path.rglob("docs.txt"))
    docs.write_text("retained documentation")
    (tmp_path / "navigation.json").write_text("stale navigation")
    native.export_example(tmp_path)
    assert node.read_text() == "retained node"
    assert docs.read_text() == "retained documentation"
    assert len(json.loads((tmp_path / "navigation.json").read_text())) == 11


def test_section_ownership_collision_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exporter = native.EXPORTERS[0]
    monkeypatch.setattr(native, "EXPORTERS", (exporter, exporter))
    with pytest.raises(ValueError, match="single domain owner"):
        native.export_example(tmp_path)
