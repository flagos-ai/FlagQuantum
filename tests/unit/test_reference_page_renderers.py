"""Complete source class/module/TOC renderer dependency closure."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

import pytest

from tests.unit.test_development_tools import REFERENCE, load_source
from tools.ftqc_source_inputs import FTQC_SOURCE_INVENTORY

if TYPE_CHECKING:
    import griffe

    from tools.development.reference_pages import (
        Page,
    )


pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture
def original(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    root = "_page_render_reference"
    folder = REFERENCE / "dev_tools/qualtran_dev_tools"
    for suffix, path in [
        ("", folder),
        (".make_reference_docs", folder / "make_reference_docs"),
        (
            ".make_reference_docs._components",
            folder / "make_reference_docs/_components",
        ),
        (".make_reference_docs._pages", folder / "make_reference_docs/_pages"),
    ]:
        package = types.ModuleType(root + suffix)
        package.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, package.__name__, package)
    for name in ["parse_docstrings", "write_if_different"]:
        monkeypatch.setitem(sys.modules, root + "." + name, load_source(name))
    result: dict[str, Any] = {}
    for name in [
        "_page",
        "_linking_writer",
        "_render_context",
        "_components.aliases",
        "_components.render_signature",
        "_components.render_docstring",
        "_pages.render_major_class",
        "_pages.render_module",
        "_pages.render_toc",
    ]:
        spec = importlib.util.spec_from_file_location(
            root + ".make_reference_docs." + name,
            folder / "make_reference_docs" / (name.replace(".", "/") + ".py"),
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        result[name] = module
    return result


def test_renderer_blobs() -> None:
    remaining = {"Q-0074", "Q-0075", "Q-0076"}
    with (FTQC_SOURCE_INVENTORY).open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row["id"] not in remaining:
                continue
            data = (REFERENCE / row["path"]).read_bytes()
            assert (
                hashlib.sha1(
                    b"blob " + str(len(data)).encode() + b"\0" + data
                ).hexdigest()
                == row["git_blob"]
            )
            remaining.remove(row["id"])
    assert not remaining


def circuit_class() -> griffe.Class:
    import griffe

    cls = griffe.Class(
        "Circuit",
        docstring=griffe.Docstring("A circuit.\n\nPreserves full complex amplitudes."),
    )
    for name in ["__init__", "__repr__", "_private", "append"]:
        cls.set_member(
            name,
            griffe.Function(
                name,
                parameters=griffe.Parameters(
                    griffe.Parameter("self"),
                    griffe.Parameter(
                        "op", kind=griffe.ParameterKind.positional_or_keyword
                    ),
                ),
                returns="Circuit",
                docstring=griffe.Docstring("Append one operation."),
            ),
        )
    cls.set_member(
        "width",
        griffe.Attribute(
            "width", annotation="int", docstring=griffe.Docstring("Number of qubits.")
        ),
    )
    cls.set_member(
        "alias_append", griffe.Alias("alias_append", target=cls.members["append"])
    )
    return cls


def context(original: dict[str, Any] | None, page: Any) -> Any:
    from tools.development.reference_context import RenderContext

    aliases = {"Circuit": "native.Circuit", "native.Circuit": "native.Circuit"}
    if original is None:
        return RenderContext(
            Path("reference"),
            ["native"],
            aliases,
            {"native.Circuit": (page, None)},
            aliases,
        )
    return original["_render_context"].RenderContext(
        Path("reference"),
        ["native"],
        aliases,
        {"native.Circuit": (page, None)},
        aliases,
    )


def test_full_class_page_and_alias_members(
    original: dict[str, Any], tmp_path: Path
) -> None:
    from tools.development import reference_class_pages
    from tools.development.reference_pages import MajorClassPage

    cls = circuit_class()
    source_page = original["_page"].MajorClassPage(
        obj=cls, section="native", pref_path="native.Circuit"
    )
    page = MajorClassPage(obj=cls, section="native", pref_path="native.Circuit")
    source_root, native_root = tmp_path / "source", tmp_path / "native"
    original["_pages.render_major_class"].render_major_class(
        source_root, source_page, context(original, source_page)
    )
    reference_class_pages.render_major_class(native_root, page, context(None, page))
    expected, actual = source_page.out_path(source_root), page.out_path(native_root)
    assert actual.read_bytes() == expected.read_bytes()
    text = actual.read_text()
    assert "circuit.append(op)" in text and "circuit.alias_append(op)" in text
    assert (
        "## `width`" not in text
        and "## `_private`" not in text
        and "## `__init__`" not in text
    )
    stamp = actual.stat().st_mtime_ns
    reference_class_pages.render_major_class(native_root, page, context(None, page))
    assert actual.stat().st_mtime_ns == stamp
    for source_member in cls.members.values():
        left, right = io.StringIO(), io.StringIO()
        original["_pages.render_major_class"].write_major_class_member(
            left, cls, source_member
        )
        reference_class_pages.write_major_class_member(right, cls, source_member)
        assert right.getvalue() == left.getvalue()


@pytest.mark.parametrize(
    "categories",
    [(), ("MINOR",), ("MODULE",), ("MAJOR",), ("MODULE", "MAJOR", "MINOR")],
)
def test_full_module_page_categories(
    original: dict[str, Any], tmp_path: Path, categories: tuple[str, ...]
) -> None:
    import griffe

    from tools.development import reference_module_pages
    from tools.development.reference_pages import (
        MemberType,
        ModulePage,
        ModulePageMember,
    )

    module = griffe.Module(
        "native", docstring=griffe.Docstring("A native module.\n\nDetailed overview.")
    )
    function = griffe.Function(
        "execute",
        parameters=griffe.Parameters(
            griffe.Parameter("value", kind=griffe.ParameterKind.positional_or_keyword)
        ),
        docstring=griffe.Docstring("Execute a value."),
    )
    cls = circuit_class()
    objects = {
        "MODULE": griffe.Module(
            "nested", docstring=griffe.Docstring("A nested module.")
        ),
        "MAJOR": cls,
        "MINOR": function,
    }
    source_page = original["_page"].ModulePage(
        obj=module, section="native", pref_path="native"
    )
    page = ModulePage(obj=module, section="native", pref_path="native")
    for category in categories:
        obj = objects[category]
        source_page.members.append(
            original["_page"].ModulePageMember(
                obj,
                "native." + obj.name,
                getattr(original["_page"].MemberType, category),
            )
        )
        page.members.append(
            ModulePageMember(obj, "native." + obj.name, MemberType[category])
        )
    source_root, native_root = tmp_path / "source", tmp_path / "native"
    original["_pages.render_module"].render_module(
        source_root, source_page, context(original, source_page)
    )
    reference_module_pages.render_module(native_root, page, context(None, page))
    assert (
        page.out_path(native_root).read_bytes()
        == source_page.out_path(source_root).read_bytes()
    )


def test_minor_class_documented_and_inherited_members(original: dict[str, Any]) -> None:
    import griffe

    from tools.development import reference_module_pages
    from tools.development.reference_pages import MajorClassPage

    base = griffe.Class("Base")
    base.set_member(
        "inherited",
        griffe.Function("inherited", docstring=griffe.Docstring("Inherited method.")),
    )
    cls = circuit_class()
    cls.bases = [griffe.ExprName("Base", parent=cls)]
    collection = griffe.ModulesCollection()
    module = griffe.Module("native", modules_collection=collection)
    collection.set_member("native", module)
    module.set_member("Base", base)
    module.set_member("Circuit", cls)
    assert cls.inherited_members
    expected, actual = io.StringIO(), io.StringIO()
    source_context = context(
        original, original["_page"].MajorClassPage(pref_path="native.Circuit")
    )
    native_context = context(None, MajorClassPage(pref_path="native.Circuit"))
    original["_pages.render_module"].write_module_other_member(
        source_context.get_linking_writer(expected), cls, "native.Circuit", {}
    )
    reference_module_pages.write_module_other_member(
        native_context.get_linking_writer(actual), cls, "native.Circuit", {}
    )
    assert actual.getvalue() == expected.getvalue()
    assert (
        "**All Members:**" in actual.getvalue() and "#### Members" in actual.getvalue()
    )


def test_toc_full_selection_order_and_errors(
    original: dict[str, Any], tmp_path: Path
) -> None:
    from tools.development import reference_toc
    from tools.development.reference_pages import MajorClassPage, ModulePage

    cls = circuit_class()
    sources = [
        original["_page"].ModulePage(pref_path="native.empty", section="native"),
        original["_page"].MajorClassPage(
            obj=cls, pref_path="native.Circuit", section="native"
        ),
    ]
    pages: list[Page] = [
        ModulePage(pref_path="native.empty", section="native"),
        MajorClassPage(obj=cls, pref_path="native.Circuit", section="native"),
    ]
    expected, actual = io.StringIO(), io.StringIO()

    def select(page: Page, section: str) -> bool:
        return page.section == section

    def order(page: Page) -> list[str]:
        assert page.pref_path is not None
        return page.pref_path.split(".")

    original["_pages.render_toc"].write_toc(
        expected, sources, select, order, context(original, sources[1])
    )
    reference_toc.write_toc(actual, pages, select, order, context(None, pages[1]))
    assert actual.getvalue() == expected.getvalue()
    assert (
        "/reference/native/Circuit.md" in actual.getvalue()
        and "empty.md" not in actual.getvalue()
    )
    for owner, owner_pages, owner_context in [
        (original["_pages.render_toc"], sources, context(original, sources[1])),
        (reference_toc, pages, context(None, pages[1])),
    ]:
        folder = tmp_path / ("source" if owner is not reference_toc else "native")
        folder.mkdir()
        owner.render_toc(folder, owner_pages, select, order, owner_context)
        assert (folder / "autotoc.rst.inc").read_text() == actual.getvalue()
        owner_context.sections = [""]
        with pytest.raises(ValueError):
            owner.write_toc(io.StringIO(), owner_pages, select, order, owner_context)


def test_toc_failure_closes_native_output(
    original: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools.development import reference_toc
    from tools.development.reference_pages import MajorClassPage

    opened: list[TextIO] = []
    path_open = Path.open

    def capture(path: Path, *args: Any, **kwargs: Any) -> Any:
        stream = path_open(path, *args, **kwargs)
        opened.append(stream)
        return stream

    monkeypatch.setattr(Path, "open", capture)
    source = original["_pages.render_toc"]
    source_context = context(
        original, original["_page"].MajorClassPage(pref_path="native.Circuit")
    )
    native_context = context(None, MajorClassPage(pref_path="native.Circuit"))
    source_context.sections = native_context.sections = [""]
    for owner, value in [(source, source_context), (reference_toc, native_context)]:
        with pytest.raises(ValueError):
            owner.render_toc(
                tmp_path, [], lambda page, section: True, lambda page: [], value
            )
    assert not opened[0].closed and opened[1].closed
    opened[0].close()
