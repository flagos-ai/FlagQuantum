"""Complete frozen template generator, assets and native isolation comparisons."""

import csv
import hashlib
import importlib
import io
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from html import unescape
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest

from tests.unit.test_development_tools import REFERENCE, load_source
from tools.ftqc_source_inputs import FTQC_SOURCE_INVENTORY

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@contextmanager
def source_templates(source: Any) -> Iterator[None]:
    from tensorflow_docs.api_generator import parser

    previous = parser.TABLE_TEMPLATE, parser.ITEMS_TEMPLATE
    try:
        parser.TABLE_TEMPLATE, parser.ITEMS_TEMPLATE = (
            source.MY_TABLE_TEMPLATE,
            source.MyItemTemplate,
        )
        yield
    finally:
        parser.TABLE_TEMPLATE, parser.ITEMS_TEMPLATE = previous


@pytest.fixture
def source(monkeypatch: pytest.MonkeyPatch) -> Any:
    from tensorflow_docs.api_generator import parser

    package = types.ModuleType("_development_reference")
    package.__path__ = [str(REFERENCE / "dev_tools/qualtran_dev_tools")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(
        sys.modules, package.__name__ + ".git_tools", load_source("git_tools")
    )
    previous = parser.TABLE_TEMPLATE, parser.ITEMS_TEMPLATE
    try:
        result = load_source("reference_docs")
        assert parser.TABLE_TEMPLATE == result.MY_TABLE_TEMPLATE
        assert parser.ITEMS_TEMPLATE is result.MyItemTemplate
    finally:
        parser.TABLE_TEMPLATE, parser.ITEMS_TEMPLATE = previous
    return result


@pytest.fixture
def package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[types.ModuleType]:
    root = tmp_path / "fixture_templates"
    root.mkdir()
    (root / "child.py").write_text('''"""A child module."""
class Parent:
    """A base class."""
    def inherited(self):
        """An inherited method."""
''')
    (root / "__init__.py").write_text('''"""A module overview.

Args:
    package: A package description.
"""
from typing import Union
from . import child
from .child import Parent
TYPE_CHECKING = False
Number = Union[int, float]
class Zebra(Parent):
    """A circuit with documented members.

    Attributes:
        width: A positive width.
    """
    width = 2
    def __init__(self, width: int = 2):
        """Create the circuit.

        Args:
            width: The requested width.
        Raises:
            ValueError: If invalid.
        """
        self.width = width
    def zebra(self, value: int) -> int:
        """Append one operation.

        Args:
            value: A value.
                A continuation.
        Returns:
            The same value.
        """
        return value
    def alpha(self):
        """Another method."""
    @property
    def size(self):
        """Return the register size."""
        return self.width
class Alpha:
    """A second class."""
def transform(value: int) -> int:
    """Transform a value.

    Args:
        value: A documented value.
    Returns:
        An output.
    """
    return value
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "fixture_templates", raising=False)
    monkeypatch.delitem(sys.modules, "fixture_templates.child", raising=False)
    importlib.invalidate_caches()
    try:
        yield importlib.import_module("fixture_templates")
    finally:
        sys.modules.pop("fixture_templates.child", None)
        sys.modules.pop("fixture_templates", None)


def test_complete_source_and_six_asset_blobs() -> None:
    remaining = {"Q-0086", "Q-0113", "Q-0114", "Q-0115", "Q-0116", "Q-0117", "Q-0118"}
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
            if row["path"].endswith(".jinja"):
                assert (
                    Path("tools/development/templates") / Path(row["path"]).name
                ).read_bytes() == data
            remaining.remove(row["id"])
    assert not remaining
    assert metadata.version("tensorflow-docs") == "2023.5.24.56664"
    assert metadata.version("jinja2") == "3.1.6"
    assert metadata.version("protobuf") == "6.33.2"


def test_all_filters_and_native_alias_ownership(source: Any) -> None:
    from tools.development import reference_templates as native
    from tools.development.reference_template_pages import filter_and_sort_members

    children = [("TYPE_CHECKING", False), ("ordinary", object())]
    for path in [(), ("fixture",), ("fixture", "section")]:
        assert native.filter_type_checking(
            path, object(), children
        ) == source.filter_type_checking(path, object(), children)
        assert native.filter_type_aliases_in_the_wrong_place(
            path, object(), children
        ) == source.filter_type_aliases_in_the_wrong_place(path, object(), children)
    for name, path in native._TYPE_ALIAS_LOCATIONS.items():
        module = importlib.import_module(".".join(path))
        assert hasattr(module, name)
        items = [(name, getattr(module, name)), ("ordinary", object())]
        assert (
            native.filter_type_aliases_in_the_wrong_place(path, module, items) == items
        )
        assert (
            native.filter_type_aliases_in_the_wrong_place(("elsewhere",), module, items)
            == items[1:]
        )
    parent = types.SimpleNamespace(zebra=1, alpha=2)
    members = [
        types.SimpleNamespace(short_name=name)
        for name in ["alpha", "foreign", "zebra", "alpha"]
    ]
    assert filter_and_sort_members(
        parent, iter(members)
    ) == source._filter_and_sort_members(parent, members)


def test_item_and_title_block_full_formatting_without_globals(source: Any) -> None:
    from tensorflow_docs.api_generator import parser

    from tools.development import reference_template_pages as native

    ordinary = parser.TABLE_TEMPLATE, parser.ITEMS_TEMPLATE
    for name in ["value", "a&b", ""]:
        for anchor in ["", '<a id="value"></a>']:
            for description in [
                "",
                "One line.",
                "First.\nSecond.",
                "\n  Indented.\n    More.\n",
            ]:
                args = dict(name=name, anchor=anchor, description=description)
                assert native.DescriptionListItem.format(
                    **args
                ) == source.MyItemTemplate.format(**args)
    for text in ["", "Freeform text intentionally omitted."]:
        for title in ["Args", "Returns", "", None]:
            for items in [
                [],
                [("value", "")],
                [("one", "First.\n  More."), ("two", "Second.")],
            ]:
                for anchors in [False, True]:
                    for template in [None, "", "<h2>{title}</h2>"]:
                        block = parser.TitleBlock(title=title, text=text, items=items)
                        adapted = native._local_description_blocks(block)
                        assert adapted is not block
                        with source_templates(source):
                            expected = block.table_view(template, anchors)
                        assert adapted.table_view(template, anchors) == expected
                        assert "table_view" not in vars(block)
    assert ordinary == (parser.TABLE_TEMPLATE, parser.ITEMS_TEMPLATE)


def generate(root: Path, module: types.ModuleType, builders: Any) -> None:
    from tensorflow_docs.api_generator.generate_lib import DocGenerator
    from tensorflow_docs.api_generator.public_api import local_definitions_filter

    from tools.development.reference_templates import filter_type_checking

    assert module.__file__ is not None
    generator = DocGenerator(
        root_title="FlagQuantum",
        py_modules=[(module.__name__, module)],
        base_dir=[Path(module.__file__).parent],
        code_url_prefix="https://example.org/source/fixture_templates",
        callbacks=[local_definitions_filter, filter_type_checking],
        page_builder_classes=builders,
    )
    generator.build(output_dir=root)


def test_actual_four_page_categories_and_complete_generated_bytes(
    source: Any, package: types.ModuleType, tmp_path: Path
) -> None:
    from tensorflow_docs.api_generator import parser

    from tools.development import reference_template_pages as native

    previous = parser.TABLE_TEMPLATE, parser.ITEMS_TEMPLATE
    left, right = tmp_path / "source", tmp_path / "native"
    with source_templates(source):
        generate(left, package, source._MY_PAGE_BUILDERS)
    generate(right, package, native.page_info_classes())
    expected = {
        path.relative_to(left): path.read_bytes() for path in left.rglob("*.md")
    }
    actual = {
        path.relative_to(right): path.read_bytes() for path in right.rglob("*.md")
    }
    assert actual == expected
    assert len(actual) >= 7
    for name in [
        "fixture_templates.md",
        "fixture_templates/Zebra.md",
        "fixture_templates/transform.md",
        "fixture_templates/Number.md",
    ]:
        assert Path(name) in actual
    cls = unescape(actual[Path("fixture_templates/Zebra.md")].decode())
    assert cls.index('id="zebra"') < cls.index('id="alpha"')
    assert 'id="inherited"' not in cls
    assert previous == (parser.TABLE_TEMPLATE, parser.ITEMS_TEMPLATE)


def test_full_suffix_heuristics(source: Any) -> None:
    from tools.development.reference_output import fixup_suffix

    for before in ["", " ", "\n", "\t"]:
        for target in [
            "page.md",
            "../page.ipynb",
            "https://example.org/page.md",
            "README.md",
            "file.md.ipynb",
            "path with spaces.md",
        ]:
            for after in ["", '"', '#anchor"', ")", " text", "more", "#"]:
                text = before + '<a href="' + target + after + ">text</a>"
                assert fixup_suffix(text) == source.fixup_suffix(text)
    for text in [
        "page.md",
        "[link](page.md)",
        '<a href="page.md">page</a>',
        " README.md",
        " page.md#anchor",
        ' file.ipynb"',
    ]:
        assert fixup_suffix(text) == source.fixup_suffix(text)


def test_all_postprocessing_and_toc_bytes(source: Any, tmp_path: Path) -> None:
    from tools.development import reference_output as native

    roots = [tmp_path / "source", tmp_path / "native"]
    for root, package in zip(roots, ["qualtran", "flagquantum"], strict=True):
        output = root / "docs/reference"
        for path, text in [
            (package + "/_toc.yaml", "metadata"),
            (package + "/_redirects.yaml", "metadata"),
            (package + "/_api_cache.json", "metadata"),
            (package + "/api_report.pb", "metadata"),
            (package + "/section/zebra.md", ' <a href="alpha.md#x">value</a>'),
            (package + "/section/alpha.md", "A page."),
            (package + "/all_symbols.md", "redundant"),
            ("all_symbols.md", "redundant"),
        ]:
            target = output / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
    source.apply_fixups(roots[0])
    native.apply_fixups(roots[1])
    source.generate_ref_toc(roots[0])
    native.generate_ref_toc(roots[1])
    expected = {
        str(path.relative_to(roots[0]))
        .replace("qualtran", "flagquantum"): path.read_text()
        .replace("qualtran", "flagquantum")
        for path in roots[0].rglob("*")
        if path.is_file()
    }
    actual = {
        str(path.relative_to(roots[1])): path.read_text()
        for path in roots[1].rglob("*")
        if path.is_file()
    }
    assert actual == expected
    stamp = (roots[1] / "docs/reference/autotoc.rst.inc").stat().st_mtime_ns
    native.generate_ref_toc(roots[1])
    assert (roots[1] / "docs/reference/autotoc.rst.inc").stat().st_mtime_ns == stamp
    left, right = io.StringIO(), io.StringIO()
    grouped = {
        Path("b/deep"): [Path("b/deep/z.md"), Path("b/deep/a.md")],
        Path("a"): [],
    }
    source.write_ref_toc(left, grouped, Path("."))
    native.write_ref_toc(right, grouped, Path("."))
    assert left.getvalue() == right.getvalue()


def test_native_full_entry_and_generation_configuration(
    package: types.ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.development import reference_templates as native

    calls: list[Any] = []

    class Generator:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)

        def build(self, output_dir: Path) -> None:
            calls.append(output_dir)

    with monkeypatch.context() as context:
        context.setattr(native, "DocGenerator", Generator)
        native.generate_ref_docs(tmp_path, package)
        assert calls[0]["root_title"] == "FlagQuantum"
        assert calls[0]["py_modules"] == [("fixture_templates", package)]
        assert calls[1] == tmp_path / "docs/reference"
        assert len(calls[0]["callbacks"]) == 3
    native.build_reference_docs(tmp_path / "workflow", package)
    output = tmp_path / "workflow/docs/reference"
    assert (output / "autotoc.rst.inc").is_file()
    assert (output / "fixture_templates/Zebra.md").is_file()
    assert not list(output.rglob("all_symbols.md"))
    assert not list((output / "fixture_templates").glob("*.yaml"))


def test_complete_frozen_workflow_with_identical_reference_policy(
    source: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tensorflow_docs.api_generator.generate_lib import DocGenerator

    from tools.development import reference_templates as native

    source_root, native_root = tmp_path / "frozen_source", tmp_path / "frozen_native"
    monkeypatch.setattr(source, "get_git_root", lambda: source_root)
    with source_templates(source):
        source.build_reference_docs()
    module = importlib.import_module("qualtran")

    def reference_generator(**kwargs: Any) -> Any:
        # Both providers receive the frozen package's policy for this comparison.
        kwargs["root_title"] = "Qualtran"
        kwargs["code_url_prefix"] = (
            "https://github.com/quantumlib/Qualtran/blob/main/qualtran"
        )
        return DocGenerator(**kwargs)

    monkeypatch.setattr(native, "DocGenerator", reference_generator)
    monkeypatch.setattr(native, "_TYPE_ALIAS_LOCATIONS", source._TYPE_ALIAS_LOCATIONS)
    native.build_reference_docs(native_root, module)
    left, right = source_root / "docs/reference", native_root / "docs/reference"
    expected = {
        path.relative_to(left): path.read_bytes()
        for path in left.rglob("*")
        if path.is_file()
    }
    actual = {
        path.relative_to(right): path.read_bytes()
        for path in right.rglob("*")
        if path.is_file()
    }
    assert actual == expected
    assert len(actual) >= 50


def test_native_block_isolation_concurrent_and_failed_build(source: Any) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from tensorflow_docs.api_generator import parser

    from tools.development import reference_template_pages as native

    block = parser.TitleBlock(title="Args", text="", items=[("value", "A value.")])
    before = block.table_view()
    globals_before = parser.TABLE_TEMPLATE, parser.ITEMS_TEMPLATE
    with source_templates(source):
        expected = block.table_view()

    def render(local: bool) -> str:
        value = native._local_description_blocks(block) if local else block
        text: str = value.table_view()
        return text

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(render, [True, False] * 8)) == [expected, before] * 8
    assert "table_view" not in vars(block)

    def fail(page: object) -> None:
        raise ValueError("injected rendering failure")

    page = types.SimpleNamespace(
        _doc=None, attr_block=block, DEFAULT_BUILDER_CLASS=fail
    )
    with pytest.raises(ValueError, match="injected rendering failure"):
        native._build_page(page)
    assert "table_view" not in vars(block)
    assert globals_before == (parser.TABLE_TEMPLATE, parser.ITEMS_TEMPLATE)


def test_default_root_and_pipeline_failure_order(
    source: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools.development import reference_templates as native

    for owner in [source, native]:
        calls: list[tuple[str, Path]] = []
        with monkeypatch.context() as context:
            context.setattr(owner, "get_git_root", lambda: tmp_path)
            context.setattr(
                owner,
                "generate_ref_docs",
                lambda root, *args, calls=calls: calls.append(("generate", root)),
            )
            context.setattr(
                owner,
                "apply_fixups",
                lambda root, *args, calls=calls: calls.append(("fixups", root)),
            )
            context.setattr(
                owner,
                "generate_ref_toc",
                lambda root, *args, calls=calls: calls.append(("toc", root)),
            )
            owner.build_reference_docs()
            assert calls == [
                ("generate", tmp_path),
                ("fixups", tmp_path),
                ("toc", tmp_path),
            ]
            calls.clear()

            def fail(
                root: Path, *args: object, events: list[tuple[str, Path]] = calls
            ) -> None:
                events.append(("generate", root))
                raise ValueError("injected generator failure")

            context.setattr(owner, "generate_ref_docs", fail)
            with pytest.raises(ValueError, match="injected generator failure"):
                owner.build_reference_docs()
            assert calls == [("generate", tmp_path)]
