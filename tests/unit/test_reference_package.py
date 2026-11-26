"""Complete frozen static package walker and native documentation pipeline."""

from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import io
import re
import sys
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests.unit.test_development_tools import REFERENCE
from tests.unit.test_reference_page_renderers import original as original
from tools.ftqc_source_inputs import FTQC_SOURCE_INVENTORY

if TYPE_CHECKING:
    import griffe


pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@contextmanager
def source_decorator_tables() -> Iterator[None]:
    from griffe._internal.agents import visitor

    builtin = dict(visitor.builtin_decorators)
    standard = dict(visitor.stdlib_decorators)
    try:
        visitor.builtin_decorators.pop("property", None)
        visitor.stdlib_decorators.pop("functools.cached_property", None)
        yield
    finally:
        visitor.builtin_decorators.clear()
        visitor.builtin_decorators.update(builtin)
        visitor.stdlib_decorators.clear()
        visitor.stdlib_decorators.update(standard)


@pytest.fixture
def source(original: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> Any:
    from griffe._internal.agents import visitor

    name = "_page_render_reference.make_reference_docs._make"
    path = REFERENCE / "dev_tools/qualtran_dev_tools/make_reference_docs/_make.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    builtin, standard = dict(visitor.builtin_decorators), dict(
        visitor.stdlib_decorators
    )
    try:
        spec.loader.exec_module(module)
        assert "property" not in visitor.builtin_decorators
        assert "functools.cached_property" not in visitor.stdlib_decorators
    finally:
        visitor.builtin_decorators.clear()
        visitor.builtin_decorators.update(builtin)
        visitor.stdlib_decorators.clear()
        visitor.stdlib_decorators.update(standard)
    return module


def test_full_source_and_export_blobs() -> None:
    from tools.development import reference_docs

    remaining = {"Q-0065", "Q-0066", "Q-0071", "Q-0073"}
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
            tree = ast.parse(data)
            if row["id"] == "Q-0065":
                assert len(tree.body) == 1 and isinstance(tree.body[0], ast.ImportFrom)
                assert [alias.name for alias in tree.body[0].names] == [
                    "make_reference_docs"
                ]
            elif row["id"] in {"Q-0066", "Q-0073"}:
                assert tree.body == []
            remaining.remove(row["id"])
    assert not remaining
    assert callable(reference_docs.make_reference_docs)


@pytest.fixture
def package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "fixture_docs"
    root.mkdir()
    (root / "__init__.py").write_text('''"""A package overview.

A detailed paragraph.
"""
import attrs
from functools import cached_property
from . import section
from .section import Register

@attrs.frozen
class Circuit:
    """A typed circuit."""
    _width: int
    @property
    def width(self) -> int:
        """Current width."""
        return self._width
    @cached_property
    def total(self) -> int:
        """Total width."""
        return self._width
    def append(self, op: int) -> int:
        """Append one operation."""
        return op

class Plain:
    value: int
    def __init__(self, value: int):
        self.value = value

class Mutable:
    value: int

@attrs.define
class DecoratedMutable:
    value: int

def execute(value: int = 0) -> int:
    """Execute one value."""
    return value

_private = 0
''')
    (root / "section.py").write_text('''"""Registers and operations."""
class Register:
    """A register."""
    pass

def operation() -> int:
    """An operation."""
    return 0
''')
    for name in ["hidden_test", "generated_pb2", "_private"]:
        (root / (name + ".py")).write_text('"""Skipped module."""\nvalue = 1\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    return root


def configure_fixture(source: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from tools.development import reference_package as native

    for owner in [source, native]:
        monkeypatch.setattr(owner, "MAJOR_CLASSES", ["fixture_docs.Circuit"])
        monkeypatch.setattr(owner, "SKIP_MODULES", [])
        monkeypatch.setattr(owner, "MINIMAL_SKIP_CANONPATHS", [])
        monkeypatch.setattr(owner, "MINIMAL", True)
        monkeypatch.setattr(owner, "DEFINED_IN_CONTAINER_EXCEPTIONS", [])


def page_snapshot(pages: Any) -> list[object]:
    return [
        (
            type(page).__name__,
            page.pref_path,
            page.section,
            page.obj.canonical_path if page.obj is not None else None,
            [
                (member.pref_dotpath, member.mytype.value, member.obj.canonical_path)
                for member in getattr(page, "members", [])
            ],
        )
        for page in pages
    ]


def context_snapshot(context: Any) -> object:
    return (
        context.refdoc_relpath,
        context.sections,
        context.aliases_d,
        {
            name: (page.pref_path, anchor)
            for name, (page, anchor) in context.link_d.items()
        },
        context.linkable_to_prefpath,
    )


def test_complete_walk_configuration_and_local_parser_replacement(
    source: Any, package: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import griffe
    from griffe._internal.agents import visitor

    from tools.development import reference_package as native

    configure_fixture(source, monkeypatch)
    before = dict(visitor.builtin_decorators), dict(visitor.stdlib_decorators)
    with source_decorator_tables():
        expected_pages, expected_context = source.walk_and_configure(
            "fixture_docs",
            Path("reference"),
            ["fixture_docs"],
            ["fixture_docs.section"],
            {"Circuit": "fixture_docs.Circuit"},
        )
    pages, context = native.walk_and_configure(
        "fixture_docs",
        Path("reference"),
        ["fixture_docs"],
        ["fixture_docs.section"],
        {"Circuit": "fixture_docs.Circuit"},
    )
    assert page_snapshot(pages) == page_snapshot(expected_pages)
    assert context_snapshot(context) == context_snapshot(expected_context)
    assert context.sections == ["fixture_docs"]
    assert (visitor.builtin_decorators, visitor.stdlib_decorators) == before
    circuit = next(
        page.obj for page in pages if page.pref_path == "fixture_docs.Circuit"
    )
    assert circuit is not None
    assert circuit.members["width"].kind is griffe.Kind.FUNCTION
    assert circuit.members["total"].kind is griffe.Kind.FUNCTION
    assert isinstance(circuit, griffe.Class)
    assert [parameter.name for parameter in circuit.parameters] == ["width"]
    assert [str(parameter.annotation) for parameter in circuit.parameters] == ["int"]
    ordinary = griffe.GriffeLoader().load("fixture_docs")
    assert ordinary.members["Circuit"].members["width"].kind is griffe.Kind.ATTRIBUTE
    assert ordinary.members["Circuit"].members["total"].kind is griffe.Kind.ATTRIBUTE
    assert all(
        "hidden_test" not in str(item)
        and "generated_pb2" not in str(item)
        and "_private" not in str(item)
        for item in page_snapshot(pages)
    )


def test_alias_filters_preference_and_source_boundaries(
    source: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import griffe

    from tools.development import reference_package as native

    configure_fixture(source, monkeypatch)
    module = griffe.Module("visible", filepath=Path("visible/__init__.py"))
    internal = griffe.Module("_internal", parent=module)
    cls = griffe.Class("Entry", parent=internal)
    internal.set_member("Entry", cls)
    module.set_member("_internal", internal)
    module.set_member("Entry", griffe.Alias("Entry", target=cls))
    foreign = griffe.Module("foreign")
    foreign.set_member("Entry", griffe.Alias("Entry", target=cls))
    for owner in [source, native]:
        monkeypatch.setattr(owner, "DEFINED_IN_CONTAINER_EXCEPTIONS", ["foreign"])
        assert owner._get_all_aliases(cls) == {"visible.Entry", "foreign.Entry"}
        assert owner._get_preferred_dotpath({"a", "a.b", "a.b.c"}) == "a.b.c"
        with pytest.raises(IndexError):
            owner._get_preferred_dotpath(set())
        private = griffe.Class("_Private", parent=internal)
        with pytest.warns(UserWarning, match="No valid aliases"):
            assert owner._get_all_aliases(private) == set()
        with pytest.raises(AssertionError):
            owner.get_pages(griffe.Module("not_package", filepath=Path("module.py")))
        with pytest.raises(AssertionError):
            owner.get_pages(
                griffe.Module(
                    "nested", filepath=Path("nested/__init__.py"), parent=module
                )
            )


@pytest.mark.parametrize("minimal", [False, True])
def test_skip_policies_and_mutable_walker_state(
    source: Any, package: Path, monkeypatch: pytest.MonkeyPatch, minimal: bool
) -> None:
    import griffe

    from tools.development import reference_package as native

    configure_fixture(source, monkeypatch)
    root = griffe.GriffeLoader().load("fixture_docs")
    assert isinstance(root, griffe.Module)
    for owner in [source, native]:
        monkeypatch.setattr(owner, "SKIP_MODULES", ["fixture_docs.section"])
        monkeypatch.setattr(owner, "MINIMAL_SKIP_CANONPATHS", ["fixture_docs.Plain"])
        monkeypatch.setattr(owner, "MINIMAL", minimal)
    with warnings.catch_warnings(record=True) as expected_warnings:
        expected, expected_aliases, expected_links = source.get_pages(root)
    with warnings.catch_warnings(record=True) as actual_warnings:
        pages, aliases, links = native.get_pages(root)
    assert page_snapshot(pages) == page_snapshot(expected)
    assert aliases == expected_aliases
    assert {
        name: (page.pref_path, anchor) for name, (page, anchor) in links.items()
    } == {
        name: (page.pref_path, anchor)
        for name, (page, anchor) in expected_links.items()
    }
    assert [str(value.message) for value in actual_warnings] == [
        str(value.message) for value in expected_warnings
    ]
    assert ("fixture_docs.Plain" in aliases) != minimal
    assert "fixture_docs.section" not in aliases
    for owner in [source, native]:
        first, second = owner._PackageWalker(), owner._PackageWalker()
        first.seen.add("seen")
        assert not second.seen
        first._walk_table_of_contents(root)
        with pytest.raises(AssertionError):
            first._walk_table_of_contents(root)


def test_complete_generation_outputs_and_new_destination_fix(
    source: Any, package: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools.development import reference_docs

    configure_fixture(source, monkeypatch)
    source_root, native_root = tmp_path / "source", tmp_path / "native"
    (source_root / "docs/reference").mkdir(parents=True)
    with source_decorator_tables():
        source.make_reference_docs(source_root, rootmod_dotpath="fixture_docs")
    reference_docs.make_reference_docs(native_root, rootmod_dotpath="fixture_docs")
    expected = {
        path.relative_to(source_root / "docs/reference"): path.read_text()
        for path in (source_root / "docs/reference").rglob("*.md")
    }
    actual = {
        path.relative_to(native_root / "docs/reference"): path.read_text()
        for path in (native_root / "docs/reference").rglob("*.md")
    }
    assert actual == expected and actual
    source_toc = (source_root / "docs/reference/autotoc.rst.inc").read_text()
    native_toc = (native_root / "docs/reference/autotoc.rst.inc").read_text()
    assert [line for line in native_toc.splitlines() if line.startswith("   /")] == [
        line for line in source_toc.splitlines() if line.startswith("   /")
    ]
    assert ":caption: flagquantum.core" in native_toc
    assert ":caption: qualtran.resource_counting" in source_toc
    for owner, folder in [
        (source, tmp_path / "missing_source"),
        (reference_docs, tmp_path / "missing_native"),
    ]:
        if owner is source:
            with source_decorator_tables(), pytest.raises(FileNotFoundError):
                owner.make_reference_docs(folder, rootmod_dotpath="fixture_docs")
        else:
            owner.make_reference_docs(folder, rootmod_dotpath="fixture_docs")
            assert (folder / "docs/reference/autotoc.rst.inc").exists()


def test_complete_frozen_package_walk_and_render_outcomes(
    source: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools.development import reference_docs, reference_signatures
    from tools.development import reference_package as native

    for name in [
        "MAJOR_CLASSES",
        "SKIP_MODULES",
        "MINIMAL_SKIP_CANONPATHS",
        "MINIMAL",
        "DEFINED_IN_CONTAINER_EXCEPTIONS",
    ]:
        monkeypatch.setattr(native, name, getattr(source, name))
    monkeypatch.setattr(
        reference_docs,
        "TOP_SECTIONS",
        [
            "qualtran",
            "qualtran.resource_counting",
            "qualtran.simulation",
            "qualtran.drawing",
            "qualtran.symbolics",
            "interop",
        ],
    )
    monkeypatch.setattr(
        reference_docs,
        "GROUPED_SECTIONS",
        {
            "qualtran": ("qualtran.dtype", "qualtran.exception"),
            "interop": ("qualtran.qref_interop", "qualtran.cirq_interop"),
        },
    )
    monkeypatch.setattr(
        reference_docs,
        "ADDITIONAL_LINKS",
        {
            "Bloq": "qualtran.Bloq",
            "CompositeBloq": "qualtran.CompositeBloq",
            "Soquet": "qualtran.Soquet",
        },
    )
    monkeypatch.setattr(
        reference_signatures,
        "BESPOKE_OBJECT_NAMES_FOR_CLASSES",
        {"BloqBuilder": "bb", "CompositeBloq": "cbloq"},
    )
    arguments: tuple[str, Path, list[str], list[str], dict[str, str]] = (
        "qualtran",
        Path("reference"),
        ["qualtran", "qualtran.resource_counting"],
        [],
        {"Bloq": "qualtran.Bloq"},
    )
    with source_decorator_tables():
        expected, expected_context = source.walk_and_configure(*arguments)
    pages, context = native.walk_and_configure(*arguments)
    assert len(pages) == len(expected) == 41
    assert len(context.aliases_d) == 353 and len(context.link_d) == 271
    assert page_snapshot(pages) == page_snapshot(expected)
    assert context_snapshot(context) == context_snapshot(expected_context)
    errors: list[tuple[str, str] | None] = []
    outputs = []
    for name, owner in [("source", source), ("native", reference_docs)]:
        root = tmp_path / name
        (root / "docs/reference").mkdir(parents=True)
        try:
            if name == "source":
                with source_decorator_tables():
                    owner.make_reference_docs(root, rootmod_dotpath="qualtran")
            else:
                owner.make_reference_docs(root, rootmod_dotpath="qualtran")
        except (ValueError, AssertionError) as error:
            errors.append((type(error).__name__, str(error)))
        else:
            errors.append(None)
        outputs.append(
            {
                path.relative_to(root): path.read_text()
                for path in root.rglob("*")
                if path.is_file()
            }
        )
    assert errors == [None, None]
    assert outputs[0].keys() == outputs[1].keys()
    for path in outputs[0]:
        assert canonical_unordered_members(
            outputs[0][path]
        ) == canonical_unordered_members(outputs[1][path]), path
    assert len(outputs[0]) == 42
    print(
        f"Complete frozen render outcome {errors[0]}, {len(outputs[0])} matching output files after source unordered-member canonicalization"
    )


def test_local_extension_concurrent_with_ordinary_griffe(
    source: Any, package: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    import griffe

    from tools.development import reference_package as native

    configure_fixture(source, monkeypatch)

    def load(local: bool) -> griffe.Kind:
        if local:
            pages, _ = native.walk_and_configure(
                "fixture_docs", Path("reference"), [], [], {}
            )
            cls = next(
                page.obj for page in pages if page.pref_path == "fixture_docs.Circuit"
            )
            assert cls is not None
            return cls.members["width"].kind
        root = griffe.GriffeLoader().load("fixture_docs")
        return root.members["Circuit"].members["width"].kind

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(load, [True, False, True, False]))
    assert outcomes == [griffe.Kind.FUNCTION, griffe.Kind.ATTRIBUTE] * 2


def canonical_unordered_members(text: str) -> str:
    # The source hashes (name, Griffe object) pairs in a set. New parses give
    # those objects new identities, changing this region's entry ordering.
    # Keep every other byte unchanged and compare each named entry's content.
    def replace(match: re.Match[str]) -> str:
        body = match.group(1)
        entries = re.split(r"(?=^`[^`\n]+`\n: )", body, flags=re.M)
        assert not entries[0].strip()
        return (
            "#### Members\n"
            + "\n\n".join(sorted(entry.strip("\n") for entry in entries[1:]))
            + "\n\n"
        )

    return re.sub(
        r"^#### Members\n(.*?)(?=^\*\*All Members:\*\*|^### `|^\[`|\Z)",
        replace,
        text,
        flags=re.M | re.S,
    )


def test_original_module_alias_cycle_and_native_cleanup(
    source: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import griffe

    from tools.development import reference_package as native

    configure_fixture(source, monkeypatch)

    def root() -> griffe.Module:
        module = griffe.Module("cycle", filepath=Path("cycle/__init__.py"))
        module.set_member("again", griffe.Alias("again", target=module))
        return module

    with pytest.raises(RecursionError):
        source.get_pages(root())
    with pytest.warns(UserWarning, match="Orphan"):
        pages, aliases, links = native.get_pages(root())
    assert [page.pref_path for page in pages] == ["cycle"]
    assert aliases["cycle"] == "cycle" and links["cycle"][0] is pages[0]
    walker = native._PackageWalker()
    broken = griffe.Module("broken", filepath=Path("broken/__init__.py"))
    walker.aliases_d["broken"] = "collision"
    with pytest.raises(AssertionError):
        walker._walk_table_of_contents(broken)
    assert not walker._active


def test_actual_native_package_walk_and_inherited_setter_refusal(
    original: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import griffe

    from tools.development import reference_docs, reference_signatures
    from tools.development import reference_package as native

    pages, context = native.walk_and_configure(
        "flagquantum",
        Path("reference"),
        reference_docs.TOP_SECTIONS,
        [],
        reference_docs.ADDITIONAL_LINKS,
    )
    circuit = next(
        page.obj for page in pages if page.pref_path == "flagquantum.circuit.Circuit"
    )
    assert isinstance(circuit, griffe.Class)
    assert all(name in context.link_d for name in native.MAJOR_CLASSES)
    assert context.linkable_to_prefpath["Circuit"] == "flagquantum.circuit.Circuit"
    errors = []
    for owner in [original["_components.render_signature"], reference_signatures]:
        with pytest.raises(ValueError) as error:
            owner.write_method_signature(
                io.StringIO(), circuit, circuit.members["n_qubits"]
            )
        errors.append(str(error.value))
    assert errors[0] == errors[1] and "setter" in errors[0]
    monkeypatch.setattr(
        reference_docs, "walk_and_configure", lambda *args: (pages, context)
    )
    with pytest.raises(ValueError) as error:
        reference_docs.make_reference_docs(tmp_path)
    assert str(error.value) == errors[0]
    assert (tmp_path / "docs/reference/autotoc.rst.inc").exists()
