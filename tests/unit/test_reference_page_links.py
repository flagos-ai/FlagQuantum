"""Complete frozen reference page, context, link and alias source comparisons."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import sys
import types
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests.unit.test_development_tools import REFERENCE
from tools.development.files import WriteIfDifferent
from tools.ftqc_source_inputs import FTQC_SOURCE_INVENTORY

if TYPE_CHECKING:

    from tools.development.reference_pages import (
        Page,
    )


pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture
def original(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    root = "_page_link_reference"
    folder = REFERENCE / "dev_tools" / "qualtran_dev_tools" / "make_reference_docs"
    for name, path in [(root, folder), (root + "._components", folder / "_components")]:
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, package)
    result: dict[str, Any] = {}
    for name in ["_page", "_linking_writer", "_render_context", "_components.aliases"]:
        path = folder / (name.replace(".", "/") + ".py")
        spec = importlib.util.spec_from_file_location(root + "." + name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
        result[name] = module
    return result


def test_frozen_blobs_and_optional_dependency_versions() -> None:
    selected = {"Q-0067", "Q-0070", "Q-0072", "Q-0077", "Q-0104"}
    with (FTQC_SOURCE_INVENTORY).open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row["id"] not in selected:
                continue
            data = (REFERENCE / row["path"]).read_bytes()
            assert (
                hashlib.sha1(
                    b"blob " + str(len(data)).encode() + b"\0" + data
                ).hexdigest()
                == row["git_blob"]
            )
            selected.remove(row["id"])
    assert not selected
    lock = (REFERENCE / "dev_tools/requirements/envs/docs.env.txt").read_text()
    for name in ["griffe", "mdit-py-plugins", "markdown-it-py", "mdurl", "colorama"]:
        assert f"{name}=={metadata.version(name)}\n" in lock


@pytest.mark.parametrize("major", [False, True])
@pytest.mark.parametrize(
    "path", ["flagquantum.Circuit", "flagquantum.compiler.Program", "standalone", ""]
)
def test_complete_page_records(
    original: dict[str, Any], tmp_path: Path, major: bool, path: str
) -> None:
    import griffe

    from tools.development.reference_pages import (
        MajorClassPage,
        MemberType,
        ModulePage,
        ModulePageMember,
    )

    original_type = (
        original["_page"].MajorClassPage if major else original["_page"].ModulePage
    )
    native_type = MajorClassPage if major else ModulePage
    source = original_type()
    actual = native_type()
    for page in [source, actual]:
        with pytest.raises(AssertionError, match="Uninitialized"):
            page.out_path(tmp_path)
        with pytest.raises(AssertionError, match="Uninitialized"):
            _ = page.section_not_none
        page.pref_path = path
        page.section = "flagquantum"
    assert actual.out_path(tmp_path) == source.out_path(tmp_path)
    assert actual.section_not_none == source.section_not_none
    assert actual.obj is source.obj is None
    assert actual == native_type(section="flagquantum", pref_path=path)
    if not major:
        module = ModulePage()
        function = griffe.Function("execute")
        source_member = original["_page"].ModulePageMember(
            function, "flagquantum.execute", original["_page"].MemberType.MINOR
        )
        member = ModulePageMember(function, "flagquantum.execute", MemberType.MINOR)
        assert (
            member.obj is source_member.obj
            and member.pref_dotpath == source_member.pref_dotpath
        )
        assert member.mytype.value == source_member.mytype.value
        module.members.append(member)
        assert not ModulePage().members
    assert [value.value for value in MemberType] == [
        value.value for value in original["_page"].MemberType
    ]


@pytest.mark.parametrize(
    "aliases",
    [
        {},
        {"other": "other"},
        {"one": "fq.Circuit"},
        {"fq.Circuit": "fq.Circuit", "second": "fq.Circuit", "first": "fq.Circuit"},
    ],
)
def test_complete_alias_formatter(
    original: dict[str, Any], aliases: dict[str, str]
) -> None:
    from tools.development.reference_links import get_aliases_str

    assert get_aliases_str("fq.Circuit", aliases_d=aliases) == original[
        "_components.aliases"
    ].get_aliases_str("fq.Circuit", aliases_d=aliases)


def mappings() -> tuple[dict[str, str], dict[str, tuple[Page, str | None]]]:
    from tools.development.reference_pages import MajorClassPage, ModulePage

    page = ModulePage(pref_path="flagquantum.compiler")
    major = MajorClassPage(pref_path="flagquantum.Circuit")
    return (
        {
            "Circuit": "flagquantum.Circuit",
            "fq.Circuit": "flagquantum.Circuit",
            "append": "flagquantum.Circuit.append",
            "word": "word",
            "missing-target": "absent",
        },
        {
            "flagquantum.Circuit": (major, None),
            "flagquantum.Circuit.append": (major, "Append_Operation & λ"),
            "word": (page, "word"),
        },
    )


@pytest.mark.parametrize(
    "root", [Path("reference"), Path("nested/reference"), Path("/")]
)
def test_complete_link_writer_and_context(original: dict[str, Any], root: Path) -> None:
    from tools.development.reference_context import RenderContext

    aliases, targets = mappings()
    source_stream = io.StringIO()
    native_stream = io.StringIO()
    source_context = original["_render_context"].RenderContext(
        root, ["flagquantum"], aliases, targets, aliases
    )
    context = RenderContext(root, ["flagquantum"], aliases, targets, aliases)
    source = source_context.get_linking_writer(source_stream)
    actual = context.get_linking_writer(native_stream)
    assert (
        context.sections,
        context.aliases_d,
        context.link_d,
        context.linkable_to_prefpath,
    ) == (
        source_context.sections,
        source_context.aliases_d,
        source_context.link_d,
        source_context.linkable_to_prefpath,
    )
    for name in [
        "Circuit",
        "fq.Circuit",
        "append",
        "word",
        "missing",
        "missing-target",
    ]:
        assert actual.resolve_dotpath(name) == source.resolve_dotpath(name)
    text = "`Circuit` `fq.Circuit` `append` `word` `missing` `missing-target` `call()` `name space` `ΩName`"
    assert actual.linkify(text) == source.linkify(text)
    assert actual.linkify(actual.linkify(text)) == source.linkify(source.linkify(text))
    for writer in [source, actual]:
        writer.write(text)
        writer.write_nl("\n`Circuit` stays literal here.\n")
        writer.write_link_targets()
        writer.write_link_targets()
    assert native_stream.getvalue() == source_stream.getvalue()
    assert "#append_operation--λ" in native_stream.getvalue()


def test_original_cross_document_cache_defect_and_native_mapping_updates(
    original: dict[str, Any],
) -> None:
    from tools.development.reference_links import LinkingWriter
    from tools.development.reference_pages import ModulePage

    source_type = original["_linking_writer"].LinkingWriter
    aliases = {"shared": "shared"}
    source_map: dict[str, tuple[Page, str | None]] = {
        "shared": (ModulePage(pref_path="native.first"), None)
    }
    second_map: dict[str, tuple[Page, str | None]] = {
        "shared": (ModulePage(pref_path="native.second"), None)
    }

    def writer(owner: Any, targets: dict[str, tuple[Page, str | None]]) -> Any:
        return owner(
            io.StringIO(),
            link_aliases=aliases,
            link_d=targets,
            refdoc_relpath=Path("reference"),
        )

    assert (
        writer(source_type, source_map).resolve_dotpath("shared")
        == "/reference/native/first.md"
    )
    assert (
        writer(source_type, second_map).resolve_dotpath("shared")
        == "/reference/native/first.md"
    )
    actual = writer(LinkingWriter, second_map)
    assert actual.resolve_dotpath("shared") == "/reference/native/second.md"
    second_map["shared"] = (ModulePage(pref_path="native.third"), "updated")
    assert actual.resolve_dotpath("shared") == "/reference/native/third.md#updated"


def test_native_page_links_reuse_existing_equal_content_writer(tmp_path: Path) -> None:
    from tools.development.reference_context import RenderContext

    path = tmp_path / "circuit.md"
    aliases, targets = mappings()
    context = RenderContext(
        Path("reference"), ["flagquantum"], aliases, targets, aliases
    )
    stamps = []
    for _ in range(2):
        with WriteIfDifferent(path) as stream:
            writer = context.get_linking_writer(stream)
            writer.write("Use `Circuit` and `append` for circuit.append(op).\n")
            writer.write_link_targets()
        stamps.append(path.stat().st_mtime_ns)
    assert stamps[0] == stamps[1]
    assert "[`Circuit`]: /reference/flagquantum/Circuit.md" in path.read_text()
