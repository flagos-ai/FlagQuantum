"""Complete frozen documentation parsing and its reference corpus."""

import csv
import hashlib
import sys
import types
from typing import Any

import pytest

from tests.unit.test_development_tools import REFERENCE, load_source
from tools.ftqc_source_inputs import FTQC_SOURCE_INVENTORY

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture
def docstrings() -> types.ModuleType:
    from tools.development import docstrings as parser

    return parser


@pytest.fixture
def source() -> types.ModuleType:
    return load_source("parse_docstrings")


def test_complete_source_blobs() -> None:
    remaining = {"Q-0081", "Q-0082"}
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


def reference_record(reference: Any) -> tuple[str, str, tuple[str, ...]]:
    fields = tuple(
        getattr(reference, field)
        for field in ("url", "title", "extra")
        if hasattr(reference, field)
    )
    return type(reference).__name__, reference.text, fields


@pytest.mark.parametrize(
    "text",
    [
        "",
        "not a reference",
        "[Title](https://example.com). extra",
        "[Title](http://example.com/path?a=b#c).",
        "[www](www.example.com)",
        "[Unicode λ](https://example.com)",
        "[Title](https://example.com:42)",
        "[A-B_(test)](https://example.com/a_(b)) additional\ntext",
        "[Title](https://example.com).\n\n[Second](http://example.org)",
        "[Title](ftp://example.com)",
    ],
)
def test_complete_reference_parsing(
    source: types.ModuleType, docstrings: types.ModuleType, text: str
) -> None:
    assert reference_record(docstrings.parse_reference(text)) == reference_record(
        source.parse_reference(text)
    )
    assert list(map(reference_record, docstrings.parse_references(text))) == list(
        map(reference_record, source.parse_references(text))
    )


@pytest.mark.parametrize(
    "text",
    [
        None,
        "",
        "A class description.",
        """Description.

    Parameters:
        value (int): A parameter.
            Continuation text.

    Registers:
        selection: The MSB address.
        work: Clean workspace.

    References:
        [Title](https://example.com). Citation.

        Unstructured reference.

    Notes:
        Use :py:func:`helper` and :py:func:`pkg.helper`.
    """,
    ],
)
def test_class_documentation(
    source: types.ModuleType, docstrings: types.ModuleType, text: str | None
) -> None:
    cls = type("Documented", (), {"__doc__": text})
    assert docstrings.get_markdown_docstring(cls) == source.get_markdown_docstring(cls)
    assert docstrings.get_markdown_docstring_lines(
        cls
    ) == source.get_markdown_docstring_lines(cls)
    assert list(map(reference_record, docstrings.get_references(cls))) == list(
        map(reference_record, source.get_references(cls))
    )


def test_frozen_reference_records(
    source: types.ModuleType, docstrings: types.ModuleType
) -> None:
    from attrs.exceptions import FrozenInstanceError

    for owner in (source, docstrings):
        structured = owner.Reference(
            title="Title", url="https://example.com", extra="Author"
        )
        assert structured.text == "[Title](https://example.com). Author"
        assert structured == owner.Reference(
            title="Title", url="https://example.com", extra="Author"
        )
        assert hash(structured) == hash(
            owner.Reference(title="Title", url="https://example.com", extra="Author")
        )
        with pytest.raises(FrozenInstanceError):
            setattr(structured, "title", "changed")
        with pytest.raises(TypeError):
            owner.Reference("https://example.com", "Title", "Author")
        fallback = owner.UnparsedReference("literal")
        assert fallback.text == "literal"
        with pytest.raises(FrozenInstanceError):
            setattr(fallback, "text", "changed")


def test_unmodified_original_golden_case(
    monkeypatch: pytest.MonkeyPatch, docstrings: types.ModuleType
) -> None:
    package = types.ModuleType("_development_reference")
    finder = types.ModuleType("_development_reference.bloq_finder")
    setattr(finder, "get_bloq_classes", lambda: [])
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, finder.__name__, finder)
    monkeypatch.setitem(sys.modules, package.__name__ + ".parse_docstrings", docstrings)
    original = load_source("parse_docstrings_test")
    original.test_get_markdown_docstring_lines()


def test_entire_frozen_class_reference_corpus(
    source: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    docstrings: types.ModuleType,
) -> None:
    package = types.ModuleType("_development_reference")
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(
        sys.modules, package.__name__ + ".git_tools", load_source("git_tools")
    )
    finder = load_source("bloq_finder")
    classes = finder.get_bloq_classes(bloqs_root=REFERENCE / "qualtran" / "bloqs")
    assert len(classes) > 100
    print(f"Verified complete source documentation corpus of {len(classes)} classes")
    for cls in classes:
        assert docstrings.get_markdown_docstring_lines(
            cls
        ) == source.get_markdown_docstring_lines(cls)
        assert list(map(reference_record, docstrings.get_references(cls))) == list(
            map(reference_record, source.get_references(cls))
        )
    setattr(finder, "get_bloq_classes", lambda: classes)
    monkeypatch.setitem(sys.modules, package.__name__ + ".bloq_finder", finder)
    for parser in (source, docstrings):
        monkeypatch.setitem(sys.modules, package.__name__ + ".parse_docstrings", parser)
        original = load_source("parse_docstrings_test")
        original.test_parse_all_references()
