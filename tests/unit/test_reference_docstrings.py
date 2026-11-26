"""Complete Griffe section renderer and immutable source corpus comparisons."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import sys
import types
import warnings
from typing import TYPE_CHECKING, Any

import pytest

from tests.unit.test_development_tools import REFERENCE, load_source
from tools.ftqc_source_inputs import FTQC_SOURCE_INVENTORY

if TYPE_CHECKING:
    import griffe


pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture
def original(monkeypatch: pytest.MonkeyPatch) -> Any:
    root = "_render_reference"
    folder = REFERENCE / "dev_tools" / "qualtran_dev_tools"
    for suffix, path in [
        ("", folder),
        (".make_reference_docs", folder / "make_reference_docs"),
        (
            ".make_reference_docs._components",
            folder / "make_reference_docs/_components",
        ),
    ]:
        package = types.ModuleType(root + suffix)
        package.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(
        sys.modules, root + ".parse_docstrings", load_source("parse_docstrings")
    )
    spec = importlib.util.spec_from_file_location(
        root + ".make_reference_docs._components.render_docstring",
        folder / "make_reference_docs/_components/render_docstring.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_docstring_source_blob() -> None:
    with (FTQC_SOURCE_INVENTORY).open(encoding="utf-8-sig") as stream:
        row = next(row for row in csv.DictReader(stream) if row["id"] == "Q-0068")
    data = (REFERENCE / row["path"]).read_bytes()
    assert (
        hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        == row["git_blob"]
    )


def sections() -> list[griffe.DocstringSection]:
    import griffe

    return [
        griffe.DocstringSectionText("  Text and `Circuit`.  "),
        griffe.DocstringSectionParameters(
            [
                griffe.DocstringParameter(
                    "typed", annotation="int", description="A number."
                ),
                griffe.DocstringParameter("untyped", description="A value."),
            ]
        ),
        griffe.DocstringSectionAttributes(
            [
                griffe.DocstringAttribute(
                    "typed", annotation="int", description="Stored number."
                ),
                griffe.DocstringAttribute("untyped", description="Stored value."),
            ]
        ),
        griffe.DocstringSectionReturns(
            [griffe.DocstringReturn("", description="Single value.")]
        ),
        griffe.DocstringSectionReturns(
            [
                griffe.DocstringReturn("named", description="First value."),
                griffe.DocstringReturn("", description="Second value."),
            ]
        ),
        griffe.DocstringSectionYields(
            [griffe.DocstringYield("", description="Single yield.")]
        ),
        griffe.DocstringSectionYields(
            [
                griffe.DocstringYield("named", description="First yield."),
                griffe.DocstringYield("", description="Second yield."),
            ]
        ),
        griffe.DocstringSectionRaises(
            [
                griffe.DocstringRaise(
                    annotation="ValueError", description="Invalid input."
                )
            ]
        ),
        griffe.DocstringSectionAdmonition("note", "Keep the workspace clean."),
        griffe.DocstringSectionAdmonition("see-also", "Circuit.append"),
        griffe.DocstringSectionAdmonition(
            "references",
            "[A title](https://example.com). Author.\n\nUnparsed citation.",
        ),
        griffe.DocstringSectionExamples(
            [
                (griffe.DocstringSectionKind.text, "A narrative."),
                (griffe.DocstringSectionKind.examples, ">>> 1 + 1\n2"),
            ]
        ),
        griffe.DocstringSectionAdmonition("unknown", "Unknown admonition."),
        griffe.DocstringSectionDeprecated("1.0", "Deprecated section."),
    ]


def rendered(
    owner: Any, parts: list[griffe.DocstringSection], level: int
) -> tuple[str, list[str]]:
    stream = io.StringIO()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        owner.write_docstring_parts(stream, parts, level)
    return stream.getvalue(), [str(warning.message) for warning in captured]


@pytest.mark.parametrize("level", [-1, 0, 1, 3])
def test_all_section_functions_and_full_dispatch(original: Any, level: int) -> None:
    from tools.development import reference_docstrings as native

    parts = sections()
    for part in parts:
        assert rendered(native, [part], level) == rendered(original, [part], level)
    assert rendered(native, parts, level) == rendered(original, parts, level)
    assert rendered(native, [], level) == rendered(original, [], level) == ("\n\n", [])
    assert "\n````\n" in rendered(native, parts, level)[0]


def test_original_section_errors_preserved(original: Any) -> None:
    import griffe

    from tools.development import reference_docstrings as native

    invalid_examples = griffe.DocstringSectionExamples([])
    setattr(
        invalid_examples, "value", [(griffe.DocstringSectionKind.parameters, "invalid")]
    )
    for owner in [original, native]:
        with pytest.raises(ValueError):
            owner.write_docstring_parts(io.StringIO(), [invalid_examples], 0)
        with pytest.raises(ValueError):
            owner.write_docstring_parts(
                io.StringIO(),
                [
                    griffe.DocstringSectionRaises(
                        [griffe.DocstringRaise(description="missing type")]
                    )
                ],
                0,
            )
        for text in [""]:
            with pytest.raises(ValueError):
                owner.split_docstring(
                    griffe.Class("Example", docstring=griffe.Docstring(text))
                )


@pytest.mark.parametrize(
    "text",
    [
        None,
        "Summary.",
        "Args:\n    value: Parameter without summary.",
        "Summary.\n\nSecond paragraph.\n\nThird paragraph.",
        "Summary.\n\nReturns:\n    int: A value.",
    ],
)
def test_summary_and_remaining_parts(original: Any, text: str | None) -> None:
    import griffe

    from tools.development import reference_docstrings as native

    obj = griffe.Class(
        "Example", docstring=griffe.Docstring(text) if text is not None else None
    )
    expected, source_parts = original.split_docstring(obj)
    actual, parts = native.split_docstring(obj)
    assert actual == expected
    assert rendered(native, parts, 0) == rendered(original, source_parts, 0)


def test_entire_frozen_class_documentation_corpus(
    original: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import griffe

    from tools.development import reference_docstrings as native

    package = types.ModuleType("_development_reference")
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(
        sys.modules, package.__name__ + ".git_tools", load_source("git_tools")
    )
    finder = load_source("bloq_finder")
    classes = finder.get_bloq_classes(bloqs_root=REFERENCE / "qualtran/bloqs")
    assert len(classes) == 340
    for cls in classes:
        obj = griffe.Class(
            cls.__name__,
            docstring=griffe.Docstring(cls.__doc__) if cls.__doc__ else None,
        )
        expected, source_parts = original.split_docstring(obj)
        actual, parts = native.split_docstring(obj)
        assert actual == expected
        assert rendered(native, parts, 0) == rendered(original, source_parts, 0)


def test_nontext_first_section_rejection(
    original: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import griffe

    from tools.development import reference_docstrings as native

    monkeypatch.setattr(
        griffe.Docstring,
        "parse",
        lambda self, parser: [griffe.DocstringSectionParameters([])],
    )
    obj = griffe.Class("Example", docstring=griffe.Docstring("description"))
    for owner in [original, native]:
        with pytest.raises(ValueError) as error:
            owner.split_docstring(obj)
        assert error.value.args == (obj,)
