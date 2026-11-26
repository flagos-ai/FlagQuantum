"""Full frozen signature renderer comparisons and native object naming."""

from __future__ import annotations

import csv
import hashlib
import io
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests.unit.test_development_tools import REFERENCE, load_source
from tools.ftqc_source_inputs import FTQC_SOURCE_INVENTORY

if TYPE_CHECKING:
    import griffe


pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def original() -> Any:
    return load_source("make_reference_docs/_components/render_signature")


def test_signature_source_blob() -> None:
    with (FTQC_SOURCE_INVENTORY).open(encoding="utf-8-sig") as stream:
        row = next(row for row in csv.DictReader(stream) if row["id"] == "Q-0069")
    data = (REFERENCE / row["path"]).read_bytes()
    assert (
        hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        == row["git_blob"]
    )


def test_all_parameter_kinds_annotations_defaults_and_naming() -> None:
    import griffe

    from tools.development import reference_signatures as native

    source = original()
    for kind, annotation, default in product(
        griffe.ParameterKind, [None, "int", "list[str]"], [None, "0", "None"]
    ):
        parameter = griffe.Parameter(
            "value", kind=kind, annotation=annotation, default=default
        )
        assert native.format_parameter(parameter) == source.format_parameter(parameter)
    parameter = griffe.Parameter("bad")
    for owner in [native, source]:
        with pytest.raises(ValueError) as error:
            owner.format_parameter(parameter)
        assert error.value.args == (None,)
    for name in ["Circuit", "HTTPServer", "ABC", "lower_case", "X2QFT", ""]:
        assert native.camel_to_snake(name) == source.camel_to_snake(name)
        assert native.get_obj_instance_name(
            griffe.Class(name)
        ) == source.get_obj_instance_name(griffe.Class(name))
    assert (
        native.get_obj_instance_name(griffe.Class("ComponentProgramBuilder"))
        == "builder"
    )
    assert native.get_obj_instance_name(griffe.Class("ProgramRecord")) == "program"
    assert source.get_obj_instance_name(griffe.Class("BloqBuilder")) == "bb"
    assert source.get_obj_instance_name(griffe.Class("CompositeBloq")) == "cbloq"


def method(
    *,
    first: str | None = "self",
    count: int = 0,
    returns: str | None = None,
    decorators: list[str] | None = None,
) -> griffe.Function:
    import griffe

    parameters = (
        []
        if first is None
        else [griffe.Parameter(first, kind=griffe.ParameterKind.positional_or_keyword)]
    )
    parameters.extend(
        griffe.Parameter(
            f"value{index}",
            annotation="int",
            default="0",
            kind=griffe.ParameterKind.keyword_only,
        )
        for index in range(count)
    )
    return griffe.Function(
        "operation",
        parameters=griffe.Parameters(*parameters),
        returns=returns,
        decorators=[
            griffe.Decorator(value, lineno=None, endlineno=None)
            for value in decorators or []
        ],
    )


def test_complete_function_and_generic_method_rendering() -> None:
    import griffe

    from tools.development import reference_signatures as native

    source = original()
    obj = griffe.Class("Circuit")
    for count, returns in product(range(3), [None, "Circuit"]):
        function = method(first=None, count=count, returns=returns)
        expected, actual = io.StringIO(), io.StringIO()
        source.write_function_signature(expected, function)
        native.write_function_signature(actual, function)
        assert actual.getvalue() == expected.getvalue()
        for first in [None, "self", "cls"]:
            function = method(first=first, count=count, returns=returns)
            expected, actual = io.StringIO(), io.StringIO()
            source.write_generic_method_signature(
                expected,
                obj,
                function,
                first_arg_name=first,
                caller_name=lambda value: value.name,
            )
            native.write_generic_method_signature(
                actual,
                obj,
                function,
                first_arg_name=first,
                caller_name=lambda value: value.name,
            )
            assert actual.getvalue() == expected.getvalue()
    for owner in [source, native]:
        with pytest.raises(IndexError):
            owner.write_generic_method_signature(io.StringIO(), obj, method(first=None))
        with pytest.raises(AssertionError):
            owner.write_generic_method_signature(
                io.StringIO(), obj, method(first="cls")
            )
        with pytest.raises(AssertionError):
            owner.write_property_method_signature(io.StringIO(), obj, method())


@pytest.mark.parametrize(
    "decorators,first",
    [
        ([], "self"),
        (["abc.abstractmethod"], "self"),
        (["property"], "self"),
        (["cached_property"], "self"),
        (["classmethod"], "cls"),
        (["staticmethod"], None),
        (["abc.abstractmethod", "classmethod"], "cls"),
    ],
)
def test_all_supported_method_dispatch(
    decorators: list[str], first: str | None
) -> None:
    import griffe

    from tools.development import reference_signatures as native

    source = original()
    function = method(first=first, count=2, returns="int", decorators=decorators)
    obj = griffe.Class("Circuit")
    expected, actual = io.StringIO(), io.StringIO()
    source.write_method_signature(expected, obj, function)
    native.write_method_signature(actual, obj, function)
    assert actual.getvalue() == expected.getvalue()
    function.overloads = [method()]
    for owner in [source, native]:
        stream = io.StringIO()
        owner.write_method_signature(stream, obj, function)
        assert stream.getvalue() == ""


@pytest.mark.parametrize(
    "decorators",
    [["unsupported"], ["property", "staticmethod"], ["functools.cached_property"]],
)
def test_original_decorator_rejections_preserved(decorators: list[str]) -> None:
    import griffe

    from tools.development import reference_signatures as native

    errors = []
    for owner in [original(), native]:
        with pytest.raises(ValueError) as error:
            owner.write_method_signature(
                io.StringIO(), griffe.Class("Circuit"), method(decorators=decorators)
            )
        errors.append(str(error.value))
    assert errors[0] == errors[1]


def test_native_signature_uses_existing_reference_link_writer(tmp_path: Path) -> None:
    import griffe

    from tools.development import reference_signatures as native
    from tools.development.files import WriteIfDifferent
    from tools.development.reference_links import LinkingWriter
    from tools.development.reference_pages import MajorClassPage

    path = tmp_path / "append.md"
    with WriteIfDifferent(path) as stream:
        writer = LinkingWriter(
            stream,
            link_aliases={"Circuit": "flagquantum.Circuit"},
            link_d={
                "flagquantum.Circuit": (
                    MajorClassPage(pref_path="flagquantum.Circuit"),
                    None,
                )
            },
            refdoc_relpath=Path("reference"),
        )
        native.write_method_signature(
            writer,
            griffe.Class("Circuit"),
            griffe.Function(
                "append",
                parameters=griffe.Parameters(
                    griffe.Parameter("self"),
                    griffe.Parameter(
                        "op", kind=griffe.ParameterKind.positional_or_keyword
                    ),
                ),
                returns="Circuit",
            ),
        )
        writer.write("Use `Circuit` with circuit.append(op).\n")
        writer.write_link_targets()
    assert "circuit.append(op) -> Circuit" in path.read_text()
    assert "[`Circuit`]: /reference/flagquantum/Circuit.md" in path.read_text()
