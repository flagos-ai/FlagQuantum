"""Complete block-encoding UI group compared with locked source factories."""

import importlib
import json
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from examples.algorithms import block_encoding_ui_export as native
from tests.unit.test_component_ui_export import source_metadata
from tests.unit.test_development_tools import REFERENCE
from tools.development.component_ui_export import ComponentUIEncoder

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture
def originals(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, dict[str, Any]]:
    package = types.ModuleType("qualtran_dev_tools")
    package.__path__ = [str(REFERENCE / "dev_tools/qualtran_dev_tools")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    repository = importlib.import_module("qualtran_dev_tools.git_tools")
    monkeypatch.setattr(repository, "get_git_root", lambda: REFERENCE)
    monkeypatch.delitem(sys.modules, "qualtran_dev_tools.notebook_specs", raising=False)
    source = importlib.import_module("qualtran_dev_tools.notebook_specs")
    sections = tuple(
        (name, specs)
        for name, specs in source.NB_BY_SECTION
        if name == "Block Encoding"
    )
    return sections, {
        example.name: example.make()
        for _, specs in sections
        for spec in specs
        for documentation in spec.bloq_specs
        for example in documentation.examples
    }


def test_complete_group_and_factory_order(
    originals: tuple[Any, dict[str, Any]],
) -> None:
    sections, examples = originals
    actual = native.sections()
    assert len(actual) == len(sections) == 1
    assert sum(len(specs) for _, specs in actual) == 10
    assert sum(len(spec.component_specs) for _, specs in actual for spec in specs) == 16
    assert examples.keys() == native.INSTANCES.keys() and len(examples) == 26
    for (name, specs), (original_name, originals_specs) in zip(
        actual, sections, strict=True
    ):
        assert name == original_name
        for target, source in zip(specs, originals_specs, strict=True):
            assert target.title == source.title
            assert [
                native.documentation_name(doc) for doc in target.component_specs
            ] == [doc.bloq_cls.__name__ for doc in source.bloq_specs]
            assert [
                [example.name for example in doc.examples]
                for doc in target.component_specs
            ] == [
                [example.name for example in doc.examples] for doc in source.bloq_specs
            ]


def _assert_parameters(actual: object, expected: object) -> None:
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert isinstance(actual, Mapping) and actual.keys() == expected.keys()
        for key in expected:
            _assert_parameters(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert isinstance(actual, (tuple, list)) and len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _assert_parameters(left, right)
    else:
        assert actual == expected


@pytest.mark.parametrize("identifier", tuple(native.INSTANCES))
def test_all_actual_factories_signatures_nested_parameters_and_filenames(
    identifier: str, originals: tuple[Any, dict[str, Any]]
) -> None:
    source = originals[1][identifier]
    ports = native.signature(identifier)
    assert [(p.name, p.bitsize, p.shape, p.side.value) for p in ports] == [
        (p.name, p.bitsize, p.shape, p.side.value) for p in source.signature
    ]
    for port, original in zip(ports, source.signature, strict=True):
        assert port.dtype.encoding == (
            "uint" if type(original.dtype).__name__ == "BQUInt" else "opaque"
        )
        assert dict(port.dtype.parameters) == (
            {"bound": original.dtype.iteration_length}
            if type(original.dtype).__name__ == "BQUInt"
            else {}
        )
    metadata = source_metadata()
    _assert_parameters(native.attributes(identifier), metadata["bloq_attrs"](source))
    assert json.dumps(
        native.attributes(identifier), cls=ComponentUIEncoder, sort_keys=True
    ) == json.dumps(
        metadata["bloq_attrs"](source), cls=metadata["BloqEncoder"], sort_keys=True
    )
    assert native.filename(identifier) == metadata["bloq_filename"](source)


def test_complete_fresh_interfaces_and_empty_abstract_documents(tmp_path: Path) -> None:
    native.export_example(tmp_path)
    navigation = json.loads((tmp_path / "navigation.json").read_text())
    assert len(navigation["Block Encoding"]) == 10
    assert len(list(tmp_path.rglob("docs.txt"))) == 16
    assert len(list(tmp_path.rglob("*.json"))) == 27
    for identifier in native.INSTANCES:
        record = json.loads(
            (
                tmp_path / native.name(identifier) / native.filename(identifier)
            ).read_text()
        )
        assert record["msd"] is not None
        assert record["msd"] == native.score(identifier)
        assert record["callees"] == []
    empty = [
        doc
        for _, specs in native.sections()
        for spec in specs
        for doc in spec.component_specs
        if not doc.examples
    ]
    assert [native.documentation_name(doc) for doc in empty] == [
        "BlockEncoding",
        "SelectOracle",
        "PrepareOracle",
    ]
    assert (
        native.signature("lcu_block").n_qubits()
        == native.signature("black_box_lcu_block").n_qubits()
        == 32
    )


def test_unregistered_identifier_is_refused() -> None:
    with pytest.raises(TypeError, match="registered native example"):
        native.score("unknown")
