"""Whole Modular Arithmetic UI group against the unchanged frozen factories."""

import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from examples.algorithms import modular_ui_export as native
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
    specs = source.NB_BY_SECTION[3][1]
    return specs, {
        example.name: example.make()
        for spec in specs
        for documentation in spec.bloq_specs
        for example in documentation.examples
    }


def test_entire_source_group_and_document_factory_order(
    originals: tuple[Any, dict[str, Any]],
) -> None:
    specs, examples = originals
    actual = native.sections()[0][1]
    assert len(actual) == len(specs) == 7
    assert sum(len(spec.component_specs) for spec in actual) == 20
    assert examples.keys() == native.INSTANCES.keys() and len(examples) == 35
    for target, source in zip(actual, specs, strict=True):
        assert target.title == source.title
        assert [native.documentation_name(doc) for doc in target.component_specs] == [
            doc.bloq_cls.__name__ for doc in source.bloq_specs
        ]
        assert [
            [example.name for example in doc.examples] for doc in target.component_specs
        ] == [[example.name for example in doc.examples] for doc in source.bloq_specs]
        assert [
            [example.description for example in doc.examples]
            for doc in target.component_specs
        ] == [
            [example.docstring for example in doc.examples] for doc in source.bloq_specs
        ]


@pytest.mark.parametrize("identifier", tuple(native.INSTANCES))
def test_every_original_factory_nominal_ports_and_exact_parameters(
    identifier: str,
    originals: tuple[Any, dict[str, Any]],
) -> None:
    _, examples = originals
    source = examples[identifier]
    assert native.name(identifier) == type(source).__name__
    ports = native.signature(identifier)
    assert [(p.name, p.bitsize, p.shape, p.side.value) for p in ports] == [
        (p.name, p.bitsize, p.shape, p.side.value) for p in source.signature
    ]
    encodings = {
        "QBit": "bit",
        "QAny": "opaque",
        "QUInt": "uint",
        "QMontgomeryUInt": "montgomery",
    }
    for port, original in zip(ports, source.signature, strict=True):
        assert port.dtype.encoding == encodings[type(original.dtype).__name__]
        if type(original.dtype).__name__ == "QMontgomeryUInt":
            assert dict(port.dtype.parameters) == (
                {}
                if original.dtype.modulus is None
                else {"modulus": original.dtype.modulus}
            )
        else:
            assert not port.dtype.parameters
    metadata = source_metadata()
    # Compare actual expression objects before JSON encoding, which otherwise
    # erases symbol assumptions. Nested EC point parameters are included.
    assert dict(native.attributes(identifier)) == metadata["bloq_attrs"](source)
    assert json.dumps(
        native.attributes(identifier), cls=ComponentUIEncoder, sort_keys=True
    ) == json.dumps(
        metadata["bloq_attrs"](source), cls=metadata["BloqEncoder"], sort_keys=True
    )
    assert native.filename(identifier) == metadata["bloq_filename"](source)


def test_complete_fresh_output_including_empty_and_shaped_interfaces(
    tmp_path: Path,
) -> None:
    native.export_example(tmp_path)
    navigation = json.loads((tmp_path / "navigation.json").read_text())[
        "Modular Arithmetic"
    ]
    assert len(navigation) == 7
    assert sum(len(docs) for docs in navigation.values()) == 20
    assert (
        sum(len(doc["examples"]) for docs in navigation.values() for doc in docs) == 35
    )
    assert len(list(tmp_path.rglob("docs.txt"))) == 20
    assert len(list(tmp_path.rglob("*.json"))) == 36
    for identifier in native.INSTANCES:
        exported = json.loads(
            (
                tmp_path / native.name(identifier) / native.filename(identifier)
            ).read_text()
        )
        assert exported["msd"] is not None
        assert exported["msd"] == native.score(identifier)
        assert exported["callees"] == []
    assert tuple(native.signature("rsa_pe")) == ()
    assert tuple(native.signature("ecc")) == ()
    assert native.signature("ec_window_add_r_small")[0].shape == (4,)
    assert native.signature("modexp_small")[1].bitsize == 2048


def test_unregistered_identifier_is_refused() -> None:
    with pytest.raises(TypeError, match="registered native example"):
        native.score("unknown")
