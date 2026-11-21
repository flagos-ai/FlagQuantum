"""Complete optimization UI group compared with locked source factories."""

import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from examples.algorithms import optimization_ui_export as native
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
        (name, specs) for name, specs in source.NB_BY_SECTION if name == "Optimization"
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
    assert sum(len(specs) for _, specs in actual) == 5
    assert sum(len(spec.component_specs) for _, specs in actual for spec in specs) == 6
    assert examples.keys() == native.INSTANCES.keys() and len(examples) == 12
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


@pytest.mark.parametrize("identifier", tuple(native.INSTANCES))
def test_actual_factory_parameters_signatures_and_filenames(
    identifier: str, originals: tuple[Any, dict[str, Any]]
) -> None:
    source = originals[1][identifier]
    ports = native.signature(identifier)
    assert [(p.name, p.bitsize, p.shape, p.side.value) for p in ports] == [
        (p.name, p.bitsize, p.shape, p.side.value) for p in source.signature
    ]
    encodings = {
        "QBit": "bit",
        "QAny": "opaque",
        "QUInt": "uint",
        "BQUInt": "uint",
    }
    for port, original in zip(ports, source.signature, strict=True):
        assert port.dtype.encoding == encodings[type(original.dtype).__name__]
        if type(original.dtype).__name__ == "BQUInt":
            assert dict(port.dtype.parameters) == {
                "bound": original.dtype.iteration_length
            }
        else:
            assert not port.dtype.parameters
    metadata = source_metadata()
    assert dict(native.attributes(identifier)) == metadata["bloq_attrs"](source)
    assert json.dumps(
        native.attributes(identifier), cls=ComponentUIEncoder, sort_keys=True
    ) == json.dumps(
        metadata["bloq_attrs"](source), cls=metadata["BloqEncoder"], sort_keys=True
    )
    assert native.filename(identifier) == metadata["bloq_filename"](source)


def test_complete_fresh_outputs_preserve_symbolic_and_array_interfaces(
    tmp_path: Path,
) -> None:
    native.export_example(tmp_path)
    navigation = json.loads((tmp_path / "navigation.json").read_text())
    assert list(navigation) == ["Optimization"]
    assert len(navigation["Optimization"]) == 5
    assert len(list(tmp_path.rglob("docs.txt"))) == 6
    assert len(list(tmp_path.rglob("*.json"))) == 12
    for identifier in native.INSTANCES:
        record = json.loads(
            (
                tmp_path / native.name(identifier) / native.filename(identifier)
            ).read_text()
        )
        assert record["msd"] is not None
        # The original encoder collapses two distinct symbolic expressions.
        retained = (
            "guiding_state_symb_c" if identifier == "guiding_state_symb" else identifier
        )
        assert record["msd"] == native.score(retained)
        assert native.score(identifier) is not None
        assert record["callees"] == []
    assert native.filename("guiding_state_symb_c") == native.filename(
        "guiding_state_symb"
    )
    assert (
        native.signature("guiding_state_symb_c")[0].bitsize
        != native.signature("guiding_state_symb")[0].bitsize
    )
    assert native.score("guiding_state_symb_c") != native.score("guiding_state_symb")
    loader = native.signature("load_scopes")
    assert loader.get_left("j").bitsize == 3
    assert loader.get_right("U").bitsize == 12
    assert dict(loader.get_left("j").dtype.parameters)["bound"] == 5
    assert native.signature("kikuchi_nonzero_index")[0].shape == (8,)
    assert native.signature("kikuchi_matrix").n_qubits() == 66


def test_unknown_identifier_is_refused() -> None:
    with pytest.raises(TypeError, match="registered native example"):
        native.score("unknown")
