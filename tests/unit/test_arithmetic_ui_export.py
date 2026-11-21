"""Complete Arithmetic UI catalog compared with the unchanged locked factories."""

import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from examples.algorithms import arithmetic_ui_export as native
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
    specs = source.NB_BY_SECTION[2][1]
    return specs, {
        example.name: example.make()
        for spec in specs
        for documentation in spec.bloq_specs
        for example in documentation.examples
    }


def test_complete_group_order_and_every_original_factory(
    originals: tuple[Any, dict[str, Any]],
) -> None:
    specs, examples = originals
    actual = native.sections()[0][1]
    assert len(actual) == len(specs) == 14
    assert sum(len(spec.component_specs) for spec in actual) == 45
    assert examples.keys() == native.INSTANCES.keys() and len(examples) == 74
    for target, source in zip(actual, specs, strict=True):
        assert target.title == source.title
        assert [native.documentation_name(doc) for doc in target.component_specs] == [
            doc.bloq_cls.__name__ for doc in source.bloq_specs
        ]
        assert [
            [example.name for example in doc.examples] for doc in target.component_specs
        ] == [[example.name for example in doc.examples] for doc in source.bloq_specs]


@pytest.mark.parametrize("identifier", tuple(native.INSTANCES))
def test_exact_ports_nominal_types_symbols_and_parameters(
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
        "QInt": "signed",
        "QIntOnesComp": "ones_complement",
        "QFxp": "fixed_point",
        "BQUInt": "uint",
    }
    for port, original in zip(ports, source.signature, strict=True):
        assert port.dtype.encoding == encodings[type(original.dtype).__name__]
        if type(original.dtype).__name__ == "BQUInt":
            assert (
                dict(port.dtype.parameters)["bound"] == original.dtype.iteration_length
            )
        if type(original.dtype).__name__ == "QFxp":
            assert dict(port.dtype.parameters) == {
                "fractional_width": original.dtype.num_frac,
                "signed": original.dtype.signed,
            }
    metadata = source_metadata()
    parameters = dict(native.attributes(identifier))
    if identifier.startswith("ctrl_add_or_sub"):
        for key in ("a_dtype", "b_dtype"):
            value = parameters[key]
            assert isinstance(value, dict)
            parameters[key] = {
                name: item for name, item in value.items() if name != "encoding"
            }
    assert json.dumps(parameters, cls=ComponentUIEncoder, sort_keys=True) == json.dumps(
        metadata["bloq_attrs"](source), cls=metadata["BloqEncoder"], sort_keys=True
    )
    if not identifier.startswith("ctrl_add_or_sub"):
        assert native.filename(identifier) == metadata["bloq_filename"](source)


def test_signed_conditional_filename_collision_keeps_native_type_identity(
    originals: tuple[Any, dict[str, Any]],
) -> None:
    _, examples = originals
    metadata = source_metadata()
    signed, unsigned = "ctrl_add_or_sub_signed", "ctrl_add_or_sub_unsigned"
    assert metadata["bloq_filename"](examples[signed]) == metadata["bloq_filename"](
        examples[unsigned]
    )
    assert native.filename(signed) != native.filename(unsigned)
    assert (
        native.attributes(signed)["a_dtype"] != native.attributes(unsigned)["a_dtype"]
    )


def test_all_fresh_interfaces_and_documentation_only_sort(tmp_path: Path) -> None:
    native.export_example(tmp_path)
    navigation = json.loads((tmp_path / "navigation.json").read_text())["Arithmetic"]
    assert len(navigation) == 14
    assert sum(len(docs) for docs in navigation.values()) == 45
    assert (
        sum(len(doc["examples"]) for docs in navigation.values() for doc in docs) == 74
    )
    assert len(list(tmp_path.rglob("docs.txt"))) == 45
    children = list((tmp_path / "ComponentFunction").glob("*.json"))
    assert children
    assert len(list(tmp_path.rglob("*.json"))) == 75 + len(children)
    sorts = [
        doc
        for docs in navigation.values()
        for doc in docs
        if doc["name"] == "SortInPlace"
    ]
    assert len(sorts) == 1 and sorts[0]["examples"] == []
    assert (tmp_path / "SortInPlace/docs.txt").read_text().startswith("# SortInPlace\n")
    for identifier in native.INSTANCES:
        path = tmp_path / native.name(identifier) / native.filename(identifier)
        exported = json.loads(path.read_text())
        assert exported["msd"] is not None
        assert exported["msd"] == native.score(identifier)
        record = native.decomposition(identifier)
        assert bool(exported["callees"]) == (record is not None)
        for link in exported["callees"]:
            assert (tmp_path / link["name"] / link["filename"]).is_file()
    assert native.signature("cadd_large").n_qubits() == 2001


def test_unregistered_identifier_is_refused() -> None:
    with pytest.raises(TypeError, match="registered native example"):
        native.score("unknown")


def test_all_word_dependencies_retain_existing_compiler_edges() -> None:
    from examples.algorithms.component_ui_dependencies import (
        ComponentFunction,
        dependency_graph,
    )
    from flagquantum.compiler._component_trace.program_call_graph import (
        program_component_call_graph,
    )

    graph = dependency_graph(
        native.INSTANCES,
        owner=lambda component: native.name(component)
        + "/"
        + native.filename(component),
        program=native.decomposition,
    )
    assert all(child in graph for children in graph.values() for child in children)
    for identifier in (
        "negate",
        "negate_symb",
        "xor",
        "bitwise_not",
        "bitwise_not_symb",
    ):
        record = native.decomposition(identifier)
        assert record is not None
        owner = native.name(identifier) + "/" + native.filename(identifier)
        nodes = {
            node.entry: node
            for node in graph
            if isinstance(node, ComponentFunction) and node.owner == owner
        }
        entry = record.functions[0].name
        expected = program_component_call_graph(record, entry)
        owned: dict[str, object] = {entry: identifier, **nodes}
        actual = {
            (name, child.entry)
            for name, node in owned.items()
            for child in graph[node]
            if isinstance(child, ComponentFunction)
        }
        assert actual == expected.direct_calls.keys()
        assert nodes.keys() == {
            name for edge in expected.direct_calls for name in edge if name != entry
        }
