"""Every stored dependency has a native node and a resolvable publication link."""

import json
from pathlib import Path

import pytest

from examples.algorithms import register_gate_ui_export as native
from examples.algorithms.basic_gate_decomposition import decomposition_program
from examples.algorithms.component_ui_dependencies import (
    ComponentFunction,
    dependency_graph,
)
from flagquantum.compiler._component_trace.program_call_graph import (
    program_component_call_graph,
)
from tests.unit.test_register_gate_ui_export import EXAMPLES

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def graph() -> dict[object, tuple[object, ...]]:
    return dependency_graph(
        EXAMPLES.values(),
        owner=lambda component: native.name(component)
        + "/"
        + native.filename(component),
    )


def test_complete_reachable_native_functions_and_edges() -> None:
    actual = graph()
    assert set(EXAMPLES.values()) <= actual.keys()
    assert all(child in actual for children in actual.values() for child in children)
    for root in EXAMPLES.values():
        record = decomposition_program(root)
        if record is None:
            assert actual[root] == ()
            continue
        entry = record.functions[0].name
        calls = program_component_call_graph(record, entry)
        owned: dict[str, object] = {
            node.entry: node
            for node in actual
            if isinstance(node, ComponentFunction)
            and node.owner == native.name(root) + "/" + native.filename(root)
        }
        owned[entry] = root
        expected_names = {
            entry,
            *(name for edge in calls.direct_calls for name in edge),
        }
        assert owned.keys() == expected_names
        edges = {
            (name, child.entry)
            for name, node in owned.items()
            for child in actual[node]
            if isinstance(child, ComponentFunction)
        }
        assert edges == calls.direct_calls.keys()
        for name, node in owned.items():
            if isinstance(node, ComponentFunction):
                assert node.function == next(
                    f for f in record.functions if f.name == name
                )
                assert node.score() is not None


def test_native_dependency_publication_resolves_every_link(tmp_path: Path) -> None:
    actual = graph()
    native.export_example(tmp_path)
    paths = set()
    for node, children in actual.items():
        path = tmp_path / native.name(node) / native.filename(node)
        data = json.loads(path.read_text())
        paths.add(path)
        assert data["msd"] is not None
        assert data["callees"] == [
            {"name": native.name(child), "filename": native.filename(child)}
            for child in children
        ]
        for link in data["callees"]:
            assert (tmp_path / link["name"] / link["filename"]).is_file()
    assert paths == set(tmp_path.rglob("*.json")) - {tmp_path / "navigation.json"}
    functions = [node for node in actual if isinstance(node, ComponentFunction)]
    assert functions and len({(node.owner, node.entry) for node in functions}) == len(
        functions
    )
    # Shared source primitives retain distinct owning Program contexts.
    assert len({node.entry for node in functions}) < len(functions)
