"""Complete library-graph traversal against the locked source example corpus."""

import csv
import hashlib
import importlib.util
import logging
import sys
import types
from collections import Counter
from collections.abc import Callable
from functools import partial
from typing import Any

import pytest
import sympy

from flagquantum.compiler._component_trace.call_graph import (
    SymbolAllocator,
    symbolic_call_graph,
    symbolic_call_graph_union,
)
from flagquantum.errors import CapabilityError
from flagquantum.testing.component_examples import ComponentExample
from tests.unit.test_development_tools import REFERENCE
from tools.development.library_call_graph import (
    library_call_graph,
    resource_library_call_graph,
)
from tools.ftqc_source_inputs import FTQC_SOURCE_INVENTORY

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_shared_visit_roots_symbols_and_omission(
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: Counter[str] = Counter()
    factories: Counter[str] = Counter()

    def make(name: str) -> str:
        factories[name] += 1
        return name

    examples = [
        ComponentExample(partial(make, name), name, str)
        for name in ("root-a", "root-b", "root-a", "ignored-root")
    ]

    def declare(node: str, allocator: SymbolAllocator) -> dict[str, Any]:
        calls[node] += 1
        if node.startswith("root"):
            return {"shared": allocator.new_symbol("n")}
        return {}

    def generalize(node: str) -> str | None:
        return None if node == "ignored-root" else node

    with caplog.at_level(logging.INFO):
        graph = library_call_graph(examples, declare, generalizer=generalize)
    n = sympy.Symbol("_n0")
    assert graph == {
        "root-a": {"shared": n},
        "shared": {},
        "root-b": {"shared": n},
        "ignored-root": {},
    }
    assert calls == Counter({"root-a": 1, "root-b": 1, "shared": 1, "ignored-root": 1})
    assert factories == Counter({"root-a": 2, "root-b": 1, "ignored-root": 1})
    assert len(caplog.records) == 4
    assert "Building call graph for: ignored-root" in caplog.text
    with pytest.raises(ValueError, match="Cannot remove the root"):
        symbolic_call_graph("ignored-root", declare, generalizer=generalize)


def test_union_native_failures_and_terminals() -> None:
    def declare(node: str, allocator: SymbolAllocator) -> dict[str, int]:
        if node == "root":
            return {"missing": 1, "kept": 2}
        if node == "missing":
            raise CapabilityError("No declared model")
        raise AssertionError("Kept leaf must not be expanded")

    assert symbolic_call_graph_union(
        ["root"], declare, keep=lambda node: node == "kept"
    ) == {"root": {"missing": 1, "kept": 2}, "missing": {}, "kept": {}}
    assert symbolic_call_graph_union([], declare) == {}
    with pytest.raises(ValueError, match="cycle"):
        symbolic_call_graph_union(["a"], lambda node, allocator: {node: 1})
    assert symbolic_call_graph_union(
        ["a"], lambda node, allocator: {"b": -1} if node == "a" else {}
    ) == {"a": {"b": -1}, "b": {}}
    with pytest.raises(RuntimeError, match="factory"):
        library_call_graph(
            [
                ComponentExample(
                    lambda: (_ for _ in ()).throw(RuntimeError("factory")),
                    "failure",
                    str,
                )
            ],
            declare,
        )


def test_source_library_file_blobs() -> None:
    selected = {"Q-0055", "Q-0056"}
    with (FTQC_SOURCE_INVENTORY).open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row["id"] in selected:
                data = (REFERENCE / row["path"]).read_bytes()
                assert (
                    hashlib.sha1(
                        b"blob " + str(len(data)).encode() + b"\0" + data
                    ).hexdigest()
                    == row["git_blob"]
                )
                selected.remove(row["id"])
    assert not selected


def load_reference(name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_library_reference." + name,
        REFERENCE / "dev_tools" / "qualtran_dev_tools" / (name + ".py"),
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_complete_original_library_graph(
    monkeypatch: pytest.MonkeyPatch, pytestconfig: pytest.Config
) -> None:
    import networkx as nx
    from qualtran._infra.bloq import DecomposeNotImplementedError, DecomposeTypeError

    # Unmodified source modules declare this marker during library discovery.
    pytestconfig.addinivalue_line(
        "markers", "notebook: unit tests that execute a notebook."
    )
    monkeypatch.syspath_prepend(str(REFERENCE / "openfermion-reference"))
    package = types.ModuleType("_library_reference")
    package.__path__ = [str(REFERENCE / "dev_tools" / "qualtran_dev_tools")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    original = load_reference("all_call_graph")
    finder = load_reference("bloq_finder")
    examples = finder.get_bloq_examples(REFERENCE / "qualtran" / "bloqs")
    assert len(examples) == 393
    profiles: list[tuple[Any, Callable[[Any], bool]]] = []
    original_builder = original._build_call_graph

    def capture_profile(**kwargs: Any) -> None:
        if not profiles:
            profiles.append((kwargs["generalizer"], kwargs["keep"]))
        original_builder(**kwargs)

    monkeypatch.setattr(original, "_build_call_graph", capture_profile)

    def identity(value: Any) -> Any:
        return value

    # Use the same constructed roots: several original factories create fresh
    # callable or object identities on every invocation.
    replay = [
        types.SimpleNamespace(name=example.name, make=partial(identity, example.make()))
        for example in examples
    ]
    expected = original.get_all_call_graph(replay)
    assert len(expected) == 3693 and len(expected.edges) == 8550
    generalize, keep = profiles[0]

    def declare(node: Any, allocator: SymbolAllocator) -> Any:
        try:
            return node.build_call_graph(allocator)
        except (DecomposeNotImplementedError, DecomposeTypeError) as error:
            raise CapabilityError(str(error)) from error

    native_examples = [
        ComponentExample(example.make, example.name, object) for example in replay
    ]
    actual = library_call_graph(
        native_examples, declare, generalizer=generalize, keep=keep
    )
    assert set(actual) == set(expected.nodes)
    assert {
        (parent, child): count
        for parent, children in actual.items()
        for child, count in children.items()
    } == {
        (parent, child): data["n"] for parent, child, data in expected.edges(data=True)
    }
    assert list(nx.simple_cycles(nx.DiGraph(actual))) == []

    # Execute the original whole-library acceptance body with the native graph
    # consumer; only the discovery root and graph representation are adapted.
    native_provider = types.ModuleType("_library_reference.all_call_graph")

    def native_graph(items: Any) -> Any:
        records = [ComponentExample(item.make, item.name, object) for item in items]
        adjacency = library_call_graph(
            records, declare, generalizer=generalize, keep=keep
        )
        graph = nx.DiGraph()
        graph.add_nodes_from(adjacency)
        for parent, children in adjacency.items():
            for child, count in children.items():
                graph.add_edge(parent, child, n=count)
        return graph

    setattr(native_provider, "get_all_call_graph", native_graph)
    monkeypatch.setitem(sys.modules, native_provider.__name__, native_provider)
    setattr(finder, "get_bloq_examples", lambda: replay)
    monkeypatch.setitem(sys.modules, "_library_reference.bloq_finder", finder)
    original_test = load_reference("all_call_graph_test")
    original_test.test_get_all_call_graph()


def test_native_resource_profile_reuses_owned_generalizers() -> None:
    from flagquantum.compiler._component_trace.call_graph import RawCallees
    from flagquantum.compiler._component_trace.resource_generalization import (
        BookkeepingCall,
        CliffordCall,
        PhaseCall,
        ResourceCall,
        ResourceSequenceCall,
        RotationCall,
        _resource_callees,
    )

    root = ResourceSequenceCall(
        (
            RotationCall("rx", 0.1),
            RotationCall("rx", 0.2),
            PhaseCall("t", True),
            CliffordCall("h", 1),
            BookkeepingCall("allocate", 1),
        )
    )

    def factory() -> ResourceCall:
        return root

    def declare(
        node: ResourceCall, allocator: SymbolAllocator
    ) -> RawCallees[ResourceCall]:
        return tuple(_resource_callees(node, None))

    example: ComponentExample[ResourceCall] = ComponentExample(
        factory, "native_rotations", ResourceSequenceCall
    )
    graph = resource_library_call_graph([example], declare)
    rotation = RotationCall("rx", None)
    phase = PhaseCall("t")
    assert graph == {root: {rotation: 2, phase: 1}, rotation: {}, phase: {}}


def test_native_profile_missing_capability_warns_and_keeps_leaf() -> None:
    from flagquantum.compiler._component_trace.resource_generalization import (
        ResourceCall,
    )
    from flagquantum.compiler._component_trace.resource_model import ComponentCost

    missing = ComponentCost("Missing", 1)

    def factory() -> ResourceCall:
        return missing

    def declare(node: ResourceCall, allocator: SymbolAllocator) -> Any:
        raise CapabilityError("missing model")

    example: ComponentExample[ResourceCall] = ComponentExample(
        factory, "missing", ComponentCost
    )
    with pytest.warns(UserWarning, match="lacks a call graph"):
        assert resource_library_call_graph([example], declare) == {missing: {}}
