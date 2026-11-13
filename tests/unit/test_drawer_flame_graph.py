"""Folded T-cost paths follow the locked source DFS and weight order."""

import csv
import importlib.util
from collections.abc import Callable, Mapping
from hashlib import sha1
from itertools import islice
from pathlib import Path
from typing import Any

import pytest

from flagquantum.compiler._component_trace.call_graph import modeled_call_graph
from flagquantum.compiler._component_trace.cost_query import (
    GATE_COST,
    query_component_cost,
)
from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from flagquantum.drawer.flame_graph import iter_flame_graph_data

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
REFERENCE = Path(__file__).resolve().parents[3] / "ftqc-reference-env"


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    with (REFERENCE.parent / "FlagQuantum-Qualtran-Kickmix-移植进度.csv").open(
        encoding="utf-8-sig"
    ) as stream:
        blobs = {row["源文件"]: row["源对象"] for row in csv.DictReader(stream)}
    for name in (
        "qualtran/drawing/flame_graph.py",
        "qualtran/drawing/flame_graph_test.py",
        "qualtran/resource_counting/_call_graph.py",
        "qualtran/resource_counting/t_counts_from_sigma.py",
    ):
        raw = (REFERENCE / name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blobs[name]


def native_reference_data(bloq: Any, **kwargs: Any) -> list[str]:
    """Translate source-only graph fixtures to owned integer keys at the boundary."""
    from qualtran.bloqs.basic_gates import TGate
    from qualtran.drawing.flame_graph import (
        _keep_if_small,
        _pretty_name,
        _t_counts_for_bloq,
    )
    from qualtran.resource_counting.generalizers import cirq_to_bloqs

    visible, _ = bloq.call_graph(
        keep=kwargs.pop("keep", _keep_if_small), generalizer=cirq_to_bloqs, **kwargs
    )
    full, _ = bloq.call_graph()
    keys = {node: i for i, node in enumerate(visible)}
    calls = {(keys[a], keys[b]): edge["n"] for a, b, edge in visible.edges(data=True)}
    totals = {keys[node]: int(_t_counts_for_bloq(node, full)) for node in visible}
    titles = {keys[node]: _pretty_name(node) for node in visible}
    t_gates = {keys[node] for node in visible if node in (TGate(), TGate().adjoint())}
    return list(
        iter_flame_graph_data(
            keys[bloq],
            calls,
            totals,
            title=titles.__getitem__,
            is_t_gate=t_gates.__contains__,
        )
    )


@pytest.mark.parametrize("width", [1, 2, 3, 10])
@pytest.mark.parametrize("depth", [None, 0, 1, 2])
def test_cswap_full_costs_survive_visible_depth_cutoff(
    width: int, depth: int | None
) -> None:
    from qualtran.bloqs.basic_gates.swap import CSwap
    from qualtran.drawing.flame_graph import get_flame_graph_data

    bloq = CSwap(width)
    assert native_reference_data(bloq, max_depth=depth) == get_flame_graph_data(
        bloq, max_depth=depth
    )


@pytest.mark.parametrize("controls", [3, 4, 6, 12])
@pytest.mark.parametrize("uncompute", [False, True])
@pytest.mark.parametrize("cutoff", [False, True])
def test_multiand_both_directions_and_t_leaf_collapse(
    controls: int, uncompute: bool, cutoff: bool
) -> None:
    from qualtran.bloqs.mcmt import MultiAnd
    from qualtran.drawing.flame_graph import get_flame_graph_data

    bloq = MultiAnd(tuple(i % 2 for i in range(controls)))
    if uncompute:
        bloq = bloq.adjoint()
    kwargs = {} if cutoff else {"keep": None}
    assert native_reference_data(bloq, **kwargs) == get_flame_graph_data(bloq, **kwargs)


@pytest.mark.parametrize("size", [3, 5, 12])
def test_rotation_synthesis_costs_are_supplied_from_full_reference_graph(
    size: int,
) -> None:
    from qualtran.bloqs.state_preparation import PrepareUniformSuperposition
    from qualtran.drawing.flame_graph import get_flame_graph_data

    bloq = PrepareUniformSuperposition(size)
    assert native_reference_data(bloq) == get_flame_graph_data(bloq)


@pytest.mark.parametrize(
    "name",
    [
        "test_get_flame_graph_data_cswap",
        "test_get_flame_graph_data_multi_and",
        "test_get_flame_graph_data_prep_uniform",
    ],
)
@pytest.mark.parametrize("replace_producer", [False, True])
def test_original_source_assertions_execute_unchanged(
    monkeypatch: pytest.MonkeyPatch, name: str, replace_producer: bool
) -> None:
    spec = importlib.util.spec_from_file_location(
        "locked_flame_graph_tests", REFERENCE / "qualtran/drawing/flame_graph_test.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if replace_producer:
        monkeypatch.setattr(module, "get_flame_graph_data", native_reference_data)
    original: Callable[[], None] = getattr(module, name)
    original()


def test_shared_dag_repeats_whole_subtrees_in_edge_order() -> None:
    import networkx as nx
    from qualtran.bloqs.basic_gates import TGate
    from qualtran.bloqs.mcmt import And, MultiAnd
    from qualtran.drawing.flame_graph import _populate_flame_graph_data

    root, left, right, leaf = MultiAnd((1,) * 5), MultiAnd((1,) * 3), And(), TGate()
    graph = nx.DiGraph()
    for a, b, count in (
        (root, left, 2),
        (root, right, 3),
        (left, right, 4),
        (right, leaf, 4),
    ):
        graph.add_edge(a, b, n=count)
    # These metadata nodes need only provide stable titles and T-cost terminal leaves.
    # The source helper reads the supplied graph, without decomposing the fixtures.
    keys = {node: i for i, node in enumerate(graph)}
    from qualtran.drawing.flame_graph import _pretty_name, _t_counts_for_bloq

    totals = {keys[node]: int(_t_counts_for_bloq(node, graph)) for node in graph}
    calls = {(keys[a], keys[b]): edge["n"] for a, b, edge in graph.edges(data=True)}
    titles = {keys[node]: _pretty_name(node) for node in graph}
    actual = list(
        iter_flame_graph_data(
            keys[root],
            calls,
            totals,
            title=titles.__getitem__,
            is_t_gate=lambda key: key == keys[leaf],
        )
    )
    assert actual == _populate_flame_graph_data(root, graph, graph, [])
    assert len(actual) == 11
    assert sum(int(row.split("\t")[1]) for row in actual) == totals[keys[root]]


def test_native_component_model_and_full_cost_cache_are_reused() -> None:
    shared = ComponentCost("And", 3, GateCost(and_bloq=1))
    branch = ComponentCost("Branch", 3, calls=(CostCall(shared, 2),))
    root = ComponentCost("Root", 3, calls=(CostCall(branch, 3), CostCall(shared, 5)))
    costs: dict[ComponentCost, GateCost] = {}
    query_component_cost(root, GATE_COST, cache=costs)
    totals = {
        node: gates.total_t_count(ts_per_cswap=7) for node, gates in costs.items()
    }
    full = modeled_call_graph(root)
    rows = list(
        iter_flame_graph_data(
            root, full.direct_calls, totals, title=lambda node: node.name
        )
    )
    assert rows[:6] == ["Root(T:44);Branch(T:8);And(T:4)\t4"] * 6
    assert rows[6:] == ["Root(T:44);And(T:4)\t4"] * 5
    limited = modeled_call_graph(root, max_depth=1)
    assert (
        list(
            iter_flame_graph_data(
                root, limited.direct_calls, totals, title=lambda node: node.name
            )
        )
        == [
            "Root(T:44);Branch(T:8)\t8",
        ]
        * 3
        + ["Root(T:44);And(T:4)\t4"] * 5
    )
    assert sum(int(row.split("\t")[1]) for row in rows) == 44


@pytest.mark.parametrize("count", [0, 1, 7])
def test_terminal_root_includes_zero_count(count: int) -> None:
    assert list(iter_flame_graph_data("root", {}, {"root": count})) == [
        f"root(T:{count})\t{count}"
    ]


def test_zero_cost_children_omitted_and_zero_edge_retained() -> None:
    edges = {("r", "free"): 2, ("r", "paid"): 0}
    assert list(iter_flame_graph_data("r", edges, {"r": 0, "free": 0, "paid": 4})) == []
    assert list(
        iter_flame_graph_data("r", {("r", "free"): 2}, {"r": 0, "free": 0})
    ) == ["r(T:0)\t0"]


def test_t_adjoint_leaf_collapse_is_explicit() -> None:
    edges = {("r", "tdg"): 4}
    totals = {"r": 4, "tdg": 1}
    assert list(
        iter_flame_graph_data("r", edges, totals, is_t_gate=lambda key: key == "tdg")
    ) == ["r(T:4)\t4"]
    assert list(iter_flame_graph_data("r", edges, totals)) == ["r(T:4);tdg(T:1)\t1"] * 4


def test_iterator_memory_is_independent_of_repetition_count() -> None:
    edges = {("r", "a"): 10**12}
    totals = {"r": 4 * 10**12, "a": 4}
    iterator = iter_flame_graph_data("r", edges, totals)
    assert list(islice(iterator, 3)) == ["r(T:4000000000000);a(T:4)\t4"] * 3
    iterator.close()
    assert edges == {("r", "a"): 10**12}
    assert totals == {"r": 4 * 10**12, "a": 4}


@pytest.mark.parametrize("bad", [-1, True, 1.5, "2", None])
@pytest.mark.parametrize("target", ["multiplicity", "cost"])
def test_nonconcrete_and_negative_counts_fail_before_output(
    bad: Any, target: str
) -> None:
    edges = {("r", "a"): bad if target == "multiplicity" else 1}
    costs = {"r": 4, "a": bad if target == "cost" else 4}
    with pytest.raises(ValueError, match="nonnegative integers"):
        next(iter_flame_graph_data("r", edges, costs))


@pytest.mark.parametrize("label", ["a;b", "a\tb", "a\nb", "a\rb"])
def test_frame_separators_are_rejected_before_output(label: str) -> None:
    with pytest.raises(ValueError, match="frame separators"):
        next(iter_flame_graph_data("r", {}, {"r": 4}, title=lambda node: label))


def test_missing_cost_and_cycle_fail_without_partial_output() -> None:
    with pytest.raises(ValueError, match="T costs"):
        next(iter_flame_graph_data("r", {("r", "a"): 1}, {"r": 4}))
    from graphlib import CycleError

    with pytest.raises(CycleError):
        next(iter_flame_graph_data("r", {("r", "r"): 1}, {"r": 4}))


def test_title_error_propagates_and_retry_preserves_inputs() -> None:
    error = RuntimeError("title unavailable")

    def fail(node: str) -> str:
        raise error

    with pytest.raises(RuntimeError) as caught:
        next(iter_flame_graph_data("r", {}, {"r": 1}, title=fail))
    assert caught.value is error
    assert list(iter_flame_graph_data("r", {}, {"r": 1})) == ["r(T:1)\t1"]


def test_native_precision_sensitive_terminal_owner_supplies_counts() -> None:
    import networkx as nx
    from qualtran.bloqs.basic_gates import Rz, TGate, TwoBitCSwap
    from qualtran.drawing.flame_graph import _populate_flame_graph_data

    from flagquantum.compiler._component_trace.call_graph import symbolic_call_graph
    from flagquantum.compiler._component_trace.resource_generalization import (
        FixedTCall,
        PhaseCall,
        ResourceCall,
        RotationCall,
    )
    from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
    from flagquantum.compiler._component_trace.terminal_t_cost import (
        estimate_terminal_t_cost,
    )

    root = ComponentCost("Root", 3)
    swap, rotation, adjoint_t = (
        FixedTCall("fredkin"),
        RotationCall("rz", 0.25, 1e-4),
        PhaseCall("t", True),
    )
    terminals: dict[ResourceCall, int] = {swap: 2, rotation: 3, adjoint_t: 5}
    declarations: dict[ResourceCall, Mapping[ResourceCall, SymbolicInt]] = {
        root: terminals
    }
    graph = symbolic_call_graph(root, declarations)
    totals = {node: int(estimate_terminal_t_cost({node: 1})) for node in terminals}
    totals[root] = int(estimate_terminal_t_cost(graph.leaf_counts))
    titles = {root: "Root", swap: "TwoBitCSwap", rotation: "Rz(0.25)", adjoint_t: "T†"}
    assert totals[swap] == 7 and totals[rotation] == 25 and totals[root] == 94
    calls = {(a, b): int(count) for (a, b), count in graph.direct_calls.items()}
    actual = list(
        iter_flame_graph_data(
            root,
            calls,
            totals,
            title=titles.__getitem__,
            is_t_gate=lambda node: node == adjoint_t,
        )
    )
    source = nx.DiGraph()
    for terminal, count in (
        (TwoBitCSwap(), 2),
        (Rz(0.25, eps=1e-4), 3),
        (TGate().adjoint(), 5),
    ):
        # A distinct root avoids a self-cycle while preserving the declared mixture.
        source.add_edge("root", terminal, n=count)
    from attrs import frozen
    from qualtran import Bloq, Signature

    source_root = frozen(
        type("Root", (Bloq,), {"signature": property(lambda self: Signature(()))})
    )()
    source = nx.relabel_nodes(source, {"root": source_root})
    expected = _populate_flame_graph_data(source_root, source, source, [])
    assert actual == expected
    assert sum(int(row.split("\t")[1]) for row in actual) == 94


@pytest.mark.parametrize("collapse", [False, True])
def test_source_retains_unequal_parent_and_child_totals(collapse: bool) -> None:
    import networkx as nx
    from qualtran.bloqs.basic_gates import TGate, TwoBitCSwap
    from qualtran.bloqs.mcmt import And
    from qualtran.drawing import flame_graph

    root, child = And(), TGate() if collapse else TwoBitCSwap()
    graph = nx.DiGraph()
    graph.add_edge(root, child, n=4 if collapse else 1)
    full = nx.DiGraph()
    full.add_edge(root, child, n=9 if collapse else 2)
    expected = flame_graph._populate_flame_graph_data(root, graph, full, [])
    child_name = "T" if collapse else "TwoBitCSwap"
    actual = list(
        iter_flame_graph_data(
            "And",
            {("And", child_name): 4 if collapse else 1},
            {"And": 9 if collapse else 14, child_name: 1 if collapse else 7},
            is_t_gate=lambda node: node == "T",
        )
    )
    assert actual == expected
    assert actual == (
        ["And(T:9)\t9"] if collapse else ["And(T:14);TwoBitCSwap(T:7)\t7"]
    )


def test_qft_data_matches_actual_source_with_original_failure_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qualtran.bloqs.qft.qft_text_book import QFTTextBook
    from qualtran.drawing.flame_graph import get_flame_graph_data

    bloq = QFTTextBook(5)
    assert native_reference_data(bloq) == get_flame_graph_data(bloq)
    spec = importlib.util.spec_from_file_location(
        "locked_qft_flame_test", REFERENCE / "qualtran/drawing/flame_graph_test.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The locked expectation fails for both producers. Preserve the assertion.
    with pytest.raises(AssertionError):
        module.test_get_flame_graph_data_qft_textbook()
    monkeypatch.setattr(module, "get_flame_graph_data", native_reference_data)
    with pytest.raises(AssertionError):
        module.test_get_flame_graph_data_qft_textbook()
