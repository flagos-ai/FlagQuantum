"""Frozen Hamming tree liveness, generic adjoint and full cache conformance."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, get_args

import pytest
from sympy import Integer, Symbol

from flagquantum.compiler._component_trace.arithmetic_call import ArithmeticCall
from flagquantum.compiler._component_trace.resource_generalization import (
    BookkeepingCall,
    CliffordCall,
    ConjunctionCall,
    ResourceCall,
)
from flagquantum.compiler._component_trace.resource_width import (
    query_resource_logical_width,
)
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.compiler.registers.arithmetic.arithmetic_declarations import (
    arithmetic_call_graph,
)
from flagquantum.compiler.registers.bits.hamming_weight_width import (
    hamming_weight_width,
)
from flagquantum.drawer.call_graph import (
    GateAggregation,
    export_resource_call_graph_dot,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
n = Symbol("n", positive=True, integer=True)
p = Symbol("p", nonnegative=True, integer=True)
WIDTHS = (2, 3, 4, 5, 6, 8, 16, Integer(0), Integer(1), Integer(3), n, p, n + 1)


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    reference = Path(__file__).resolve().parents[3] / "ftqc-reference-env"
    with (reference.parent / "FlagQuantum-Qualtran-Kickmix-移植进度.csv").open(
        encoding="utf-8-sig"
    ) as stream:
        blobs = {row["源文件"]: row["源对象"] for row in csv.DictReader(stream)}
    for name in (
        "qualtran/bloqs/arithmetic/hamming_weight.py",
        "qualtran/_infra/adjoint.py",
        "qualtran/_infra/composite_bloq.py",
        "qualtran/_infra/binst_graph_iterators.py",
        "qualtran/resource_counting/_qubit_counts.py",
    ):
        raw = (reference / name).read_bytes()
        assert (
            hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blobs[name]
        )


def source_for(call: ArithmeticCall) -> Any:
    from qualtran.bloqs.arithmetic.hamming_weight import HammingWeightCompute

    source = HammingWeightCompute(call.widths[0])
    return source.adjoint() if call.adjoint else source


def source_key(source: Any, root: ArithmeticCall) -> ResourceCall:
    from qualtran.bloqs.basic_gates import CNOT
    from qualtran.bloqs.bookkeeping import Allocate, Free, Join, Split
    from qualtran.bloqs.mcmt import And

    if source == source_for(root):
        return root
    if isinstance(source, (Split, Join, Allocate, Free)):
        opcodes: dict[type[Any], Literal["split", "join", "allocate", "free"]] = {
            Split: "split",
            Join: "join",
            Allocate: "allocate",
            Free: "free",
        }
        opcode = opcodes[type(source)]
        return BookkeepingCall(opcode, source.dtype.num_qubits)
    if isinstance(source, And):
        return ConjunctionCall((1, 1), source.uncompute)
    assert isinstance(source, CNOT)
    return CliffordCall("cx", 2)


def compare(
    call: ArithmeticCall, seeds: tuple[Any, ...] = ()
) -> dict[ResourceCall, SymbolicInt]:
    from qualtran.resource_counting import QubitCount, get_cost_value

    expected: dict[Any, SymbolicInt] = dict.fromkeys(seeds, 30)
    actual = {source_key(key, call): value for key, value in expected.items()}
    assert query_resource_logical_width(call, cache=actual) == get_cost_value(
        source_for(call), QubitCount(), costs_cache=expected
    )
    normalized = {source_key(key, call): value for key, value in expected.items()}
    assert actual == normalized
    assert list(actual) == list(normalized)
    return actual


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("adjoint", [False, True])
def test_full_cache_and_order(width: SymbolicInt, adjoint: bool) -> None:
    compare(ArithmeticCall("hamming_weight", (width,), adjoint=adjoint))


@pytest.mark.parametrize("width", [2, 3, 6, 8, Integer(3), n])
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize(
    "seed_name", ["cx", "and", "split", "join", "allocate", "free"]
)
def test_seeded_child_peaks(width: SymbolicInt, adjoint: bool, seed_name: str) -> None:
    from qualtran import QAny, QUInt
    from qualtran.bloqs.basic_gates import CNOT
    from qualtran.bloqs.bookkeeping import Allocate, Free, Join, Split
    from qualtran.bloqs.mcmt import And

    call = ArithmeticCall("hamming_weight", (width,), adjoint=adjoint)
    seeds = {
        "cx": CNOT(),
        "and": And(uncompute=adjoint),
        "split": Split(QUInt(width if type(width) is int else 3)),
        "join": Join(QUInt(width if type(width) is int else 3)),
        "allocate": Allocate(QAny(1)),
        "free": Free(QAny(1)),
    }
    compare(call, (seeds[seed_name],))


@pytest.mark.parametrize("width", [3, 6, Integer(3), n])
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("depth", [0, 1, None])
@pytest.mark.parametrize("aggregation", get_args(GateAggregation))
def test_full_dot(
    width: SymbolicInt, adjoint: bool, depth: int | None, aggregation: GateAggregation
) -> None:
    call = ArithmeticCall("hamming_weight", (width,), adjoint=adjoint)
    widths = compare(call)
    declarations = arithmetic_call_graph(call)
    assert export_resource_call_graph_dot(
        call, declared_calls=declarations, max_depth=depth, aggregation=aggregation
    ) == export_resource_call_graph_dot(
        call,
        declared_calls=declarations,
        logical_width=widths.get,
        max_depth=depth,
        aggregation=aggregation,
    )


@pytest.mark.parametrize("width", [0, 1])
@pytest.mark.parametrize("adjoint", [False, True])
def test_source_failure_empty_cache_and_retry(width: int, adjoint: bool) -> None:
    from qualtran.resource_counting import QubitCount, get_cost_value

    call = ArithmeticCall("hamming_weight", (width,), adjoint=adjoint)
    source_cache: dict[Any, SymbolicInt] = {}
    with pytest.raises(RuntimeError):
        get_cost_value(source_for(call), QubitCount(), costs_cache=source_cache)
    cache: dict[ResourceCall, SymbolicInt] = {}
    with pytest.raises(ValueError, match="at least two"):
        query_resource_logical_width(call, cache=cache)
    assert cache == source_cache == {}
    assert query_resource_logical_width(replace(call, widths=(3,)), cache=cache) == 6


def test_owner_and_decomposition_priority() -> None:
    with pytest.raises(NotImplementedError):
        hamming_weight_width(ArithmeticCall("xor", (3,)), lambda call: 0)
    call = ArithmeticCall("hamming_weight", (3,))
    cache: dict[ResourceCall, SymbolicInt] = {}
    assert (
        query_resource_logical_width(call, declared_calls={call: {}}, cache=cache) == 6
    )
    assert ConjunctionCall((1, 1)) in cache


@pytest.mark.parametrize("width", [2, 3, 4, 5, 6, 8, 16])
@pytest.mark.parametrize("adjoint", [False, True])
def test_complete_connection_graph(
    width: int, adjoint: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    import networkx as nx
    from qualtran import LeftDangle, RightDangle

    from flagquantum.compiler._component_trace.width_schedule import (
        WidthConnection,
        WidthStep,
    )
    from flagquantum.compiler.registers.bits import hamming_weight_width as owner

    call = ArithmeticCall("hamming_weight", (width,), adjoint=adjoint)
    expected = nx.DiGraph()
    cbloq = source_for(call).decompose_bloq()
    expected.add_node(LeftDangle, key="input")
    expected.add_node(RightDangle, key="output")
    for instance in cbloq.bloq_instances:
        expected.add_node(instance, key=source_key(instance.bloq, call))
    for edge in cbloq.connections:
        left, right = edge.left.binst, edge.right.binst
        previous = expected.get_edge_data(left, right, {}).get("width", 0)
        expected.add_edge(left, right, width=previous + edge.left.reg.dtype.num_qubits)
    markers: dict[ResourceCall, int] = {}

    def child(key: ResourceCall) -> int:
        return markers.setdefault(key, len(markers) + 100)

    def inspect(steps: list[WidthStep], edges: list[WidthConnection]) -> int:
        actual = nx.DiGraph()
        identities = {marker: key for key, marker in markers.items()}
        for index, step in enumerate(steps):
            key = (
                step.kind
                if step.kind in {"input", "output"}
                else identities[step.qubits]
            )
            actual.add_node(index, key=key)
        for edge in edges:
            previous = actual.get_edge_data(edge.source, edge.target, {}).get(
                "width", 0
            )
            actual.add_edge(edge.source, edge.target, width=previous + edge.qubits)
        assert nx.is_isomorphic(
            actual,
            expected,
            node_match=lambda a, b: a["key"] == b["key"],
            edge_match=lambda a, b: a["width"] == b["width"],
        )
        return 0

    monkeypatch.setattr(owner, "estimate_connection_width", inspect)
    assert owner.hamming_weight_width(call, child) == 0
