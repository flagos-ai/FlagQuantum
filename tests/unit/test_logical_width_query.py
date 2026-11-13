"""Check recursive width visitation and connected lifetimes against locked source."""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Literal

import pytest
from sympy import Symbol

from flagquantum.compiler._component_trace.logical_width import query_logical_width
from flagquantum.compiler._component_trace.resource_generalization import ResourceCall
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.compiler._component_trace.width_schedule import (
    WidthConnection,
    WidthStep,
    estimate_connection_width,
)
from flagquantum.drawer.call_graph import export_resource_call_graph_dot

pytestmark = pytest.mark.unit


def no_calls(node: str) -> Iterable[tuple[str, SymbolicInt]]:
    return ()


def test_connected_width_retains_bystanders_and_sparse_cache() -> None:
    root = ComponentCost("root", 2)
    child = ComponentCost("child", 8)
    unused = ComponentCost("unused", 100)
    rows: dict[ResourceCall, dict[ResourceCall, SymbolicInt]] = {
        root: {unused: 1},
        child: {},
        unused: {},
    }
    cache: dict[ResourceCall, SymbolicInt] = {}

    def signature(node: ResourceCall) -> SymbolicInt:
        assert isinstance(node, ComponentCost)
        return node.qubits

    def connected(
        node: ResourceCall, visit: Callable[[ResourceCall], SymbolicInt]
    ) -> SymbolicInt | None:
        if node != root:
            return None
        return estimate_connection_width(
            [
                WidthStep("input"),
                WidthStep("component", visit(child)),
                WidthStep("output"),
            ],
            [
                WidthConnection(0, 1, 2),
                WidthConnection(1, 2, 2),
                WidthConnection(0, 2, 5),
            ],
        )

    assert (
        query_logical_width(
            root,
            signature_width=signature,
            callees=lambda node: rows[node].items(),
            decomposition_width=connected,
            cache=cache,
        )
        == 13
    )
    assert cache == {child: 8, root: 13}
    # The visual gate DAG is independent from decomposition-width visitation.
    dot = export_resource_call_graph_dot(
        root,
        logical_width=cache.get,
        declared_calls=rows,
        max_depth=0,
    )
    assert "13" in dot and "unused" not in dot
    assert unused not in cache


def test_fallback_visits_zero_repetitions_and_symbolic_widths() -> None:
    n = Symbol("n", positive=True, integer=True)
    rows: dict[str, tuple[tuple[str, SymbolicInt], ...]] = {
        "root": (("first", 0), ("second", 1000)),
        "first": (),
        "second": (),
    }
    cache: dict[str, SymbolicInt] = {}
    assert (
        query_logical_width(
            "root",
            signature_width={"root": 1, "first": n + 4, "second": n}.__getitem__,
            callees=rows.__getitem__,
            cache=cache,
        )
        == n + 4
    )
    assert list(cache) == ["first", "second", "root"]


def test_zero_decomposition_bypasses_signature_and_callees() -> None:
    def unexpected(node: str) -> Any:
        pytest.fail("A resolved decomposition must bypass fallback")

    assert (
        query_logical_width(
            "root",
            signature_width=unexpected,
            callees=unexpected,
            decomposition_width=lambda node, child: 0,
        )
        == 0
    )


def test_seed_static_and_generalizer_use_existing_cost_engine() -> None:
    cache: dict[str, SymbolicInt] = {"seed": 7}
    seen: list[str] = []

    def signature(node: str) -> SymbolicInt:
        seen.append(node)
        return 2

    assert (
        query_logical_width(
            "seed",
            cache=cache,
            static_costs={"seed": 9},
            signature_width=signature,
            callees=no_calls,
        )
        == 7
    )
    assert (
        query_logical_width(
            "static",
            cache=cache,
            static_costs={"static": 4},
            signature_width=signature,
            callees=no_calls,
        )
        == 4
    )
    assert (
        query_logical_width(
            "alias",
            cache=cache,
            generalizer=lambda node: "seed",
            signature_width=signature,
            callees=no_calls,
        )
        == 7
    )
    assert (
        query_logical_width(
            "omit",
            cache=cache,
            generalizer=lambda node: None,
            signature_width=signature,
            callees=no_calls,
        )
        == 0
    )
    assert (
        query_logical_width(
            "fresh",
            cache=cache,
            generalizer=[lambda node: node],
            signature_width=signature,
            callees=no_calls,
        )
        == 2
    )
    assert seen == ["fresh"] and "omit" not in cache and "alias" not in cache


@pytest.mark.parametrize("connected", [False, True])
def test_cycle_failure_and_retry(connected: bool) -> None:
    cache: dict[str, SymbolicInt] = {}
    calls = {"root": (("root", 1),)}
    with pytest.raises(ValueError, match="cycle"):
        query_logical_width(
            "root",
            signature_width=lambda node: 1,
            callees=calls.__getitem__,
            decomposition_width=(
                (lambda node, child: child(node)) if connected else None
            ),
            cache=cache,
        )
    assert cache == {}
    assert (
        query_logical_width(
            "root", signature_width=lambda node: 1, callees=no_calls, cache=cache
        )
        == 1
    )


@pytest.mark.parametrize("width", [-1, True])
@pytest.mark.parametrize("connected", [False, True])
def test_invalid_computed_width_is_rejected(width: int, connected: bool) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        query_logical_width(
            "root",
            signature_width=lambda node: width,
            callees=no_calls,
            decomposition_width=(lambda node, child: width) if connected else None,
        )


def test_unexpected_decomposition_error_propagates_without_fallback() -> None:
    def broken(node: str, child: Callable[[str], SymbolicInt]) -> SymbolicInt | None:
        raise RuntimeError("missing register connection")

    cache: dict[str, SymbolicInt] = {}
    with pytest.raises(RuntimeError, match="missing register"):
        query_logical_width(
            "root",
            signature_width=lambda node: 100,
            callees=no_calls,
            decomposition_width=broken,
            cache=cache,
        )
    assert cache == {}


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "case",
    [
        "interior",
        "symbolic",
        "independent",
        "serial",
        "many",
        "once",
        "abstracted",
        "multi_alloc",
        "and-positive",
        "and-negative",
        "swap",
        "symbolic-swap",
    ],
)
def test_full_recursive_cache_matches_unmodified_source(case: str) -> None:
    from qualtran import DecomposeNotImplementedError, DecomposeTypeError
    from qualtran._infra.binst_graph_iterators_test import MultiAlloc
    from qualtran._infra.quantum_graph import LeftDangle, RightDangle
    from qualtran.bloqs.bookkeeping import Allocate, Free
    from qualtran.bloqs.for_testing.interior_alloc import InteriorAlloc
    from qualtran.bloqs.for_testing.qubit_count_many_alloc import (
        TestManyAllocAbstracted,
        TestManyAllocMany,
        TestManyAllocOnce,
    )
    from qualtran.bloqs.for_testing.with_decomposition import (
        TestIndependentParallelCombo,
        TestSerialCombo,
    )
    from qualtran.bloqs.mcmt import MultiAnd
    from qualtran.bloqs.swap_network.cswap_approx import CSwapApprox
    from qualtran.resource_counting import QubitCount, get_cost_value
    from qualtran.resource_counting._call_graph import get_bloq_callee_counts

    reference = Path(__file__).resolve().parents[3] / "ftqc-reference-env"
    with (reference.parent / "FlagQuantum-Qualtran-Kickmix-移植进度.csv").open(
        encoding="utf-8-sig"
    ) as stream:
        blobs = {row["源文件"]: row["源对象"] for row in csv.DictReader(stream)}
    name = "qualtran/resource_counting/_qubit_counts.py"
    raw = (reference / name).read_bytes()
    assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blobs[name]
    n = Symbol("n", positive=True, integer=True)
    root: Any = {
        "interior": InteriorAlloc(10),
        "symbolic": InteriorAlloc(n),
        "independent": TestIndependentParallelCombo(),
        "serial": TestSerialCombo(),
        "many": TestManyAllocMany(10),
        "once": TestManyAllocOnce(10),
        "abstracted": TestManyAllocAbstracted(10),
        "multi_alloc": MultiAlloc(),
        "and-positive": MultiAnd(cvs=(1, 1, 1, 1)),
        "and-negative": MultiAnd(cvs=(0, 1, 0, 1)),
        "swap": CSwapApprox(3),
        "symbolic-swap": CSwapApprox(n),
    }[case]

    def connected(node: Any, visit: Callable[[Any], SymbolicInt]) -> SymbolicInt | None:
        try:
            graph = node.decompose_bloq()._binst_graph
        except (DecomposeNotImplementedError, DecomposeTypeError):
            return None
        indices = {instance: i for i, instance in enumerate(graph.nodes)}
        steps = []
        for instance in indices:
            if instance == LeftDangle:
                steps.append(WidthStep("input"))
            elif instance == RightDangle:
                steps.append(WidthStep("output"))
            else:
                kind: Literal["allocate", "free", "component"] = (
                    "allocate"
                    if instance.bloq_is(Allocate)
                    else "free" if instance.bloq_is(Free) else "component"
                )
                steps.append(WidthStep(kind, visit(instance.bloq)))
        return estimate_connection_width(
            steps,
            [
                WidthConnection(indices[start], indices[end], connection.shape)
                for start, end, data in graph.edges(data=True)
                for connection in data["cxns"]
            ],
        )

    expected_cache: dict[Any, SymbolicInt] = {}
    expected = get_cost_value(root, QubitCount(), costs_cache=expected_cache)
    actual_cache: dict[Any, SymbolicInt] = {}
    actual = query_logical_width(
        root,
        signature_width=lambda node: node.signature.n_qubits(),
        callees=get_bloq_callee_counts,
        decomposition_width=connected,
        cache=actual_cache,
    )
    assert actual == expected
    assert actual_cache == expected_cache
