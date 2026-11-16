"""Verify the pinned success-probability rule, including declared repetitions."""

from hashlib import sha1
from importlib import import_module
from pathlib import Path

import pytest
from sympy import Expr, Symbol

from flagquantum.compiler._component_trace.call_graph import RawCallees, SymbolAllocator
from flagquantum.compiler._component_trace.cost_query import (
    SUCCESS_PROBABILITY,
    query_component_cost,
    query_declared_success_probability,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost, CostCall
from flagquantum.core.value_math import SymbolicReal
from flagquantum.errors import CapabilityError

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
REFERENCE = (
    Path(__file__).resolve().parents[3]
    / "ftqc-reference-env/qualtran/resource_counting"
)
BLOBS = {
    "_success_prob.py": "4aab783bdd90e7d48af85f414ef70c908c948588",
    "_success_prob_test.py": "3c4eeb552d7a473c97e00e9b5a236e34b891bd85",
}


def test_frozen_success_metric_and_original_test_file() -> None:
    pytest.importorskip("qualtran")
    for name, expected in BLOBS.items():
        path = REFERENCE / name
        if not path.is_file():
            pytest.skip("Frozen success-probability assets are not installed")
        data = path.read_bytes()
        assert sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == expected


def test_original_coin_flip_result_and_full_metric_cache() -> None:
    costs = pytest.importorskip("qualtran.resource_counting")
    factory = import_module("qualtran.bloqs.for_testing.costing").CostingBloq
    metric = costs.SuccessProb()
    assert metric.zero() == SUCCESS_PROBABILITY.zero == 1.0
    assert str(metric) == "success prob"
    source_coin = factory("CoinFlip", 1, static_costs=[(metric, 0.5)])
    source_root = factory("Algo", 0, callees=[(source_coin, 4)])
    expected_cache = {
        node.name: value
        for node, value in costs.get_cost_cache(source_root, metric).items()
    }

    cache: dict[str, SymbolicReal] = {}
    assert (
        query_declared_success_probability(
            "Algo",
            {"Algo": {"CoinFlip": 4}},
            cache=cache,
            static_costs={"CoinFlip": 0.5},
        )
        == costs.get_cost_value(source_root, metric)
        == 0.0625
    )
    assert cache == expected_cache == {"CoinFlip": 0.5, "Algo": 0.0625}

    coin = ComponentCost("CoinFlip", 1, success_probability=0.5)
    root = ComponentCost("Algo", 0, calls=(CostCall(coin, 4),))
    native_cache: dict[ComponentCost, float] = {}
    assert query_component_cost(root, SUCCESS_PROBABILITY, cache=native_cache) == 0.0625
    assert {node.name: value for node, value in native_cache.items()} == cache


@pytest.mark.parametrize("override", [None, 0.0, 0.75])
def test_symbolic_repetition_and_zero_calls_match_source_cache(
    override: float | None,
) -> None:
    costs = pytest.importorskip("qualtran.resource_counting")
    factory = import_module("qualtran.bloqs.for_testing.costing").CostingBloq
    metric = costs.SuccessProb()
    n = Symbol("n", positive=True, integer=True)
    source_coin = factory("coin", 1, static_costs=[(metric, 0.5)])
    source_failure = factory("failure", 0, static_costs=[(metric, 0.0)])
    source_identity = factory("identity", 0)
    source_calls = [(source_coin, n), (source_failure, 0), (source_identity, 1)]
    source_root = factory(
        "root",
        0,
        callees=source_calls,
        static_costs=[] if override is None else [(metric, override)],
    )
    static = {"coin": 0.5, "failure": 0.0}
    if override is not None:
        static["root"] = override
    visited = []
    symbols = SymbolAllocator()

    def declared(node: str, allocator: SymbolAllocator) -> RawCallees[str]:
        assert allocator is symbols
        visited.append(node)
        if node == "root":
            return (("coin", n), ("failure", 0), ("identity", 1))
        raise CapabilityError("Leaf has no child declaration")

    cache: dict[str, SymbolicReal] = {}
    value = query_declared_success_probability(
        "root", declared, cache=cache, static_costs=static, allocator=symbols
    )
    assert value == costs.get_cost_value(source_root, metric)
    source_cache = costs.get_cost_cache(source_root, metric)
    assert list(cache.items()) == [
        (node.name, value) for node, value in source_cache.items()
    ]
    if override is None:
        assert isinstance(value, Expr)
        assert float(value.subs(n, 4)) == 0.0625
        assert visited == ["root", "identity"]
    else:
        assert not visited


def test_generalization_seeded_cache_and_cycle_keep_shared_cost_semantics() -> None:
    source = pytest.importorskip("qualtran.resource_counting")
    factory = import_module("qualtran.bloqs.for_testing.costing").CostingBloq
    metric = source.SuccessProb()
    left, right = factory("left", 1), factory("right", 1)
    canonical = factory("coin", 1, static_costs=[(metric, 0.5)])
    root = factory("root", 0, callees=[(left, 1), (right, 3)])
    reference = source.get_cost_cache(
        root,
        metric,
        generalizer=lambda node: canonical if node in {left, right} else node,
    )
    cache: dict[str, SymbolicReal] = {}
    assert (
        query_declared_success_probability(
            "root",
            {"root": {"left": 1, "right": 3}},
            cache=cache,
            static_costs={"coin": 0.5},
            generalizer=lambda node: "coin" if node in {"left", "right"} else node,
        )
        == reference[root]
    )
    assert cache == {node.name: value for node, value in reference.items()}
    with pytest.raises(ValueError, match="cycle"):
        query_declared_success_probability("root", {"root": {"root": 1}})
    assert (
        query_declared_success_probability(
            "root",
            {"root": {"root": 1}},
            cache={"root": 0.0},
            static_costs={"root": 0.5},
        )
        == 0.0
    )
    assert (
        query_declared_success_probability(
            "root", {"root": {"root": 1}}, generalizer=lambda _: None
        )
        == 1.0
    )
