"""Exercise generic cost query behavior against the pinned costing files."""

from collections.abc import Callable
from hashlib import sha1
from pathlib import Path

import pytest
from sympy import Symbol

from flagquantum.compiler._component_trace.cost_query import (
    CostMetric,
    query_component_cost,
    query_component_costs,
)
from flagquantum.compiler._component_trace.program_cost_query import query_program_cost
from flagquantum.core.value_math import SymbolicInt
from tests.unit.test_component_expansion import _hierarchy

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
REFERENCE = (
    Path(__file__).resolve().parents[3]
    / "ftqc-reference-env/qualtran/resource_counting"
)


def test_costing_frozen_files() -> None:
    pytest.importorskip("qualtran")
    for name, expected in (
        ("_costing.py", "4200b0acad3c04d1256b66d8f135ec1e48249cf5"),
        ("_costing_test.py", "9ea0f97d2b7e4673f948e223a39109ba415b1e43"),
    ):
        path = REFERENCE / name
        if not path.is_file():
            pytest.skip("Frozen costing files are not installed")
        data = path.read_bytes()
        assert sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == expected


def test_declared_metric_validates_static_values_and_supports_symbolic_rows() -> None:
    n = Symbol("n", positive=True, integer=True)
    seen: list[SymbolicInt] = []

    def validate(value: SymbolicInt) -> None:
        seen.append(value)
        if value == -1:
            raise ValueError("invalid static value")

    def compute(name: str, child: Callable[[str], SymbolicInt]) -> SymbolicInt:
        if name == "leaf":
            return -1
        return n * child("leaf")

    metric: CostMetric[str, SymbolicInt] = CostMetric(compute, 0, validate)
    leaf_only: CostMetric[str, int] = CostMetric(lambda _node, _child: 2, 0)
    table = query_component_costs("root", (metric, leaf_only))
    assert list(table) == ["leaf", "root"]
    assert table == {"leaf": {metric: -1}, "root": {metric: -n, leaf_only: 2}}
    assert query_component_cost("root", metric, cache={"leaf": -1}) == -n
    seed: dict[str, SymbolicInt] = {"leaf": 3}
    assert metric.cost_cache("root", cache=seed) is seed
    assert seed == {"leaf": 3, "root": 3 * n}
    assert not seen
    with pytest.raises(ValueError, match="invalid static value"):
        query_component_cost("root", metric, static_costs={"leaf": -1})
    assert seen == [-1]


@pytest.mark.parametrize("seeded", [False, True])
def test_program_composites_recompute_while_leaves_cache(seeded: bool) -> None:
    costs = pytest.importorskip("qualtran.resource_counting")
    from qualtran.bloqs.basic_gates import Hadamard

    program, source_root = _hierarchy(2)
    n = Symbol("n", positive=True, integer=True)
    source_leaf = Hadamard()

    def source_compute(
        _metric: object, node: object, child: Callable[[object], SymbolicInt]
    ) -> SymbolicInt:
        if node == source_leaf:
            return 2
        return n * sum(
            count * child(callee)
            for callee, count in costs.get_bloq_callee_counts(node)
        )

    source_metric = type(
        "SymbolicCost",
        (costs.CostKey,),
        {"compute": source_compute, "zero": lambda _metric: 0},
    )()
    src_cache = {source_root: 99, source_leaf: 3} if seeded else {}
    expected = costs.get_cost_value(source_root, source_metric, costs_cache=src_cache)
    calls = {"main": "level0", "level0": "h"}
    visited: list[str] = []

    def compute(node: str, child: Callable[[str], SymbolicInt]) -> SymbolicInt:
        visited.append(node)
        if node == "h":
            return 2
        callee = calls[node]
        return n * (child(callee) + child(callee))

    cache: dict[str, SymbolicInt] = {"main": 99, "h": 3} if seeded else {}
    actual = query_program_cost(program, CostMetric(compute, 0), cache=cache)
    assert actual == expected == (12 if seeded else 8) * n**2
    assert cache == ({"main": 99, "h": 3} if seeded else {"h": 2})
    assert visited.count("level0") == 2
    assert visited.count("h") == (0 if seeded else 1)
    assert set(src_cache) == ({source_root, source_leaf} if seeded else {source_leaf})


def test_program_composite_static_overrides_its_seed() -> None:
    pytest.importorskip("qualtran")
    program, _ = _hierarchy(1)
    validated: list[int] = []

    def compute(_node: str, _child: Callable[[str], int]) -> int:
        raise AssertionError("static query must not compute")

    metric = CostMetric(compute, 0, validated.append)
    cache = {"main": 99}
    assert (
        query_program_cost(program, metric, cache=cache, static_costs={"main": 7}) == 7
    )
    assert cache == {"main": 7}
    assert validated == [7]
    assert query_program_cost(program, metric, generalizer=lambda _: None) == 0
    with pytest.raises(ValueError, match="unknown typed Program definition"):
        query_program_cost(program, metric, generalizer=lambda _: "missing")
