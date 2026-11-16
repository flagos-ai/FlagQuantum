"""Close the frozen GateCounts algebra and static-admission cases."""

from hashlib import sha1
from pathlib import Path
from typing import cast

import pytest
from sympy import Float, Symbol

from flagquantum.compiler._component_trace.call_graph import symbolic_call_graph
from flagquantum.compiler._component_trace.qec_gate_cost import (
    QECGateCost,
    query_qec_gate_cost,
)
from flagquantum.compiler._component_trace.resource_generalization import (
    FixedTCall,
    PhaseCall,
    ResourceCall,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.compiler._component_trace.selected_call_count import (
    count_selected_calls,
)
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_gate_count_frozen_files() -> None:
    root = (
        Path(__file__).resolve().parents[3]
        / "ftqc-reference-env/qualtran/resource_counting"
    )
    for name, expected in (
        ("_bloq_counts.py", "226e6db045513112bd3e5ff5aa79d2cb8c5bf6d4"),
        ("_bloq_counts_test.py", "2e59e69391050c90bd9a9f207ca38fbd735edc07"),
    ):
        path = root / name
        if not path.is_file():
            pytest.skip("Frozen gate-count files are not installed")
        data = path.read_bytes()
        assert sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == expected


def test_gate_counts_algebra_and_display_match_concrete_and_symbolic_source() -> None:
    source = pytest.importorskip("qualtran.resource_counting")
    n = Symbol("n", positive=True, integer=True)
    left = QECGateCost(t=2, rotation=n, measurement=1)
    right = QECGateCost(toffoli=7, cswap=1, and_bloq=3, clifford=4)
    ref_left = source.GateCounts(**left.asdict())
    ref_right = source.GateCounts(**right.asdict())
    assert n * left == left * n
    for native, reference in (
        (left + right, ref_left + ref_right),
        (3 * right, 3 * ref_right),
        (left * n + right, ref_left * n + ref_right),
        (QECGateCost(clifford=Float(0.0)), source.GateCounts(clifford=Float(0.0))),
        (QECGateCost(), source.GateCounts()),
    ):
        assert native.asdict() == reference.asdict()
        assert str(native) == str(reference)
    with pytest.raises(TypeError, match="only be added"):
        _ = left + cast(QECGateCost, 1)
    with pytest.raises(TypeError):
        _ = ref_left + 1


def test_static_types_are_checked_after_cache_and_generalization() -> None:
    source = pytest.importorskip("qualtran.resource_counting")
    from qualtran.bloqs.for_testing.costing import CostingBloq

    metric = source.QECGatesCost()
    ref = CostingBloq("leaf", 1, static_costs=[(metric, 7)])
    with pytest.raises(TypeError, match="values should be"):
        source.get_cost_value(ref, metric)
    gate = PhaseCall("t")
    original = PhaseCall("t", adjoint=True)
    invalid: dict[ResourceCall, QECGateCost] = {gate: cast(QECGateCost, 7)}
    with pytest.raises(TypeError, match="Static gate counts require"):
        query_qec_gate_cost(gate, static_costs=invalid)
    assert (
        query_qec_gate_cost(original, static_costs=invalid, generalizer=lambda _: None)
        == QECGateCost()
    )
    seed: dict[ResourceCall, QECGateCost] = {gate: QECGateCost(t=5)}
    assert query_qec_gate_cost(
        original, cache=seed, static_costs=invalid, generalizer=lambda _: gate
    ) == QECGateCost(t=5)
    source_value = source.GateCounts(t=5)
    assert (
        source.get_cost_value(ref, metric, costs_cache={ref: source_value})
        == source_value
    )


def test_selected_graph_leaves_match_frozen_gate_boundary() -> None:
    source = pytest.importorskip("qualtran.resource_counting")
    from qualtran.bloqs.basic_gates import TGate, Toffoli
    from qualtran.bloqs.for_testing.costing import CostingBloq

    n = Symbol("n", positive=True, integer=True)
    root = ComponentCost("root", 3)
    child = ComponentCost("child", 3)
    t, tof = PhaseCall("t"), FixedTCall("toffoli")
    declared: dict[ResourceCall, dict[ResourceCall, SymbolicInt]] = {
        root: {child: 2, t: n},
        child: {t: 1, tof: 3},
    }
    graph = symbolic_call_graph(cast(ResourceCall, root), declared)
    native = count_selected_calls(root, graph.leaf_counts, declared_calls=declared)
    ref_child = CostingBloq("child", 3, [(TGate(), 1), (Toffoli(), 3)])
    ref = CostingBloq("root", 3, [(ref_child, 2), (TGate(), n)])
    ref_graph, _ = ref.call_graph()
    metric = source.BloqCount.for_call_graph_leaf_bloqs(ref_graph)
    assert (
        {TGate(): native[t], Toffoli(): native[tof]}
        == source.get_cost_value(ref, metric)
        == {TGate(): n + 2, Toffoli(): 6}
    )
