"""Check selected resource call counts against the locked source BloqCount."""

import pytest

from flagquantum.compiler._component_trace.call_graph import symbolic_call_graph
from flagquantum.compiler._component_trace.resource_generalization import (
    CliffordCall,
    DeclaredResourceCalls,
    FixedTCall,
    PhaseCall,
    generalize_rotation,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost, CostCall
from flagquantum.compiler._component_trace.selected_call_count import (
    count_selected_calls,
    resource_gateset,
)

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("selection", ["t", "t+tof", "t+tof+cswap", "leaf", "top"])
def test_selected_boundaries_match_source(selection: str) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates import Hadamard, TGate, Toffoli, TwoBitCSwap
    from qualtran.bloqs.for_testing.costing import (
        CostingBloq,
        make_example_costing_bloqs,
    )
    from qualtran.resource_counting import BloqCount, get_cost_value

    root, one, two = (ComponentCost(name, 10) for name in ("Algo", "Func1", "Func2"))
    t, tdg = resource_gateset("t")
    tof = FixedTCall("toffoli")
    fredkin = FixedTCall("fredkin")
    h = CliffordCall("h", 1)
    calls: DeclaredResourceCalls = {
        root: {one: 2, two: 1, fredkin: 3},
        one: {t: 10, tdg: 10, h: 10},
        two: {tof: 100},
    }
    original = make_example_costing_bloqs()
    source = CostingBloq("Algo", 10, [*original.callees, (TwoBitCSwap(), 3)])
    native_to_source = {
        root: source,
        one: original.callees[0][0],
        two: original.callees[1][0],
        t: TGate(),
        tdg: TGate().adjoint(),
        tof: Toffoli(),
        fredkin: TwoBitCSwap(),
        h: Hadamard(),
    }
    if selection == "top":
        selected = tuple(calls[root])
        source_cost = BloqCount(tuple(node for node, _ in source.callees), "top")
    elif selection == "leaf":
        selected = tuple(symbolic_call_graph(root, calls).leaf_counts)
        source_cost = BloqCount.for_call_graph_leaf_bloqs(source.call_graph()[0])
        assert {native_to_source[node] for node in selected} == set(
            source_cost.gateset_bloqs
        )
    else:
        selected = resource_gateset(selection)
        source_cost = BloqCount.for_gateset(selection)
    actual = count_selected_calls(root, selected, declared_calls=calls)
    assert {native_to_source[node]: count for node, count in actual.items()} == (
        get_cost_value(source, source_cost)
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("omit_right", [False, True])
def test_symbolic_shared_counts_and_ordered_rewrites_match_source(
    omit_right: bool,
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates import TGate, Toffoli
    from qualtran.bloqs.for_testing.costing import CostingBloq
    from qualtran.resource_counting import BloqCount, get_cost_value
    from qualtran.resource_counting.generalizers import generalize_rotation_angle
    from sympy import Symbol

    n = Symbol("n", nonnegative=True, integer=True)
    root = ComponentCost("root", 3)
    left = ComponentCost("left", 3)
    right = ComponentCost("right", 3)
    shared = ComponentCost("shared", 3)
    t = PhaseCall("t")
    tof = FixedTCall("toffoli")
    calls: DeclaredResourceCalls = {
        root: {left: n, right: 2, tof: 0},
        left: {shared: 3},
        right: {shared: 4, t: 99},
        shared: {PhaseCall("t", True): 5, t: 1},
    }
    source_shared = CostingBloq("shared", 3, [(TGate().adjoint(), 5), (TGate(), 1)])
    source_left = CostingBloq("left", 3, [(source_shared, 3)])
    source_right = CostingBloq("right", 3, [(source_shared, 4), (TGate(), 99)])
    source_root = CostingBloq(
        "root", 3, [(source_left, n), (source_right, 2), (Toffoli(), 0)]
    )
    actual = count_selected_calls(
        root,
        (t, tof),
        declared_calls=calls,
        generalizer=(
            lambda node: None if omit_right and node == right else node,
            generalize_rotation,
        ),
    )
    expected = get_cost_value(
        source_root,
        BloqCount.for_gateset("t+tof"),
        generalizer=(
            lambda node: None if omit_right and node == source_right else node,
            generalize_rotation_angle,
        ),
    )
    assert actual == {t: expected[TGate()], tof: expected[Toffoli()]}
    assert actual == {t: 18 * n + (0 if omit_right else 246), tof: 0}
    assert (
        count_selected_calls(
            root, (t,), declared_calls=calls, generalizer=lambda _: None
        )
        == {}
    )


def test_component_declarations_stop_at_selected_interior() -> None:
    leaf = ComponentCost("leaf", 1)
    branch = ComponentCost("branch", 1, calls=(CostCall(leaf, 7),))
    root = ComponentCost("root", 1, calls=(CostCall(branch, 3), CostCall(leaf, 2)))
    assert count_selected_calls(root, (branch, leaf)) == {branch: 3, leaf: 2}
    assert count_selected_calls(root, (leaf,)) == {leaf: 23}
    assert count_selected_calls(root, (root,)) == {root: 1}
    assert count_selected_calls(root, ()) == {}
    with pytest.raises(ValueError, match="cycle"):
        count_selected_calls(root, (leaf,), declared_calls={root: {root: 1}})
    with pytest.raises(ValueError, match="Unknown gateset"):
        resource_gateset("unknown")
