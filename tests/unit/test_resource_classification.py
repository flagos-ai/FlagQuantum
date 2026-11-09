"""Verify resource gate classification and filtering against the locked source."""

from math import pi

import pytest

from flagquantum.compiler._component_trace.call_graph import compose_cost_generalizer
from flagquantum.compiler._component_trace.resource_classification import (
    attribute_t_cost,
    classify_resource_gate,
    ignore_known_cliffords,
)
from flagquantum.compiler._component_trace.resource_generalization import (
    BookkeepingCall,
    CliffordCall,
    ConjunctionCall,
    ControlledRotationCall,
    ControlledSwapCall,
    PartitionedCall,
    PhaseCall,
    ResourceCall,
    RotationCall,
    generalize_rotation,
)
from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
def test_angle_and_exponent_boundaries_match_source() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates import Rx, YPowGate
    from qualtran.resource_counting.classify_bloqs import (
        bloq_is_clifford,
        bloq_is_rotation,
        bloq_is_t_like,
    )
    from sympy import Symbol

    angle = Symbol("a", real=True)
    for value in (
        0.0,
        0.25,
        -0.25,
        0.5,
        1.5,
        -2.0,
        2.0,
        4.0,
        0.25 + 5e-14,
        0.25 + 2e-12,
        0.5 - 2e-12,
        angle,
    ):
        for actual, source in (
            (RotationCall("rx", value * pi), Rx(value * pi)),
            (RotationCall("ypow", value), YPowGate(value)),
        ):
            result = classify_resource_gate(actual)
            assert (result == "clifford") is bool(bloq_is_clifford(source))
            assert (result == "t") is bool(bloq_is_t_like(source))
            assert (result == "rotation") is bool(bloq_is_rotation(source))


@pytest.mark.ftqc_reference
def test_explicit_and_controlled_gate_families_match_source() -> None:
    pytest.importorskip("qualtran")
    from qualtran import Adjoint, Controlled, CtrlSpec
    from qualtran.bloqs.basic_gates import CNOT, Rx, SGate, TGate, TwoBitSwap
    from qualtran.bloqs.mcmt.and_bloq import And
    from qualtran.resource_counting.classify_bloqs import (
        bloq_is_clifford,
        bloq_is_rotation,
        bloq_is_t_like,
    )

    rotation = RotationCall("rx", pi / 2)
    for native, source in (
        (CliffordCall("cx", 2), CNOT()),
        (CliffordCall("swap_bit", 2, True), Adjoint(TwoBitSwap())),
        (PhaseCall("s", True), SGate().adjoint()),
        (PhaseCall("t", True), TGate().adjoint()),
        (ConjunctionCall((1, 1)), And()),
        (ControlledRotationCall(rotation, (1,)), Controlled(Rx(pi / 2), CtrlSpec())),
        (
            ControlledRotationCall(PhaseCall("s"), (0,)),
            Controlled(SGate(), CtrlSpec(cvs=0)),
        ),
        (
            ControlledRotationCall(rotation, (1, 1)),
            Controlled(Rx(pi / 2), CtrlSpec(cvs=(1, 1))),
        ),
    ):
        result = classify_resource_gate(native)
        assert (result == "clifford") is bool(bloq_is_clifford(source))
        assert (result == "t") is bool(bloq_is_t_like(source))
        assert (result == "rotation") is bool(bloq_is_rotation(source))


@pytest.mark.ftqc_reference
def test_filtering_order_inside_partition_matches_source() -> None:
    pytest.importorskip("qualtran")
    from qualtran import QAny, Register
    from qualtran.bloqs.basic_gates import Rx
    from qualtran.bloqs.bookkeeping import AutoPartition
    from qualtran.resource_counting.generalizers import (
        PHI,
        generalize_rotation_angle,
        ignore_cliffords,
    )

    layout = (("q", 1),)
    source_layout = [(Register("q", QAny(1)), ["q"])]
    source = AutoPartition(Rx(pi / 2), source_layout)
    node = PartitionedCall(RotationCall("rx", pi / 2), layout)
    assert ignore_cliffords(source) is None
    assert ignore_known_cliffords(node) is None
    assert ignore_cliffords(generalize_rotation_angle(source)) == AutoPartition(
        Rx(PHI), source_layout
    )
    rule = compose_cost_generalizer((generalize_rotation, ignore_known_cliffords))
    assert rule(node) == PartitionedCall(RotationCall("rx", None), layout)
    unknown = ComponentCost("CNOT", 2)
    assert ignore_known_cliffords(unknown) is unknown


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("custom", [False, True])
def test_component_family_attribution_matches_source(custom: bool) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.bloqs.basic_gates import CSwap, TGate
    from qualtran.bloqs.for_testing.costing import CostingBloq
    from qualtran.resource_counting.classify_bloqs import classify_t_count_by_bloq_type

    from flagquantum.compiler.registers.arithmetic.arithmetic_resource_model import (
        source_add_cost,
    )

    t = ComponentCost("T", 1, GateCost(t=1))
    swap = ComponentCost(
        "CSwap", 7, calls=(CostCall(ComponentCost("Fredkin", 3, GateCost(cswap=1)), 3),)
    )
    add = source_add_cost(4, 4)
    branch = ComponentCost("branch", 8, calls=(CostCall(add), CostCall(swap, 2)))
    model = ComponentCost(
        "root", 9, calls=(CostCall(branch, 3), CostCall(swap), CostCall(t, 5))
    )
    source_branch = CostingBloq("branch", 8, [(Add(QUInt(4)), 1), (CSwap(3), 2)])
    source = CostingBloq("root", 9, [(source_branch, 3), (CSwap(3), 1), (TGate(), 5)])
    families: dict[ResourceCall, str] = {swap: "swaps"}
    source_families = {"qualtran.bloqs.basic_gates.swap": "swaps"} if custom else None
    if not custom:
        families.update({add: "arithmetic", t: "tgate"})
    expected = classify_t_count_by_bloq_type(source, source_families)
    assert attribute_t_cost(model, families.get) == expected
    assert expected["swaps"] == 84
    assert sum(expected.values()) == 125


@pytest.mark.ftqc_reference
def test_symbolic_t_attribution_filters_calls_before_cutoff() -> None:
    pytest.importorskip("qualtran")
    from qualtran import QAny
    from qualtran.bloqs.basic_gates import CNOT, CSwap, TGate
    from qualtran.bloqs.bookkeeping import Allocate
    from qualtran.bloqs.for_testing.costing import CostingBloq
    from qualtran.bloqs.swap_network import CSwapApprox
    from qualtran.resource_counting.classify_bloqs import classify_t_count_by_bloq_type
    from qualtran.resource_counting.generalizers import generalize_cswap_approx
    from sympy import Symbol

    from flagquantum.compiler._component_trace.resource_generalization import (
        generalize_approximate_swap,
    )

    n = Symbol("n", positive=True, integer=True)
    root = ComponentCost("root", 7)
    swap = ControlledSwapCall(3)
    t = PhaseCall("t")
    zero = ComponentCost("zero", 1)
    calls: dict[ResourceCall, dict[ResourceCall, SymbolicInt]] = {
        root: {
            ControlledSwapCall(3, approximate=True, adjoint=True): n,
            swap: 2,
            t: 5,
            zero: 6,
            CliffordCall("cx", 2): 7,
            BookkeepingCall("allocate", 1): 3,
        },
        swap: {t: 99},
    }
    source = CostingBloq(
        "root",
        7,
        [
            (CSwapApprox(3).adjoint(), n),
            (CSwap(3), 2),
            (TGate(), 5),
            (CostingBloq("zero", 1), 6),
            (CNOT(), 7),
            (Allocate(QAny(1)), 3),
        ],
    )
    families: dict[ResourceCall, str] = {swap: "swaps", t: "tgate"}
    costs: dict[ResourceCall, int] = {swap: 12, t: 1, zero: 0}
    actual = attribute_t_cost(
        root,
        families.get,
        declared_calls=calls,
        unit_t_cost=costs.__getitem__,
        generalizer=generalize_approximate_swap,
    )
    assert actual == classify_t_count_by_bloq_type(
        source, generalizer=generalize_cswap_approx
    )
    assert actual == {"swaps": 12 * n + 24, "tgate": 5}


def test_nonpositive_unit_costs_are_omitted_but_zero_repetitions_are_kept() -> None:
    negative = ComponentCost("negative", 1, GateCost(t=-2))
    zero = ComponentCost("zero", 1)
    positive = ComponentCost("positive", 1, GateCost(t=7))
    root = ComponentCost(
        "root",
        1,
        calls=(CostCall(negative, 2), CostCall(zero, 2), CostCall(positive, 0)),
    )
    assert attribute_t_cost(
        root,
        lambda node: (
            node.name if isinstance(node, ComponentCost) and node != root else None
        ),
    ) == {"positive": 0}
    assert attribute_t_cost(zero, lambda _: None) == {}
