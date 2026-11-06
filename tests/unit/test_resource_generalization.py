"""Check locked resource identity rewrites and their effect on graph weights."""

import pytest

from flagquantum.compiler._component_trace.call_graph import (
    compose_cost_generalizer,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.resource_generalization import (
    BookkeepingCall,
    ConjunctionCall,
    ControlledSwapCall,
    PartitionedCall,
    PhaseCall,
    ResourceCall,
    RotationCall,
    generalize_approximate_swap,
    generalize_control_values,
    generalize_rotation,
    ignore_allocation,
    ignore_register_partition,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
def test_angles_exponents_and_adjoint_phases_follow_source() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates import (
        Rx,
        Ry,
        Rz,
        SGate,
        TGate,
        XPowGate,
        YPowGate,
        ZPowGate,
    )
    from qualtran.resource_counting.generalizers import (
        PHI,
    )
    from qualtran.resource_counting.generalizers import (
        generalize_rotation_angle as source_rule,
    )

    for opcode, gate in (
        ("rx", Rx),
        ("ry", Ry),
        ("rz", Rz),
        ("xpow", XPowGate),
        ("ypow", YPowGate),
        ("zpow", ZPowGate),
    ):
        for value, precision in ((0.1, 1e-8), (-0.7, 1e-9)):
            assert source_rule(gate(value, eps=precision)) == gate(PHI, eps=precision)
            assert generalize_rotation(RotationCall(opcode, value, precision)) == (
                RotationCall(opcode, None, precision)
            )
    for opcode, gate in (("s", SGate), ("t", TGate)):
        assert source_rule(gate().adjoint()) == gate()
        assert generalize_rotation(PhaseCall(opcode, adjoint=True)) == PhaseCall(opcode)


@pytest.mark.ftqc_reference
def test_conjunction_control_arity_and_adjoint_boundaries_match_source() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.mcmt.and_bloq import And, MultiAnd
    from qualtran.resource_counting.generalizers import CV, generalize_cvs
    from qualtran.symbolics import HasLength
    from sympy import Symbol

    size = Symbol("n", positive=True, integer=True)
    for inverse in (False, True):
        assert generalize_cvs(And(0, 1, uncompute=inverse)) == And(
            CV, CV, uncompute=inverse
        )
        assert generalize_control_values(ConjunctionCall((0, 1), inverse)) == (
            ConjunctionCall((None, None), inverse)
        )
    multi = MultiAnd((1, 0, 1, 0))
    assert generalize_cvs(multi) == MultiAnd(HasLength(4))
    assert generalize_cvs(multi.adjoint()) == multi.adjoint()
    assert generalize_cvs(MultiAnd(HasLength(size))) == MultiAnd(HasLength(size))
    assert generalize_control_values(ConjunctionCall((1, 0, 1, 0))) == ConjunctionCall(
        4
    )
    adjoint = ConjunctionCall((1, 0, 1, 0), adjoint=True)
    assert generalize_control_values(adjoint) == adjoint
    assert generalize_control_values(ConjunctionCall(size)) == ConjunctionCall(size)


@pytest.mark.ftqc_reference
def test_partition_and_allocation_wrapper_rules_match_source() -> None:
    pytest.importorskip("qualtran")
    from qualtran import QAny, Register
    from qualtran.bloqs.basic_gates import Rx
    from qualtran.bloqs.bookkeeping import Allocate, AutoPartition, Free, Join, Split
    from qualtran.resource_counting.generalizers import (
        PHI,
        generalize_rotation_angle,
        ignore_alloc_free,
        ignore_split_join,
    )

    layout = (("q", 1),)
    source_layout = [(Register("q", QAny(1)), ["q"])]
    rx = RotationCall("rx", 0.2)
    wrapped = PartitionedCall(PartitionedCall(rx, layout), layout)
    source_rx = AutoPartition(AutoPartition(Rx(0.2), source_layout), source_layout)
    assert ignore_split_join(source_rx) == Rx(0.2)
    assert ignore_register_partition(wrapped) == rx
    assert generalize_rotation_angle(source_rx) == AutoPartition(
        AutoPartition(Rx(PHI), source_layout), source_layout
    )
    assert generalize_rotation(wrapped) == PartitionedCall(
        PartitionedCall(RotationCall("rx", None), layout), layout
    )
    for opcode, source in (("split", Split), ("join", Join)):
        node = BookkeepingCall(opcode, layout)
        assert ignore_split_join(source(QAny(1))) is None
        assert ignore_register_partition(node) is None
        assert ignore_allocation(node) == node
    for opcode, source in (("allocate", Allocate), ("free", Free)):
        node = BookkeepingCall(opcode, layout)
        assert ignore_alloc_free(source(QAny(1))) is None
        assert ignore_allocation(node) is None
        assert ignore_allocation(PartitionedCall(node, layout)) is None
        assert ignore_register_partition(node) == node


@pytest.mark.ftqc_reference
def test_composed_generalization_matches_weighted_source_graph() -> None:
    pytest.importorskip("qualtran")
    from qualtran import QAny
    from qualtran.bloqs.basic_gates import CSwap, Rx, TGate
    from qualtran.bloqs.bookkeeping import Allocate
    from qualtran.bloqs.for_testing.costing import CostingBloq
    from qualtran.bloqs.swap_network import CSwapApprox
    from qualtran.resource_counting import get_bloq_call_graph
    from qualtran.resource_counting.generalizers import (
        PHI,
        generalize_cswap_approx,
        generalize_rotation_angle,
        ignore_alloc_free,
    )
    from sympy import Symbol

    n = Symbol("n", nonnegative=True, integer=True)
    root = ComponentCost("root", 7)
    raw: dict[ResourceCall, SymbolicInt] = {
        RotationCall("rx", 0.3): n,
        RotationCall("rx", -0.2): 2,
        PhaseCall("t", adjoint=True): 5,
        ControlledSwapCall(3, approximate=True): 7,
        ControlledSwapCall(3, approximate=True, adjoint=True): 1,
        ControlledSwapCall(3): 2,
        BookkeepingCall("allocate", 7): 4,
    }
    source_root = CostingBloq(
        "root",
        7,
        list(
            zip(
                (
                    Rx(0.3),
                    Rx(-0.2),
                    TGate().adjoint(),
                    CSwapApprox(3),
                    CSwapApprox(3).adjoint(),
                    CSwap(3),
                    Allocate(QAny(7)),
                ),
                raw.values(),
                strict=True,
            )
        ),
    )
    rules = (ignore_allocation, generalize_rotation, generalize_approximate_swap)
    graph = symbolic_call_graph(root, {root: raw}, generalizer=rules, max_depth=1)
    expected, sigma = get_bloq_call_graph(
        source_root,
        generalizer=(
            ignore_alloc_free,
            generalize_rotation_angle,
            generalize_cswap_approx,
        ),
        max_depth=1,
    )
    native_to_source = {
        root: source_root,
        RotationCall("rx", None): Rx(PHI),
        PhaseCall("t"): TGate(),
        ControlledSwapCall(3): CSwap(3),
    }
    assert {
        (native_to_source[parent], native_to_source[child]): count
        for (parent, child), count in graph.direct_calls.items()
    } == {(parent, child): data["n"] for parent, child, data in expected.edges.data()}
    assert {
        native_to_source[node]: count for node, count in graph.leaf_counts.items()
    } == sigma
    assert graph.leaf_counts[ControlledSwapCall(3)] == 10
    assert compose_cost_generalizer(rules)(root) is root
