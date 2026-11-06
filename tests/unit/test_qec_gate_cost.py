"""Compare typed QEC graph costs with locked Qualtran QECGatesCost."""

from dataclasses import asdict, fields
from math import pi

import pytest

from flagquantum.compiler._component_trace.qec_gate_cost import (
    QECGateCost,
    query_qec_gate_cost,
)
from flagquantum.compiler._component_trace.resource_generalization import (
    BookkeepingCall,
    CliffordCall,
    ConjunctionCall,
    ControlledRotationCall,
    ControlledSwapCall,
    DeclaredResourceCalls,
    FanoutCall,
    FixedTCall,
    MeasurementCall,
    PartitionedCall,
    PhaseCall,
    ResourceCall,
    RotationCall,
    ZeroCostCall,
)
from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)

pytestmark = pytest.mark.unit


def source_categories(cost: object) -> dict[str, object]:
    return {field.name: getattr(cost, field.name) for field in fields(QECGateCost)}


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("legacy", [False, True])
def test_classified_leaves_and_declared_symbolic_children_match_source(
    legacy: bool,
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import Controlled, CtrlSpec, QAny
    from qualtran.bloqs.basic_gates import (
        CSwap,
        Hadamard,
        Identity,
        MeasureX,
        MeasureZ,
        Rx,
        TGate,
        Toffoli,
    )
    from qualtran.bloqs.bookkeeping import Allocate
    from qualtran.bloqs.for_testing.costing import CostingBloq
    from qualtran.bloqs.mcmt import And, MultiAnd, MultiTargetCNOT
    from qualtran.bloqs.swap_network import CSwapApprox
    from qualtran.resource_counting import QECGatesCost, get_cost_value
    from qualtran.symbolics import HasLength
    from sympy import Symbol

    n = Symbol("n", positive=True, integer=True)
    model = QECGatesCost(legacy_shims=legacy)
    pairs = (
        (RotationCall("rx", pi / 4), Rx(pi / 4), n),
        (PhaseCall("t", True), TGate().adjoint(), 2),
        (FixedTCall("toffoli"), Toffoli(), 3),
        (ControlledSwapCall(n), CSwap(n), 4),
        (
            ControlledSwapCall(2, approximate=True, adjoint=True),
            CSwapApprox(2).adjoint(),
            2,
        ),
        (MeasurementCall("x"), MeasureX(), 5),
        (MeasurementCall("z"), MeasureZ(), 7),
        (ConjunctionCall((0, 1)), And(0, 1), 3),
        (ConjunctionCall((0, 0), True), And(0, 0, uncompute=True), 2),
        (ConjunctionCall((0, 1, 0), True), MultiAnd((0, 1, 0)).adjoint(), 3),
        (ConjunctionCall(n), MultiAnd(HasLength(n)), 2),
        (CliffordCall("h", 1), Hadamard(), 4),
        (
            ControlledRotationCall(RotationCall("rx", pi / 2), (1,)),
            Controlled(Rx(pi / 2), CtrlSpec()),
            1,
        ),
        (BookkeepingCall("allocate", 2), Allocate(QAny(2)), 3),
        (ZeroCostCall("identity", (2,)), Identity(2), 2),
        (FanoutCall(12), MultiTargetCNOT(12), 1),
    )
    for native, source, _ in pairs:
        assert asdict(
            query_qec_gate_cost(native, legacy_shims=legacy)
        ) == source_categories(get_cost_value(source, model))
    root = ComponentCost("program", 12)
    calls: DeclaredResourceCalls = {root: {native: count for native, _, count in pairs}}
    source_root = CostingBloq(
        "program", 12, [(source, count) for _, source, count in pairs]
    )
    assert asdict(
        query_qec_gate_cost(root, declared_calls=calls, legacy_shims=legacy)
    ) == (source_categories(get_cost_value(source_root, model)))


@pytest.mark.ftqc_reference
def test_seed_and_static_cost_precede_legacy_annotation() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates import TGate
    from qualtran.bloqs.for_testing.costing import CostingBloq
    from qualtran.cirq_interop.t_complexity_protocol import TComplexity
    from qualtran.resource_counting import GateCounts, QECGatesCost, get_cost_value

    class Annotated(CostingBloq):
        def _t_complexity_(self) -> TComplexity:
            return TComplexity(t=3, clifford=4, rotations=5)

    legacy = QECGatesCost(legacy_shims=True)
    source_static = GateCounts(toffoli=7)
    source = Annotated("model", 2, [(TGate(), 2)], [(legacy, source_static)])
    plain = Annotated("model", 2, [(TGate(), 2)])
    native = ComponentCost("model", 2, GateCost(t=2))
    alias = ComponentCost("alias", 2)
    annotation: dict[ResourceCall, GateCost] = {
        native: GateCost(t=3, clifford=4, rotation=5)
    }
    override: dict[ResourceCall, QECGateCost] = {native: QECGateCost(toffoli=7)}

    def generalize(node: ResourceCall) -> ResourceCall:
        return native if node == alias else node

    with pytest.warns(DeprecationWarning):
        assert asdict(
            query_qec_gate_cost(
                alias,
                legacy_shims=True,
                legacy_annotations=annotation,
                generalizer=generalize,
            )
        ) == source_categories(get_cost_value(plain, legacy))
    assert query_qec_gate_cost(
        alias, legacy_annotations=annotation, generalizer=generalize
    ) == QECGateCost(t=2)
    assert asdict(
        query_qec_gate_cost(
            alias,
            legacy_shims=True,
            legacy_annotations=annotation,
            static_costs=override,
            generalizer=generalize,
        )
    ) == source_categories(get_cost_value(source, legacy))
    seeded: dict[ResourceCall, QECGateCost] = {native: QECGateCost(measurement=9)}
    assert asdict(
        query_qec_gate_cost(
            alias,
            legacy_shims=True,
            legacy_annotations=annotation,
            static_costs=override,
            cache=seeded,
            generalizer=generalize,
        )
    ) == (
        source_categories(
            get_cost_value(
                source, legacy, costs_cache={source: GateCounts(measurement=9)}
            )
        )
    )


def test_component_costs_reuse_shared_cache_and_fail_closed() -> None:
    leaf = ComponentCost("leaf", 2, GateCost(t=4))
    branch = ComponentCost("branch", 2, calls=(CostCall(leaf, 3),))
    root = ComponentCost(
        "root", 2, GateCost(clifford=1), (CostCall(branch), CostCall(leaf, 2))
    )
    cache: dict[ResourceCall, QECGateCost] = {}
    assert query_qec_gate_cost(root, cache=cache) == QECGateCost(t=20, clifford=1)
    assert cache == {
        leaf: QECGateCost(t=4),
        branch: QECGateCost(t=12),
        root: QECGateCost(t=20, clifford=1),
    }
    wrapped = PartitionedCall(PhaseCall("t"), (("q", 1),))
    assert query_qec_gate_cost(wrapped) == QECGateCost(t=1)
    unsupported = ControlledRotationCall(PhaseCall("t"), (1, 1))
    with pytest.raises(NotImplementedError, match="No QEC gate model"):
        query_qec_gate_cost(unsupported)
    assert (
        query_qec_gate_cost(unsupported, declared_calls={unsupported: {}})
        == QECGateCost()
    )
    with pytest.raises(ValueError, match="cycle"):
        query_qec_gate_cost(root, declared_calls={root: {root: 1}})
