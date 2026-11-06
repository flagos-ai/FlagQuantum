"""Verify resource gate classification and filtering against the locked source."""

from math import pi

import pytest

from flagquantum.compiler._component_trace.call_graph import compose_cost_generalizer
from flagquantum.compiler._component_trace.resource_classification import (
    classify_resource_gate,
    ignore_known_cliffords,
)
from flagquantum.compiler._component_trace.resource_generalization import (
    CliffordCall,
    ConjunctionCall,
    ControlledRotationCall,
    PartitionedCall,
    PhaseCall,
    RotationCall,
    generalize_rotation,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost

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
