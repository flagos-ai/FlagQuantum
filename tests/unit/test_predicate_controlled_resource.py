"""Check structured control and measured predicate resource composition."""

from __future__ import annotations

from dataclasses import asdict, fields
from typing import Any

import pytest

from flagquantum.compiler._component_trace.qec_gate_cost import (
    QECGateCost,
    query_qec_gate_cost,
)
from flagquantum.compiler._component_trace.resource_generalization import (
    ControlledRotationCall,
    ControlPredicateCall,
    PhaseCall,
    PredicateControlledCall,
)
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from tests._reference_dependencies import unavailable_reference

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("legacy", [False, True])
def test_controlled_wrapper_and_predicate_costs_match_source(legacy: bool) -> None:
    try:
        import numpy as np
        from qualtran._infra.controlled import CtrlSpec
        from qualtran._infra.data_types import QInt, QUInt
        from qualtran.bloqs.basic_gates.t_gate import TGate
        from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd
        from qualtran.bloqs.mcmt.ctrl_spec_and import CtrlSpecAnd
        from qualtran.resource_counting._bloq_counts import QECGatesCost
        from qualtran.resource_counting._costing import get_cost_value
        from qualtran.symbolics.types import Shaped
        from sympy import Symbol
    except ModuleNotFoundError as error:
        unavailable_reference(error)

    n = Symbol("n", positive=True, integer=True)
    registers = (
        ControlRegister(3, (4, 1)),
        ControlRegister(3, (-2,), "twos_complement"),
    )
    pattern = tuple(bit for register in registers for bit in register.bits())
    cases = (
        ((0,), CtrlSpec(cvs=0)),
        ((1,), CtrlSpec()),
        ((1, 0), CtrlSpec(cvs=(1, 0))),
        ((1, 0, 1), CtrlSpec(cvs=(1, 0, 1))),
        (
            pattern,
            CtrlSpec((QUInt(3), QInt(3)), cvs=(np.array([4, 1]), np.array([-2]))),
        ),
        (2, CtrlSpec(cvs=Shaped((2,)))),
        (3, CtrlSpec(cvs=Shaped((3,)))),
        (n, CtrlSpec(cvs=Shaped((n,)))),
    )
    model = QECGatesCost(legacy_shims=legacy)
    positive = ControlledRotationCall(PhaseCall("t"), (1,))

    def compare(native: QECGateCost, source: Any) -> None:
        expected = get_cost_value(source, model)
        assert asdict(native) == {
            field.name: getattr(expected, field.name) for field in fields(QECGateCost)
        }

    for controls, spec in cases:
        compare(
            query_qec_gate_cost(
                PredicateControlledCall(positive, controls), legacy_shims=legacy
            ),
            ControlledViaAnd(TGate(), spec),
        )
        if controls not in ((0,), (1,)):
            predicate = CtrlSpecAnd(spec)
            for adjoint in (False, True):
                compare(
                    query_qec_gate_cost(
                        ControlPredicateCall(controls, adjoint), legacy_shims=legacy
                    ),
                    predicate.adjoint() if adjoint else predicate,
                )
