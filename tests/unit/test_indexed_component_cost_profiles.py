"""Compare both pinned QEC profiles without changing source or executable IR."""

from dataclasses import asdict
from itertools import product
from math import prod

import pytest

from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.control.control_predicate_resource_model import (
    control_predicate_cost,
)
from flagquantum.compiler.registers.control.indexed_component_control_resource_model import (
    specialized_indexed_component_cost,
)
from flagquantum.compiler.registers.control.indexed_component_resource_model import (
    indexed_component_cost,
)
from flagquantum.qec.resources.estimation import estimate_ftqc_resources

pytestmark = pytest.mark.unit


def _assert_model(model, source, legacy):
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    actual = estimate_ftqc_resources(model).logical
    original = get_cost_value(source, QECGatesCost(legacy_shims=legacy))
    assert asdict(actual.gates) == {
        name: getattr(original, name) for name in asdict(actual.gates)
    }
    assert actual.peak_logical_qubits == get_cost_value(source, QubitCount())
    assert actual.gates.total_t_count() == original.total_t_count()


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "values", [(0, 0), (0, 1), (1, 0), (1, 1), (0, 1, 0), (1, 0, 1, 0)]
)
@pytest.mark.parametrize("uncompute,legacy", tuple(product((False, True), repeat=2)))
def test_owned_predicate_model_matches_original(values, uncompute, legacy):
    from qualtran import CtrlSpec
    from qualtran.bloqs.mcmt.ctrl_spec_and import CtrlSpecAnd

    source = CtrlSpecAnd(CtrlSpec(cvs=values))
    _assert_model(
        control_predicate_cost(values, uncompute=uncompute, legacy_shims=legacy),
        source.adjoint() if uncompute else source,
        legacy,
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "shape,widths,controls",
    [
        ((2,), (1,), 0),
        ((3,), (2,), 0),
        ((3,), (4,), 1),
        ((2,), (1,), 3),
        ((2, 3), (1, 2), 0),
        ((2, 2), (1, 1), 1),
        ((1, 2), (1, 1), 1),
        ((2, 2, 2), (1, 1, 1), 2),
    ],
)
@pytest.mark.parametrize("adjoint,legacy", tuple(product((False, True), repeat=2)))
def test_target_position_profiles_and_nested_parent(
    shape, widths, controls, adjoint, legacy
):
    import cirq
    import numpy as np
    from qualtran import BQUInt, QAny, Register
    from qualtran.bloqs.multiplexers.apply_gate_to_lth_target import ApplyGateToLthQubit

    count = prod(shape)
    source = ApplyGateToLthQubit(
        tuple(
            Register(f"s{i}", BQUInt(width, size))
            for i, (width, size) in enumerate(zip(widths, shape, strict=True))
        ),
        lambda *indices: (
            cirq.X if int(np.ravel_multi_index(indices, shape)) % 2 == 0 else cirq.Z
        ),
        control_regs=(Register("control", QAny(controls)),) if controls else (),
    )
    children = tuple(
        ComponentCost("CX/CZ", 2, GateCost(clifford=1)) for _ in range(count)
    )
    model = indexed_component_cost(
        widths,
        shape,
        children,
        target_width=count,
        child_target_width=1,
        control_count=controls,
        adjoint=adjoint,
        legacy_shims=legacy,
    )
    _assert_model(model, source.adjoint() if adjoint else source, legacy)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("values", [(0,), (1,), (0, 1), (1, 0, 1)])
@pytest.mark.parametrize("existing", [None, 0, 1])
@pytest.mark.parametrize("adjoint,legacy", tuple(product((False, True), repeat=2)))
def test_control_system_profiles_match_original(values, existing, adjoint, legacy):
    import numpy as np
    from qualtran import CtrlSpec
    from qualtran.bloqs.basic_gates import XGate, ZGate
    from qualtran.bloqs.multiplexers.apply_lth_bloq import ApplyLthBloq

    base = ApplyLthBloq(np.array([XGate(), ZGate()]), control_val=existing)
    source = (base.adjoint() if adjoint else base).controlled(CtrlSpec(cvs=values))
    children = (
        ComponentCost("CX", 2, GateCost(clifford=1)),
        ComponentCost("CZ", 2, GateCost(clifford=1)),
    )
    model = specialized_indexed_component_cost(
        (1,),
        (2,),
        children,
        (ControlRegister(1, values),),
        target_width=1,
        current_control_value=existing,
        adjoint=adjoint,
        legacy_shims=legacy,
    )
    _assert_model(model, source, legacy)


@pytest.mark.parametrize("legacy", [None, 0, 1, "legacy"])
def test_nonboolean_profile_rejected(legacy):
    with pytest.raises(ValueError, match="boolean"):
        indexed_component_cost(
            (1,),
            (2,),
            (ComponentCost("CX", 2),) * 2,
            target_width=1,
            legacy_shims=legacy,
        )
    with pytest.raises(ValueError, match="boolean"):
        control_predicate_cost((1, 1), legacy_shims=legacy)


def test_model_profile_does_not_emit_or_change_execution(monkeypatch):
    import flagquantum.compiler.registers.control.indexed_component as dispatch
    from tests.unit.test_indexed_component_control import _native

    before = _native((0, 1), 0, False)[0]

    def refuse(*args, **kwargs):
        pytest.fail("A model query attempted circuit construction")

    with monkeypatch.context() as patch:
        patch.setattr(dispatch, "indexed_component_ir", refuse)
        children = (
            ComponentCost("CX", 2, GateCost(clifford=1)),
            ComponentCost("CZ", 2, GateCost(clifford=1)),
        )
        reports = [
            estimate_ftqc_resources(
                specialized_indexed_component_cost(
                    (1,),
                    (2,),
                    children,
                    (ControlRegister(1, (0, 1)),),
                    target_width=1,
                    current_control_value=0,
                    legacy_shims=legacy,
                )
            )
            for legacy in (False, True)
        ]
    assert _native((0, 1), 0, False)[0] == before
    assert reports[0].logical.gates.clifford == 6
    assert reports[1].logical.gates.clifford == 22
    assert {report.logical.gates.total_t_count() for report in reports} == {12}


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "position,adjoint,legacy", tuple(product((False, True), repeat=3))
)
def test_phase_children_keep_actual_frontend_identity(position, adjoint, legacy):
    import cirq
    import numpy as np
    from qualtran import BQUInt, QBit, Register
    from qualtran.bloqs.basic_gates import SGate, TGate
    from qualtran.bloqs.multiplexers.apply_gate_to_lth_target import ApplyGateToLthQubit
    from qualtran.bloqs.multiplexers.apply_lth_bloq import ApplyLthBloq
    from qualtran.cirq_interop import cirq_gate_to_bloq

    from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
    from flagquantum.compiler._component_trace.resource_generalization import (
        ControlledRotationCall,
        ControlledZPowerCall,
        PhaseCall,
    )

    regs = (Register("selection", BQUInt(1, 2)),)
    if position:
        source = ApplyGateToLthQubit(
            regs,
            lambda index: (cirq.S, cirq.T)[index],
            control_regs=(Register("control", QBit()),),
        )
        originals = [cirq_gate_to_bloq(gate.controlled()) for gate in (cirq.S, cirq.T)]
        calls = [
            ControlledZPowerCall(sign * exponent)
            for exponent in (0.5, 0.25)
            for sign in ((-1,) if adjoint else (1,))
        ]
    else:
        source = ApplyLthBloq(
            np.array([SGate(), TGate()]), selection_regs=regs, control_val=1
        )
        originals = [gate.controlled() for gate in (SGate(), TGate())]
        calls = [
            ControlledRotationCall(PhaseCall(opcode, adjoint), (1,))
            for opcode in ("s", "t")
        ]
    if adjoint:
        source = source.adjoint()
        originals = [op.adjoint() for op in originals]
    children = []
    for key, original in zip(calls, originals, strict=True):
        cost = query_qec_gate_cost(key, legacy_shims=legacy)
        child = ComponentCost(
            "controlled phase", 3 if position else 2, GateCost(**asdict(cost))
        )
        _assert_model(child, original, legacy)
        children.append(child)
    model = indexed_component_cost(
        (1,),
        (2,),
        tuple(children),
        target_width=2 if position else 1,
        child_target_width=1,
        control_count=1,
        adjoint=adjoint,
        legacy_shims=legacy,
    )
    _assert_model(model, source, legacy)
