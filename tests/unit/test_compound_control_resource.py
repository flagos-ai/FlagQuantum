"""Validate the source control dispatch and measured Fredkin composition."""

from dataclasses import asdict, fields

import pytest
import torch

from flagquantum.compiler._component_trace.compound_control_resource import (
    compound_control_key,
)
from flagquantum.compiler._component_trace.gate_budget import legacy_gate_budget
from flagquantum.compiler._component_trace.qec_gate_cost import (
    QECGateCost,
    query_qec_gate_cost,
)
from flagquantum.compiler._component_trace.resource_generalization import (
    CliffordCall,
    ControlledRegisterSwapCall,
    ControlledSwapCall,
    FixedTCall,
    MergedFredkinCall,
    PredicateControlledCall,
)
from flagquantum.compiler.registers.control_predicate import ControlRegister
from flagquantum.compiler.registers.controlled_component import controlled_component_ir
from flagquantum.compiler.registers.controlled_swap import controlled_swap_bit_ir
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("legacy", [False, True])
def test_compound_call_paths_match_locked_source(legacy: bool) -> None:
    pytest.importorskip("qualtran")
    from qualtran import CtrlSpec
    from qualtran.bloqs.basic_gates import CNOT, CSwap, Toffoli, TwoBitCSwap
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    families = (
        (CliffordCall("cx", 2), CNOT()),
        (FixedTCall("toffoli"), Toffoli()),
        (FixedTCall("fredkin"), TwoBitCSwap()),
        (ControlledSwapCall(1), CSwap(1)),
        (ControlledSwapCall(3), CSwap(3)),
    )
    for gate, source in families:
        for bits in ((1,), (0,), (1, 1), (1, 0), (0, 1, 0)):
            spec = CtrlSpec(cvs=bits[0] if len(bits) == 1 else bits)
            call = compound_control_key(gate, bits)
            assert call is not None
            expected = get_cost_value(
                source.controlled(spec), QECGatesCost(legacy_shims=legacy)
            )
            assert asdict(query_qec_gate_cost(call, legacy_shims=legacy)) == {
                field.name: getattr(expected, field.name)
                for field in fields(QECGateCost)
            }
    assert compound_control_key(CliffordCall("cx", 2), (1,)) == FixedTCall("toffoli")
    assert compound_control_key(FixedTCall("toffoli"), (0,)) == PredicateControlledCall(
        CliffordCall("cx", 2), (0, 1, 1)
    )
    assert compound_control_key(
        ControlledSwapCall(3), (1,)
    ) == ControlledRegisterSwapCall(3)
    assert compound_control_key(MergedFredkinCall((0, 1)), (0,)) == MergedFredkinCall(
        (0, 0, 1)
    )
    wrapper = PredicateControlledCall(FixedTCall("toffoli"), (0,))
    assert compound_control_key(wrapper, (1, 0)) == PredicateControlledCall(
        FixedTCall("toffoli"), (1, 0, 0)
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("new_control", [0, 1])
def test_merged_fredkin_gate_flow_and_all_measurement_branches(
    new_control: int,
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import CtrlSpec
    from qualtran.bloqs.basic_gates import TwoBitCSwap

    source = TwoBitCSwap().controlled(CtrlSpec(cvs=new_control))
    gates = controlled_component_ir(
        ((0, 1),),
        (ControlRegister(1, (new_control, 1)),),
        lambda q: controlled_swap_bit_ir(q, 2, 3),
        (4,),
        (0,),
    )
    expected_gates = reference_arithmetic_gates(
        source,
        {"ctrl": (0, 1), "x": (2,), "y": (3,)},
        (4,),
        {4: 0},
        preserve_components=True,
    )
    assert_source(gates, expected_gates)
    cost = query_qec_gate_cost(MergedFredkinCall((new_control, 1)))
    assert cost.measurement == sum(op.name == "measure" for op in gates) == 1
    assert (
        legacy_gate_budget(cost)["t"]
        == sum(op.name in {"t", "tdg"} for op in gates)
        == 11
    )

    state = torch.zeros(32, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    for word in range(16):
        active = (word >> 2) == 2 * new_control + 1
        swapped = (word & 12) | ((word & 1) << 1) | ((word & 2) >> 1)
        amplitude = complex(word + 1, 2 * word - 3) / 64
        state[2 * word] = amplitude
        expected[2 * (swapped if active else word)] = amplitude
    for outcome in (0, 1):
        torch.testing.assert_close(
            forced_branch(state, gates, (outcome,)),
            expected / 2**0.5,
            rtol=0,
            atol=2e-12,
        )
