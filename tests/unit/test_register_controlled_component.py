"""Compare structured control wrappers with the pinned source graph."""

from collections import Counter
from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.arithmetic.controlled_addition import (
    controlled_add_ir,
)
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.control.controlled_component import (
    controlled_component_ir,
)
from flagquantum.core.ir import Instruction
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_control_predicate import (
    CASES,
    ControlCase,
    control_fixture,
)
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "case",
    [
        (("QBit", 1, (), (1,), "unsigned"),),
        (("QBit", 1, (1,), (0,), "unsigned"),),
        CASES[0],
        CASES[1],
        CASES[3],
    ],
)
@pytest.mark.parametrize("axis", ("x", "z"))
def test_source_wrapper_gate_flow_and_complex_channel(
    case: ControlCase, axis: str
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.basic_gates import XGate, ZGate
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    spec, registers, groups = control_fixture(case)
    n = sum(map(len, groups))
    target = n
    work = tuple(range(n + 1, 2 * n))
    bits = tuple(range(40, 40 + len(work)))
    source = ControlledViaAnd(XGate() if axis == "x" else ZGate(), spec)
    gates = controlled_component_ir(
        groups,
        registers,
        lambda ctrl: (Instruction(f"c{axis}", (ctrl, target)),),
        work,
        bits,
    )
    ports = dict(zip(source.ctrl_reg_names, groups, strict=True))
    ports["q"] = (target,)
    expected = reference_arithmetic_gates(
        source,
        ports,
        work,
        dict(zip(work, bits, strict=True)),
    )
    assert_source(gates, expected)
    counts = Counter(gate.name for gate in gates)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 4 * (n - 1)
    assert counts["measure"] == counts["reset"] == model.measurement == n - 1
    if n == 1:
        assert counts["x"] == 2 * (registers[0].values == (0,))

    pattern = 0
    for reg in registers:
        for bit in reg.bits():
            pattern = (pattern << 1) | bit
    state = torch.zeros(1 << (2 * n), dtype=torch.complex128)
    reference = torch.zeros_like(state)
    for control, data in product(range(1 << n), (0, 1)):
        amplitude = complex(control + 1, data - control) / (1 << n)
        state[((control << 1) | data) << (n - 1)] = amplitude
        active = control == pattern
        out = data ^ (axis == "x" and active)
        sign = -1 if axis == "z" and active and data else 1
        reference[((control << 1) | out) << (n - 1)] = amplitude * sign
    for history in product((0, 1), repeat=n - 1):
        torch.testing.assert_close(
            forced_branch(state, gates, history),
            reference * 2 ** (-(n - 1) / 2),
            atol=2e-12,
            rtol=0,
        )


def test_single_control_cannot_claim_workspace() -> None:
    with pytest.raises(ValueError, match="no predicate work"):
        controlled_component_ir(
            ((0,),), (ControlRegister(1, (0,)),), lambda _: (), (1,), ()
        )


@pytest.mark.ftqc_reference
def test_predicate_wraps_addition_with_separate_child_measurements() -> None:
    pytest.importorskip("qualtran")
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    spec, registers, groups = control_fixture(CASES[0])
    source = ControlledViaAnd(Add(QUInt(1), QUInt(2)), spec)
    gates = controlled_component_ir(
        groups,
        registers,
        lambda ctrl: controlled_add_ir(ctrl, (2,), (3, 4), (6,), (7,), (51,), (52,)),
        (5,),
        (50,),
    )
    inputs = {source.ctrl_reg_names[0]: groups[0], "a": (2,), "b": (3, 4)}
    expected = reference_arithmetic_gates(
        source, inputs, (5, 6, 7), {5: 50, 6: 51, 7: 52}
    )
    assert_source(gates, expected)
    model = get_cost_value(source, QECGatesCost())
    assert model.measurement == sum(g.name == "measure" for g in gates) == 3
    assert model.total_t_count() == sum(g.name in ("t", "tdg") for g in gates) == 12

    vector = torch.zeros(256, dtype=torch.complex128)
    reference = torch.zeros_like(vector)
    for control, left, right in product(range(4), range(2), range(4)):
        amp = complex(1 + control + right, left - right) / 32
        inp = (control << 3) | (left << 2) | right
        out = (inp & ~3) | ((right + left * (control == 1)) % 4)
        vector[inp << 3] = amp
        reference[out << 3] = amp
    for history in product((0, 1), repeat=3):
        torch.testing.assert_close(
            forced_branch(vector, gates, history),
            reference / 2**1.5,
            rtol=0,
            atol=2e-12,
        )
