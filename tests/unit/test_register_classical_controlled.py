"""Source classical activation and native complex measurement feedback."""

from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.primitives.classical_controlled import (
    classically_controlled_ir,
)
from flagquantum.core.classical_control import validate_classical_control
from flagquantum.core.ir import Instruction
from tests.unit.measurement_checks import forced_branch

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("required", [0, 1, 2, 3])
def test_classical_word_activation_from_locked_source(required: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran import CtrlSpec, CUInt
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.mcmt.classically_controlled import ClassicallyControlled

    source = ClassicallyControlled(XGate(), CtrlSpec(CUInt(2), cvs=required))
    gates = classically_controlled_ir(
        ((3, 1),), (ControlRegister(2, (required,)),), (Instruction("x", (2,)),)
    )
    measurements = (
        Instruction("measure", (0,), metadata={"classical_bit": 3, "is_dynamic": True}),
        Instruction("measure", (1,), metadata={"classical_bit": 1, "is_dynamic": True}),
    )
    assert validate_classical_control((*measurements, *gates), n_qubits=3) == 4
    initial = torch.tensor(
        (1 + 2j, 2 - 1j, 2 + 1j, -1j, -3 + 1j, 1 - 1j, 3 + 2j, -1 - 2j),
        dtype=torch.complex128,
    )
    initial /= initial.norm()
    for high, low in product((0, 1), repeat=2):
        value = high * 2 + low
        expected = torch.zeros_like(initial)
        for target in (0, 1):
            output = source.call_classically(
                **{source.ctrl_reg_names[0]: value, "q": target}
            )
            assert output[0] == value
            expected[2 * value + output[1]] = initial[2 * value + target]
        actual = forced_branch(initial, (*measurements, *gates), (high, low))
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=0)


def test_existing_conditions_intersect_and_drop_unreachable_child() -> None:
    body = (
        Instruction("x", (0,), metadata={"conditions": ((0, 0),)}),
        Instruction(
            "rz",
            (0,),
            {"theta": 0.25},
            metadata={"condition_clauses": (((0, 1),), ((1, 1),)), "marker": "child"},
        ),
        Instruction("z", (0,), metadata={"condition_clauses": (((1, 0),), ((2, 1),))}),
    )
    rz, z = classically_controlled_ir(((0,),), (ControlRegister(1, (1,)),), body)
    assert rz.params == {"theta": 0.25}
    assert rz.metadata == {"conditions": ((0, 1),), "marker": "child"}
    assert z.metadata == {"condition_clauses": (((0, 1), (1, 0)), ((0, 1), (2, 1)))}


@pytest.mark.parametrize("name", ["measure", "reset"])
def test_flat_ir_rejects_guarded_side_effect(name: str) -> None:
    with pytest.raises(ValueError, match="cannot guard"):
        classically_controlled_ir(
            ((0,),),
            (ControlRegister(1, (1,)),),
            (Instruction(name, (0,), metadata={"is_dynamic": True}),),
        )
