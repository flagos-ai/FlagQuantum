"""Subtraction drawing callbacks preserve original typed and sided labels."""

import pytest
from attrs import asdict

from flagquantum.core.ir.program.register_types import unsigned_register_type
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.drawer.register_score import RegisterStep, layout_register_score
from flagquantum.drawer.register_score_data import register_score_data
from flagquantum.drawer.subtraction_symbols import subtraction_wire_symbol
from tests.unit.test_drawer_component_score import assert_source_geometry

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("side", tuple(RegisterSide))
def test_original_labels_and_directional_glyphs(
    reverse: bool, side: RegisterSide
) -> None:
    from qualtran import QUInt, Side
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.arithmetic.subtraction import Subtract, SubtractFrom

    source = (SubtractFrom if reverse else Subtract)(QUInt(3))
    step = RegisterStep(str(source), gate="subtract_from" if reverse else "subtract")
    for name in (None, "a", "b"):
        reg = (
            None
            if name is None
            else Register(name, unsigned_register_type(3), side=side)
        )
        original = (
            None
            if name is None
            else SourceRegister(name, QUInt(3), side=Side[side.name])
        )
        symbol = source.wire_symbol(original)
        assert subtraction_wire_symbol(step, reg, ()) == (
            type(symbol).__name__,
            asdict(symbol),
        )
    with pytest.raises(ValueError):
        source.wire_symbol(SourceRegister("other", QUInt(3)))
    with pytest.raises(ValueError, match="a or b"):
        subtraction_wire_symbol(step, Register("other", unsigned_register_type(3)), ())


@pytest.mark.parametrize("reverse", (False, True))
def test_actual_register_score_consumer_matches_original_geometry(
    reverse: bool,
) -> None:
    from qualtran import BloqBuilder, QUInt
    from qualtran.bloqs.arithmetic.subtraction import Subtract, SubtractFrom
    from qualtran.drawing import get_musical_score_data

    source = (SubtractFrom if reverse else Subtract)(QUInt(3))
    builder, inputs = BloqBuilder.from_signature(source.signature)
    original = get_musical_score_data(
        builder.finalize(**builder.add_d(source, **inputs))
    )
    registers = tuple(Register(name, unsigned_register_type(3)) for name in ("a", "b"))
    step = RegisterStep(
        str(source),
        ("a", "b"),
        ("a", "b"),
        gate="subtract_from" if reverse else "subtract",
        input_registers=registers,
        output_registers=registers,
    )
    score = layout_register_score(
        {"a": 3, "b": 3},
        ("a", "b"),
        (step,),
        input_registers=registers,
        output_registers=registers,
    )
    actual = register_score_data(score, register_wire_symbol=subtraction_wire_symbol)
    assert_source_geometry(actual, original, ordered_ports=True)


def test_reject_unrelated_display_step() -> None:
    with pytest.raises(ValueError, match="subtraction display"):
        subtraction_wire_symbol(RegisterStep("Add", gate="add"), None, None)
