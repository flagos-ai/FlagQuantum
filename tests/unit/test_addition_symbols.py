# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Drawing assertions adapted from frozen Qualtran v0.7.0 addition.py.

"""Addition labels preserve source register directions and bit-port order."""

import pytest

from flagquantum.drawer.addition_symbols import (
    addition_circuit_labels,
    addition_wire_symbol,
)
from flagquantum.drawer.register_score import RegisterStep
from flagquantum.drawer.register_score_data import PortSide

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("side", ["LEFT", "RIGHT", "THRU"])
@pytest.mark.parametrize("operation", ["add", "out_of_place_add", "constant_add"])
def test_every_source_addition_glyph_and_direction(
    operation: str, side: PortSide
) -> None:
    from qualtran import QUInt, Register, Side
    from qualtran.bloqs.arithmetic import Add, AddK, OutOfPlaceAdder

    source = {
        "add": Add(QUInt(3)),
        "out_of_place_add": OutOfPlaceAdder(3),
        "constant_add": AddK(QUInt(3), 2),
    }[operation]
    step = RegisterStep(str(source), gate=operation)
    for name in {
        "add": ("a", "b"),
        "out_of_place_add": ("a", "b", "c"),
        "constant_add": ("x",),
    }[operation]:
        original = source.wire_symbol(
            Register(name, QUInt(3), side=Side[side])
        ).json_dict()
        glyph, attributes = addition_wire_symbol(step, name, side)
        assert original == {"symb_cls": glyph, "symb_attributes": attributes}
    original = source.wire_symbol(None).json_dict()
    glyph, attributes = addition_wire_symbol(step, None, None)
    assert original == {"symb_cls": glyph, "symb_attributes": attributes}


@pytest.mark.parametrize("left,right", [(3, 3), (4, 4), (2, 4), (2, 5), (3, 4), (3, 5)])
def test_original_bit_diagram_assertions(left: int, right: int) -> None:
    import cirq
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add

    assert (
        addition_circuit_labels(left, right)
        == cirq.circuit_diagram_info(Add(QUInt(left), QUInt(right))).wire_symbols
    )
