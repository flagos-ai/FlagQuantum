"""Compare native control glyphs with the unchanged complete control source."""

from typing import Any

import pytest

from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.drawer.control_symbols import (
    control_register_wire_symbol,
    controlled_wire_title,
)
from tests.unit.test_register_control_predicate import (
    CASES,
    ControlCase,
    control_fixture,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
pytest.importorskip("qualtran")


@pytest.mark.parametrize("case", CASES)
def test_all_group_encodings_shapes_and_polarities(case: ControlCase) -> None:
    from qualtran import Register as SourceRegister

    spec, patterns, _ = control_fixture(case)
    for position, ((_, width, shape, _, _), dtype, pattern) in enumerate(
        zip(case, spec.qdtypes, patterns, strict=True)
    ):
        source = SourceRegister("control", dtype, shape)
        native = Register("control", RegisterType("opaque", width), shape)
        for index in native.all_idxs():
            expected = spec.wire_symbol(position, source, index).json_dict()
            glyph, attributes = control_register_wire_symbol(
                native, index, pattern.values
            )
            assert (glyph, attributes) == (
                expected["symb_cls"],
                expected["symb_attributes"],
            )


def test_symbolic_activation_and_out_of_range_metadata_remain_presentable() -> None:
    from qualtran import CtrlSpec, QBit, QInt
    from qualtran import Register as SourceRegister
    from qualtran.symbolics import Shaped

    symbolic = CtrlSpec(cvs=Shaped((10,)))
    expected = symbolic.wire_symbol(0, SourceRegister("q", QBit())).json_dict()
    assert control_register_wire_symbol(
        Register("q", RegisterType("bit", 1)), (), None
    ) == (expected["symb_cls"], expected["symb_attributes"])
    signed = CtrlSpec(QInt(8), cvs=255)
    expected = signed.wire_symbol(0, SourceRegister("q", QInt(8))).json_dict()
    assert control_register_wire_symbol(
        Register("q", RegisterType("signed", 8)), (), (255,)
    ) == (expected["symb_cls"], expected["symb_attributes"])


@pytest.mark.parametrize("text", ["", "X", "H†", "very long custom child"])
def test_title_delegation_uses_source_wrapper(text: str) -> None:
    from qualtran import Controlled, CtrlSpec
    from qualtran.bloqs.for_testing import TestAtom
    from qualtran.drawing import Text

    # Replace only the child presentation callback. The controlled wrapper's
    # original delegation and default fontsize execute unchanged.
    source: Any = type(
        "TitledAtom", (TestAtom,), {"wire_symbol": lambda *args: Text(text)}
    )()
    expected = Controlled(source, CtrlSpec()).wire_symbol(None).json_dict()
    attributes: dict[str, str | int | bool] = {"text": text, "fontsize": 17}
    assert controlled_wire_title("Text", attributes) == (
        expected["symb_cls"],
        expected["symb_attributes"],
    )
    assert attributes == {"text": text, "fontsize": 17}


def test_non_text_title_is_rejected() -> None:
    with pytest.raises(ValueError, match="Text"):
        controlled_wire_title("Circle", {"filled": True})
