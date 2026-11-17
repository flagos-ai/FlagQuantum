"""Match the locked SELECT and LCU wire glyphs through the native Drawer."""

from functools import partial

import pytest

from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.drawer.lcu_block_symbols import (
    lcu_block_wire_symbol,
    select_block_wire_symbol,
)
from flagquantum.drawer.register_score import RegisterStep, layout_register_score
from flagquantum.drawer.register_score_data import register_score_data

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("mode", ("select", "lcu", "negative", "positive"))
def test_lcu_wire_symbols_retains_locked_name_dispatch(mode: str) -> None:
    from attrs import asdict
    from qualtran import CtrlSpec, QBit
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.block_encoding.lcu_block_encoding import (
        _lcu_block,
        _select_block,
    )

    value = {"negative": 0, "positive": 1}.get(mode)
    source = _select_block() if mode == "select" else _lcu_block()
    if value is not None:
        source = source.controlled(CtrlSpec(cvs=value))
    callback = (
        select_block_wire_symbol
        if mode == "select"
        else partial(lcu_block_wire_symbol, control_value=value)
    )
    step = RegisterStep("block")
    for name in (None, "ctrl", "control", "system"):
        native = None if name is None else Register(name, quantum_bit_type())
        locked = None if name is None else SourceRegister(name, QBit())
        symbol = source.wire_symbol(locked)
        assert callback(step, native, None) == (type(symbol).__name__, asdict(symbol))


def test_lcu_wire_symbols_render_port_metadata() -> None:
    registers = tuple(
        Register(name, quantum_bit_type()) for name in ("control", "ctrl")
    )
    names = tuple(reg.name for reg in registers)
    score = layout_register_score(
        dict.fromkeys(names, 1),
        names,
        (
            RegisterStep(
                "lcu",
                names,
                names,
                input_registers=registers,
                output_registers=registers,
            ),
        ),
        input_registers=registers,
        output_registers=registers,
    )
    data = register_score_data(
        score, register_wire_symbol=partial(lcu_block_wire_symbol, control_value=0)
    )
    glyphs = [
        (port["symb_cls"], port["symb_attributes"])
        for port in data["soqs"]
        if not port["ident"].startswith("dang,")
    ]
    assert glyphs == [("Circle", {"filled": False}), ("TextBox", {"text": "B[H]"})]
