"""Compare the complete repeated-child presentation with the frozen source."""

from functools import partial
from hashlib import sha1
from pathlib import Path
from typing import Any

import pytest
from attrs import asdict

from flagquantum.core.ir.program.register_types import opaque_register_type
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.drawer.on_each_symbols import on_each_wire_symbol
from flagquantum.drawer.register_score import RegisterStep, layout_register_score
from flagquantum.drawer.register_score_data import SymbolClass, register_score_data

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    from qualtran.bloqs.basic_gates import on_each

    raw = Path(on_each.__file__).read_bytes()
    assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == (
        "0d8a190709e5465a7305858a1ba895ab284c28b4"
    )


@pytest.mark.parametrize(
    "glyph", ("Text", "TextBox", "Circle", "ModPlus", "RarrowTextBox", "LarrowTextBox")
)
@pytest.mark.parametrize("text", ("", "long child label"))
@pytest.mark.parametrize("width", (0, 3, "n"))
def test_source_callback_forwarding_and_fallback(
    glyph: SymbolClass, text: str, width: int | str
) -> None:
    import sympy
    from qualtran import Bloq, QAny, Side
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.basic_gates import OnEach, XGate
    from qualtran.drawing import musical_score

    size = (
        sympy.Symbol("n", integer=True, positive=True)
        if isinstance(width, str)
        else width
    )
    events: list[tuple[object, object]] = []

    class Child:
        @property
        def signature(self) -> Any:
            return XGate().signature

        def wire_symbol(self, reg: Any, idx: tuple[int, ...] = ()) -> Any:
            events.append((reg, idx))
            cls = getattr(musical_score, glyph)
            if glyph == "Circle":
                return cls(filled=True)
            if glyph == "ModPlus":
                return cls()
            return cls(text, fontsize=19) if glyph == "Text" else cls(text)

        def __str__(self) -> str:
            return "long child label"

    child_type = type(
        "DisplayChild",
        (Bloq,),
        {
            "signature": Child.signature,
            "wire_symbol": Child.wire_symbol,
            "__str__": Child.__str__,
        },
    )
    original = OnEach(size, child_type())
    step = RegisterStep(str(original))
    attributes: dict[str, str | int | bool] = (
        {"filled": True}
        if glyph == "Circle"
        else {} if glyph == "ModPlus" else {"text": text}
    )
    seen: list[tuple[object, object]] = []

    def child_symbol(
        display: RegisterStep, reg: Register | None, idx: tuple[int, ...] | None
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        assert display is step
        seen.append((reg, idx))
        return glyph, attributes

    if glyph == "Text":
        attributes["fontsize"] = 19
    before = dict(attributes)
    for side in (None, RegisterSide.LEFT, RegisterSide.RIGHT, RegisterSide.THRU):
        for index in ((), (1, 2)):
            reg = (
                None
                if side is None
                else Register("q", opaque_register_type(3), shape=(2, 3), side=side)
            )
            foreign = (
                None
                if side is None
                else SourceRegister("q", QAny(3), shape=(2, 3), side=Side[side.name])
            )
            expected = original.wire_symbol(foreign, index)
            actual = on_each_wire_symbol(
                step, reg, index, width=size, child_symbol=child_symbol
            )
            assert actual == (type(expected).__name__, asdict(expected))
            assert events[-1] == (foreign, index) and seen[-1] == (reg, index)
    assert attributes == before


def test_existing_register_score_uses_owned_callback() -> None:
    register = Register("q", opaque_register_type(3))
    step = RegisterStep(
        "S⨂3", ("q",), ("q",), input_registers=(register,), output_registers=(register,)
    )

    def child_symbol(
        display: RegisterStep, reg: Register | None, index: tuple[int, ...] | None
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        return ("Text", {"text": ""}) if reg is None else ("TextBox", {"text": "S"})

    score = layout_register_score({"q": 3}, ("q",), (step,))
    result = register_score_data(
        score,
        register_wire_symbol=partial(
            on_each_wire_symbol, width=3, child_symbol=child_symbol
        ),
    )
    assert any(port["symb_attributes"].get("text") == "S⨂3" for port in result["soqs"])
    assert result["vlines"][0]["label"]["text"] == ""
