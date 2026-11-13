"""Typed wire-symbol callbacks retain the frozen register/index call cadence."""

from dataclasses import replace
from hashlib import sha1
from pathlib import Path
from typing import Any

import pytest

from flagquantum.compiler.registers.arithmetic.addition_component import (
    addition_component_program,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.register_score import RegisterStep, layout_register_score
from flagquantum.drawer.register_score_data import (
    PortSide,
    RegisterWireSymbolSupplier,
    SymbolClass,
    register_score_data,
)
from tests.unit.test_component_tensor_nodes import fixture
from tests.unit.test_drawer_component_score import assert_source_geometry

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]

Event = tuple[str | None, str | None, tuple[int, ...] | None, tuple[int, ...] | None]


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    path = (
        Path(__file__).resolve().parents[3]
        / "ftqc-reference-env/qualtran/drawing/musical_score.py"
    )
    raw = path.read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "3863bb66cdba17c4654657e275a8485d94b8da75"
    )


def symbol_text(name: str, side: str, index: tuple[int, ...]) -> str:
    return f"{name}/{side}/{index}"


def source_symbols(
    monkeypatch: pytest.MonkeyPatch,
    original: Any,
    events: list[Event],
    numbered: bool = False,
) -> None:
    from qualtran.drawing.musical_score import Text, directional_text_box

    def wire_symbol(self: Any, reg: Any = None, idx: tuple[int, ...] = ()) -> Any:
        if reg is None:
            events.append((None, None, None, None))
            return Text("caption", fontsize=13)
        events.append((reg.name, reg.side.name, reg.shape, idx))
        text = symbol_text(reg.name, reg.side.name, idx)
        if numbered:
            text += f"#{len(events)}"
        return directional_text_box(text, reg.side)

    for cls in {type(instance.bloq) for instance in original.bloq_instances}:
        monkeypatch.setattr(cls, "wire_symbol", wire_symbol)


def native_symbols(
    events: list[Event], numbered: bool = False
) -> RegisterWireSymbolSupplier:
    def supply(
        step: RegisterStep, register: Register | None, index: tuple[int, ...] | None
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        if register is None:
            assert index is None
            events.append((None, None, None, None))
            return "Text", {"text": "caption", "fontsize": 13}
        assert index is not None
        shape = register.shape
        assert all(isinstance(n, int) for n in shape)
        events.append((register.name, register.side.name, shape, index))
        glyphs: dict[RegisterSide, SymbolClass] = {
            RegisterSide.LEFT: "RarrowTextBox",
            RegisterSide.RIGHT: "LarrowTextBox",
            RegisterSide.THRU: "TextBox",
        }
        glyph = glyphs[register.side]
        text = symbol_text(register.name, str(register.side.name), index)
        if numbered:
            text += f"#{len(events)}"
        return glyph, {"text": text}

    return supply


@pytest.mark.parametrize("width", [1, 2, 3, 5, 16, 32])
@pytest.mark.parametrize("numbered", [False, True])
def test_add_typed_register_callbacks_and_all_glyphs_match_source(
    monkeypatch: pytest.MonkeyPatch, width: int, numbered: bool
) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add
    from qualtran.drawing import get_musical_score_data

    record = addition_component_program(width)
    before = hash(record)
    original = Add(QUInt(width)).decompose_bloq()
    expected_events: list[Event] = []
    actual_events: list[Event] = []
    source_symbols(monkeypatch, original, expected_events, numbered)
    actual = register_score_data(
        component_register_score(record),
        register_wire_symbol=native_symbols(actual_events, numbered),
    )
    expected = get_musical_score_data(original)
    assert actual_events == expected_events
    assert_source_geometry(actual, expected)
    assert hash(record) == before


@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (0,), (0, 2)])
def test_same_named_array_ports_use_declarations_and_indices(
    monkeypatch: pytest.MonkeyPatch, shape: tuple[int, ...]
) -> None:
    from qualtran.drawing import get_musical_score_data

    record, original = fixture(2, shape)
    source_events: list[Event] = []
    actual_events: list[Event] = []
    source_symbols(monkeypatch, original, source_events)
    actual = register_score_data(
        component_register_score(record),
        register_wire_symbol=native_symbols(actual_events),
    )
    assert_source_geometry(actual, get_musical_score_data(original))
    assert actual_events == source_events
    # Idle boundary wires never invoke a component's wire_symbol protocol.
    assert all(event[0] in ("q", None) for event in actual_events)


def test_duplicate_through_output_callback_is_invoked_but_first_result_is_rendered() -> (
    None
):
    register = Register("same", RegisterType("bit", 1), (2,))
    score = layout_register_score(
        {"opaque-a": 1, "opaque-b": 1},
        ("opaque-a", "opaque-b"),
        (
            RegisterStep(
                "gate",
                ("opaque-a", "opaque-b"),
                ("opaque-a", "opaque-b"),
                input_registers=(register,),
                output_registers=(register,),
            ),
        ),
        input_registers=(register,),
        output_registers=(register,),
    )
    calls: list[tuple[int, ...] | None] = []

    def supply(
        step: RegisterStep, reg: Register | None, idx: tuple[int, ...] | None
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        calls.append(idx)
        if reg is None:
            return "Text", {"text": "caption", "fontsize": 10}
        assert reg is register
        return "TextBox", {"text": str(len(calls))}

    data = register_score_data(score, register_wire_symbol=supply)
    assert calls == [(0,), (1,), (0,), (1,), None]
    assert [
        p["symb_attributes"]["text"]
        for p in data["soqs"]
        if not p["ident"].startswith("dang,")
    ] == ["1", "2"]


def test_untyped_layout_synthesizes_scalar_metadata_without_parsing_display_names() -> (
    None
):
    name = "call7:not-an-array[1,2]"
    score = layout_register_score(
        {name: 3}, (name,), (RegisterStep("x", (name,), (name,)),)
    )
    seen: list[Register] = []

    def typed(
        step: RegisterStep, reg: Register | None, idx: tuple[int, ...] | None
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        if reg is None:
            return "Text", {"text": "caption", "fontsize": 13}
        seen.append(reg)
        assert idx == ()
        assert reg.name == name and reg.shape == () and reg.bitsize == 3
        return "TextBox", {"text": "custom"}

    def plain(
        step: RegisterStep, key: str | None, side: PortSide | None
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        return (
            ("Text", {"text": "caption", "fontsize": 13})
            if key is None
            else ("TextBox", {"text": "custom"})
        )

    assert register_score_data(
        score, register_wire_symbol=typed
    ) == register_score_data(score, wire_symbol=plain)
    assert len(seen) == 2


@pytest.mark.parametrize("kind", ["dual", "metadata", "connector", "callback", "empty"])
def test_typed_callback_errors_and_empty_steps(kind: str) -> None:
    register = Register("q", RegisterType("bit", 1))
    step = RegisterStep(
        "step",
        ("q",),
        ("q",),
        input_registers=(register,),
        output_registers=(register,),
    )
    score = layout_register_score({"q": 1}, ("q",), (step,))
    calls: list[str] = []

    def typed(*args: Any) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        calls.append("called")
        if kind == "callback":
            raise RuntimeError("symbol failed")
        return "TextBox", {"text": "custom"}

    def plain(*args: Any) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        raise AssertionError("unexpected display callback")

    if kind == "dual":
        with pytest.raises(ValueError, match="one wire symbol"):
            register_score_data(score, wire_symbol=plain, register_wire_symbol=typed)
        assert calls == []
    elif kind == "metadata":
        broken = replace(step, output_registers=(Register("q", register.dtype, (2,)),))
        bad = replace(score, columns=(replace(score.columns[0], step=broken),))
        with pytest.raises(ValueError, match="metadata"):
            register_score_data(bad, register_wire_symbol=typed)
        assert calls == []
    elif kind == "connector":
        with pytest.raises(ValueError, match="connector"):
            register_score_data(score, register_wire_symbol=typed)
        assert len(calls) == 3
    elif kind == "callback":
        with pytest.raises(RuntimeError, match="symbol failed"):
            register_score_data(score, register_wire_symbol=typed)
        assert len(calls) == 1
    else:
        empty = layout_register_score({}, (), (RegisterStep("empty"),))
        assert register_score_data(empty, register_wire_symbol=typed)["soqs"] == []
        assert calls == []


@pytest.mark.parametrize("shape", [(), (2,), (2, 2), (0, 2)])
def test_input_through_output_shaped_callback_metadata_matches_source(
    monkeypatch: pytest.MonkeyPatch, shape: tuple[int, ...]
) -> None:
    from qualtran import BloqBuilder, QBit, Side
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran._infra.bloq import Bloq
    from qualtran.drawing import get_musical_score_data

    def signature(self: object) -> Any:
        return SourceSignature(
            [
                SourceRegister("left", QBit(), shape=shape, side=Side.LEFT),
                SourceRegister("through", QBit(), shape=shape),
                SourceRegister("right", QBit(), shape=shape, side=Side.RIGHT),
            ]
        )

    # The optional reference package is dynamically loaded. This test-only
    # source Bloq supplies metadata, with all native callback values typed above.
    gate = type("ArrayGate", (Bloq,), {"signature": property(signature)})()
    bb, values = BloqBuilder.from_signature(gate.signature)
    original = bb.finalize(**bb.add_d(gate, **values))
    declarations = tuple(
        Register(name, RegisterType("bit", 1), shape, side)
        for name, side in (
            ("left", RegisterSide.LEFT),
            ("through", RegisterSide.THRU),
            ("right", RegisterSide.RIGHT),
        )
    )
    names = {
        register.name: tuple(
            f"identity-{register.name}-{index}" for index in register.all_idxs()
        )
        for register in declarations
    }
    left, through, right = declarations
    inputs = (*names["left"], *names["through"])
    outputs = (*names["through"], *names["right"])
    score = layout_register_score(
        dict.fromkeys((*inputs, *outputs), 1),
        inputs,
        (
            RegisterStep(
                "gate",
                inputs,
                outputs,
                input_registers=(left, through),
                output_registers=(through, right),
            ),
        ),
        input_registers=(left, through),
        output_registers=(through, right),
        outputs=outputs,
    )
    expected_events: list[Event] = []
    actual_events: list[Event] = []
    source_symbols(monkeypatch, original, expected_events, numbered=True)
    actual = register_score_data(
        score, register_wire_symbol=native_symbols(actual_events, numbered=True)
    )
    if 0 in shape:
        # The source's empty graph has no RightDangle node. Native empty scores
        # already have a defined result and invoke no symbol callbacks.
        with pytest.raises(KeyError, match="RightDangle"):
            get_musical_score_data(original)
        assert actual_events == expected_events == []
        assert actual["soqs"] == []
        assert actual["hlines"] == []
        assert actual["vlines"] == []
        return
    expected = get_musical_score_data(original)
    assert actual_events == expected_events
    assert_source_geometry(actual, expected)


def test_final_return_order_drives_stateful_callbacks_and_existing_consumers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import attrs
    import matplotlib
    from qualtran import RightDangle, Soquet
    from qualtran.drawing import get_musical_score_data

    from flagquantum.drawer.register_score_html import export_register_score_html
    from flagquantum.drawer.register_score_mpl import draw_register_score

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    record, original = fixture(2, ())
    root = record.functions[0]
    block = root.blocks[0]
    assert block.return_values is not None
    swapped = replace(
        block,
        return_values=(
            block.return_values[1],
            block.return_values[0],
            *block.return_values[2:],
        ),
    )
    record = replace(
        record, functions=(replace(root, blocks=(swapped,)), *record.functions[1:])
    )
    original = attrs.evolve(
        original,
        connections=tuple(
            (
                attrs.evolve(
                    connection,
                    right=Soquet(
                        RightDangle,
                        connection.right.reg,
                        idx=(1 - connection.right.idx[0],),
                    ),
                )
                if connection.right.binst == RightDangle
                and connection.right.reg.name == "q"
                else connection
            )
            for connection in original.connections
        ),
    )
    expected_events: list[Event] = []
    actual_events: list[Event] = []
    source_symbols(monkeypatch, original, expected_events, numbered=True)
    score = component_register_score(record)
    data = register_score_data(
        score, register_wire_symbol=native_symbols(actual_events, numbered=True)
    )
    expected = get_musical_score_data(original)
    assert actual_events == expected_events
    assert_source_geometry(data, expected)
    # Older hand-authored scores can omit return-order metadata.
    assert (
        register_score_data(
            replace(score, output_order=()), register_wire_symbol=native_symbols([])
        )["hlines"]
        == data["hlines"]
    )
    html = export_register_score_html(data)
    assert "caption" in html and "#" in html
    fig, ax = draw_register_score(data)
    try:
        rendered = {text.get_text() for text in ax.texts}
        for port in data["soqs"]:
            assert port["symb_attributes"]["text"] in rendered
    finally:
        plt.close(fig)


@pytest.mark.parametrize("typed", [False, True])
def test_false_truth_value_callable_still_supplies_connector(typed: bool) -> None:
    class Supplier:
        def __init__(self) -> None:
            self.calls = 0

        def __bool__(self) -> bool:
            return False

        def __call__(
            self,
            step: RegisterStep,
            register: Register | str | None,
            context: tuple[int, ...] | PortSide | None,
        ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
            self.calls += 1
            if register is None:
                return "Text", {"text": "caption", "fontsize": 13}
            return "TextBox", {"text": "custom"}

    supplier = Supplier()
    score = layout_register_score(
        {"q": 1}, ("q",), (RegisterStep("gate", ("q",), ("q",)),)
    )
    data = (
        register_score_data(score, register_wire_symbol=supplier)
        if typed
        else register_score_data(score, wire_symbol=supplier)
    )
    assert supplier.calls == (3 if typed else 2)
    assert data["vlines"][0]["label"] == {"text": "caption", "fontsize": 13}


def test_synthesized_metadata_preserves_input_and_output_only_sides() -> None:
    score = layout_register_score(
        {"a": 2, "b": 2, "idle": 1},
        ("a", "idle"),
        (RegisterStep("replace", ("a", "idle"), ("b", "idle")),),
        outputs=("b", "idle"),
    )
    events: list[Event] = []
    data = register_score_data(score, register_wire_symbol=native_symbols(events))
    assert events == [
        ("a", "LEFT", (), ()),
        ("idle", "THRU", (), ()),
        ("b", "RIGHT", (), ()),
        ("idle", "THRU", (), ()),
        (None, None, None, None),
    ]
    assert {port["symb_cls"] for port in data["soqs"]} == {
        "Text",
        "RarrowTextBox",
        "LarrowTextBox",
        "TextBox",
    }
