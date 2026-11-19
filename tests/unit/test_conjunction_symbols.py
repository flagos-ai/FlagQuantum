"""Conjunction callbacks compare directly with frozen symbolic source metadata."""

from dataclasses import fields
from functools import partial
from itertools import product
from typing import Any

import pytest

try:
    import cirq
    import sympy
except ModuleNotFoundError:
    pytest.skip(
        "Conjunction reference dependencies are absent", allow_module_level=True
    )

from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.drawer.conjunction_symbols import (
    and_diagram_symbols,
    and_wire_symbol,
    multi_and_wire_symbol,
)
from flagquantum.drawer.register_score import RegisterStep, layout_register_score
from flagquantum.drawer.register_score_data import register_score_data

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
pytest.importorskip("qualtran")


def native_symbol(symbol: Any) -> tuple[str, dict[str, Any]]:
    data = symbol.json_dict()
    return data["symb_cls"], data["symb_attributes"]


@pytest.mark.parametrize("erase", [False, True])
def test_atomic_symbolic_controls_and_diagram(erase: bool) -> None:
    from qualtran.bloqs.mcmt import And

    a, b = sympy.symbols("a b")
    patterns = [
        *product((0, 1), repeat=2),
        (a, 0),
        (1, b),
        (a, b),
        (sympy.Integer(0), 1),
    ]
    bit = quantum_bit_type()
    for values in patterns:
        source = And(*values, uncompute=erase)
        ports = tuple(source.signature)
        callback = partial(and_wire_symbol, values=values)
        step = RegisterStep("And")
        assert callback(step, None, None) == native_symbol(source.wire_symbol(None))
        for old in ports:
            register = Register(old.name, bit, old.shape, RegisterSide(old.side.value))
            for index in old.all_idxs():
                assert callback(step, register, index) == native_symbol(
                    source.wire_symbol(old, index)
                )
        assert and_diagram_symbols(values, uncompute=erase) == tuple(
            cirq.circuit_diagram_info(source).wire_symbols
        )


@pytest.mark.parametrize("erase", [False, True])
def test_multi_symbolic_controls_junk_and_score_callback(erase: bool) -> None:
    from qualtran.bloqs.mcmt import MultiAnd

    a = sympy.Symbol("a")
    bit = quantum_bit_type()
    for values in ((1, 0, 1), (a, 0, 1), (1, a, 0, 1)):
        source = MultiAnd(values)
        callback = partial(multi_and_wire_symbol, values=values)
        for old in source.signature:
            register = Register(
                old.name,
                bit,
                old.shape,
                (
                    RegisterSide.LEFT
                    if erase and old.name != "ctrl"
                    else RegisterSide(old.side.value)
                ),
            )
            reference = old.adjoint() if erase and old.name != "ctrl" else old
            for index in old.all_idxs():
                assert callback(
                    RegisterStep("MultiAnd"), register, index
                ) == native_symbol(source.wire_symbol(reference, index))
        controls = tuple(f"ctrl[{i}]" for i in range(len(values)))
        junk = tuple(f"junk[{i}]" for i in range(len(values) - 2))
        registers = (
            Register("ctrl", bit, (len(values),)),
            Register("junk", bit, (len(values) - 2,), side=RegisterSide.RIGHT),
            Register("target", bit, side=RegisterSide.RIGHT),
        )
        step = RegisterStep(
            "MultiAnd",
            controls,
            (*controls, *junk, "target"),
            input_registers=registers[:1],
            output_registers=registers,
        )
        score = layout_register_score(
            dict.fromkeys((*controls, *junk, "target"), 1), controls, (step,)
        )
        data = register_score_data(score, register_wire_symbol=callback)
        from qualtran.drawing import get_musical_score_data

        expected = get_musical_score_data(source)
        assert data["hlines"] == [line.json_dict() for line in expected.hlines]
        assert data["vlines"] == [line.json_dict() for line in expected.vlines]
        assert [
            {key: value for key, value in port.items() if key != "ident"}
            for port in data["soqs"]
        ] == [
            {key: value for key, value in port.json_dict().items() if key != "ident"}
            for port in expected.soqs
        ]


def test_symbolic_values_retain_existing_signature_and_resource_owners() -> None:
    from qualtran.bloqs.mcmt import And, MultiAnd
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from examples.algorithms.and_notebook import and_signature
    from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
    from flagquantum.compiler._component_trace.resource_generalization import (
        ConjunctionCall,
    )

    a = sympy.Symbol("a")
    for values in ((a, 1), (0, a), (1, a, 0), (a, 0, 1, a)):
        source = And(*values) if len(values) == 2 else MultiAnd(values)
        signature = and_signature(len(values), multi=len(values) > 2)
        assert [(r.name, r.shape_symbolic, r.side.value) for r in signature] == [
            (r.name, r.shape_symbolic, r.side.value) for r in source.signature
        ]
        for erase in (False, True):
            original = source.adjoint() if erase else source
            for legacy in (False, True):
                actual = query_qec_gate_cost(
                    ConjunctionCall(values, erase), legacy_shims=legacy
                )
                expected = get_cost_value(original, QECGatesCost(legacy_shims=legacy))
                assert all(
                    getattr(actual, field.name) == getattr(expected, field.name)
                    for field in fields(actual)
                )


@pytest.mark.parametrize(
    "side", [RegisterSide.LEFT, RegisterSide.RIGHT, RegisterSide.THRU]
)
def test_arbitrary_directional_metadata_and_missing_control_index(
    side: RegisterSide,
) -> None:
    from qualtran import QBit, Side
    from qualtran import Register as SourceRegister
    from qualtran.bloqs.mcmt import And, MultiAnd

    step = RegisterStep("And")
    bit = quantum_bit_type()
    for name, index in (("target", ()), ("junk", (2, 3))):
        old = SourceRegister(name, QBit(), side=Side(side.value))
        reg = Register(name, bit, side=side)
        callback = (
            partial(and_wire_symbol, values=(1, 0))
            if name == "target"
            else partial(multi_and_wire_symbol, values=(1, 0, 1))
        )
        source = And() if name == "target" else MultiAnd((1, 0, 1))
        assert callback(step, reg, index) == native_symbol(
            source.wire_symbol(old, index)
        )
    for wire_callback in (
        and_wire_symbol,
        partial(multi_and_wire_symbol, values=(1, 0, 1)),
    ):
        with pytest.raises(ValueError, match="element index"):
            wire_callback(step, Register("ctrl", bit, (2,)), None)


def test_unknown_symbolic_control_count_keeps_source_presentation_refusal() -> None:
    from qualtran.bloqs.mcmt import MultiAnd
    from qualtran.symbolics import HasLength

    n = sympy.Symbol("n", integer=True, positive=True)
    source = MultiAnd(HasLength(n))
    bit = quantum_bit_type()
    callback = partial(multi_and_wire_symbol, values=n)
    step = RegisterStep("MultiAnd")
    for old in source.signature:
        register = Register(
            old.name, bit, old.shape_symbolic, RegisterSide(old.side.value)
        )
        if old.name == "ctrl":
            with pytest.raises(ValueError, match="symbolic"):
                source.wire_symbol(old, (0,))
            with pytest.raises(ValueError, match="no concrete control values"):
                callback(step, register, (0,))
        else:
            assert callback(step, register, ()) == native_symbol(
                source.wire_symbol(old, ())
            )


def test_concrete_control_count_uses_positive_source_normalization() -> None:
    from qualtran.bloqs.mcmt import MultiAnd
    from qualtran.symbolics import HasLength

    source = MultiAnd(HasLength(3))
    register = Register("ctrl", quantum_bit_type(), (3,))
    for i in range(3):
        assert multi_and_wire_symbol(
            RegisterStep("MultiAnd"), register, (i,), values=3
        ) == native_symbol(source.wire_symbol(source.signature.get_left("ctrl"), (i,)))
