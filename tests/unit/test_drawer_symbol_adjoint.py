"""Locked glyph adjoints preserve native display and physical graph semantics."""

import hashlib
from collections import Counter
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.component_adjoint import adjoint_component_program
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.register_score import RegisterStep
from flagquantum.drawer.register_score_data import (
    PortSide,
    SymbolClass,
    adjoint_wire_symbol,
    register_score_data,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_drawer_component_score import assert_source_geometry
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_reference() -> None:
    root = Path(__file__).resolve().parents[3] / "ftqc-reference-env"
    blobs = {
        "qualtran/drawing/musical_score.py": "3863bb66cdba17c4654657e275a8485d94b8da75",
        "qualtran/_infra/adjoint.py": "0601a6248eec488301b31dd7c800a6ba14a7fb27",
        "qualtran/bloqs/basic_gates/s_gate.py": "6111677a1e55ccab022e988598366a0f1bdcfc94",
        "qualtran/bloqs/basic_gates/t_gate.py": "77cc66bbed4f0fcf35d2565e338075890b0889f6",
    }
    blobs["qualtran/bloqs/basic_gates/z_basis.py"] = (
        "48231b5d507f076ed3ea95a4d8486f375cca7d01"
    )
    for name, expected in blobs.items():
        raw = (root / name).read_bytes()
        assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == expected


@pytest.mark.parametrize("kind", ["Text", "TextBox"])
@pytest.mark.parametrize(
    "text", ["", "S", "S†", "†S", "†S†", "S††", "†", "a†b", "S† ", "∧"]
)
def test_text_adjoint_matches_source_including_dagger_stripping(
    kind: SymbolClass, text: str
) -> None:
    source = pytest.importorskip("qualtran.drawing.musical_score")
    attributes: dict[str, str | int | bool] = {"text": text}
    if kind == "Text":
        attributes["fontsize"] = 13
    original = attributes.copy()
    expected = getattr(source, kind)(**attributes)
    actual = kind, attributes
    for _ in range(3):
        previous = actual[1].copy()
        previous_attributes = actual[1]
        actual = adjoint_wire_symbol(*actual)
        expected = expected.adjoint()
        assert actual == (
            expected.json_dict()["symb_cls"],
            expected.json_dict()["symb_attributes"],
        )
        assert previous_attributes == previous
        assert actual[1] is not previous_attributes
    assert attributes == original


@pytest.mark.parametrize(
    "kind,attributes",
    [
        ("RarrowTextBox", {"text": "†work†"}),
        ("LarrowTextBox", {"text": "work"}),
        ("Circle", {"filled": True}),
        ("Circle", {"filled": False}),
        ("ModPlus", {}),
    ],
)
def test_directional_and_self_adjoint_symbols_match_source(
    kind: SymbolClass, attributes: dict[str, str | int | bool]
) -> None:
    source = pytest.importorskip("qualtran.drawing.musical_score")
    expected = getattr(source, kind)(**attributes)
    actual = kind, attributes
    for _ in range(3):
        previous_attributes = actual[1]
        previous = previous_attributes.copy()
        actual = adjoint_wire_symbol(*actual)
        expected = expected.adjoint()
        assert actual == (
            expected.json_dict()["symb_cls"],
            expected.json_dict()["symb_attributes"],
        )
        assert previous_attributes == previous
        assert actual[1] is not previous_attributes


@pytest.mark.parametrize("text", [1, True])
def test_invalid_text_is_rejected_without_mutating_attributes(text: int | bool) -> None:
    attributes: dict[str, str | int | bool] = {"text": text}
    with pytest.raises(TypeError, match="string text"):
        adjoint_wire_symbol("TextBox", attributes)
    assert attributes == {"text": text}


def test_missing_text_reports_the_required_attribute() -> None:
    with pytest.raises(KeyError, match="text"):
        adjoint_wire_symbol("Text", {"fontsize": 13})


@pytest.mark.parametrize("inverse", [False, True])
def test_component_adjoint_callback_gates_matrix_phase_and_append(
    inverse: bool,
) -> None:
    from qualtran import BloqBuilder
    from qualtran.bloqs.basic_gates import SGate, TGate
    from qualtran.drawing import get_musical_score_data
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    registers = (Register("q", quantum_bit_type()),)
    definitions = tuple(
        bit_gate_function(name, registers, CircuitIR(1, (Instruction(name, (0,)),)))
        for name in ("s", "sdg", "t", "tdg")
    )
    signature = Signature((*registers, Register("spectator", quantum_bit_type())))
    builder = ComponentProgramBuilder(signature, definitions)
    output = builder.call_dict("s", {"q": builder.inputs["q"]})
    output = builder.call_dict("t", output)
    record = builder.finalize({**output, "spectator": builder.inputs["spectator"]})
    original, inputs = BloqBuilder.from_signature(
        pytest.importorskip("qualtran").Signature.build(q=1, spectator=1)
    )
    q = original.add(SGate(), q=inputs["q"])
    q = original.add(TGate(), q=q)
    reference = original.finalize(q=q, spectator=inputs["spectator"])
    if inverse:
        record = adjoint_component_program(record, {"s": "sdg", "t": "tdg"})
        reference = reference.adjoint()
    bound = bind_register_program(record, ((0,), (1,)))
    before = bound.circuit.instructions

    def symbols(
        step: RegisterStep, register: str | None, side: PortSide | None
    ) -> tuple[SymbolClass, dict[str, str | int | bool]]:
        if register is None:
            return "Text", {"text": "", "fontsize": 10}
        attributes: dict[str, str | int | bool] = {"text": step.name[0].upper()}
        if step.name.endswith("dg"):
            return adjoint_wire_symbol("TextBox", attributes)
        return "TextBox", attributes

    data = register_score_data(component_register_score(record), wire_symbol=symbols)
    assert_source_geometry(data, get_musical_score_data(reference))
    assert bind_register_program(record, ((0,), (1,))).circuit.instructions == before
    assert_source(
        before,
        reference_arithmetic_gates(
            reference,
            {"q": (0,), "spectator": (1,)},
            (),
            {},
            cirq_style=True,
            keep_exchange_leaves=True,
        ),
    )
    assert Counter(g.name for g in before) == (
        {"tdg": 1, "sdg": 1} if inverse else {"s": 1, "t": 1}
    )
    expected = torch.as_tensor(reference.tensor_contract(), dtype=torch.complex128)
    columns = torch.stack(
        [forced_branch(v, before, ()) for v in torch.eye(4, dtype=torch.complex128)],
        dim=1,
    )
    torch.testing.assert_close(columns, expected, rtol=0, atol=1e-12)
    initial = torch.tensor([1 + 2j, 0, 3 - 4j, 0], dtype=torch.complex128)
    initial /= initial.norm()
    actual = (
        fq.Circuit(2, inputs=initial, dtype=torch.complex128)
        .append(FieldOperation(2, before))
        .state()
        .flatten()
    )
    torch.testing.assert_close(actual, expected @ initial, rtol=0, atol=1e-12)
    assert torch.count_nonzero(actual[1::2]) == 0
    model = get_cost_value(reference, QECGatesCost())
    assert model.total_t_count() == 1 and model.clifford == 1 and model.measurement == 0
    assert get_cost_value(reference, QubitCount()) == 2


@pytest.mark.parametrize("kind", ["IntState", "IntEffect", "TGate"])
@pytest.mark.parametrize("connector", [False, True])
def test_generic_source_adjoint_delegation_matches_native_glyph(
    kind: str, connector: bool
) -> None:
    from qualtran._infra.adjoint import Adjoint

    gates = pytest.importorskip("qualtran.bloqs.basic_gates")
    original = getattr(gates, kind)() if kind == "TGate" else getattr(gates, kind)(5, 3)
    register = None if connector else next(iter(original.signature))
    glyph = original.wire_symbol(register)
    expected = (
        Adjoint(original)
        .wire_symbol(None if register is None else register.adjoint())
        .json_dict()
    )
    metadata = glyph.json_dict()
    actual = adjoint_wire_symbol(metadata["symb_cls"], metadata["symb_attributes"])
    assert actual == (expected["symb_cls"], expected["symb_attributes"])
