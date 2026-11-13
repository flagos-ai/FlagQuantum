"""Mixed storage words preserve the locked score's logical lane semantics."""

from __future__ import annotations

import json
from collections import Counter
from hashlib import sha1
from math import prod
from pathlib import Path
from typing import Any

import pytest

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import (
    classical_bit_type,
    classical_unsigned_register_type,
    opaque_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.register_lanes import RegisterLaneManager
from flagquantum.drawer.register_score import (
    RegisterScore,
    RegisterStep,
    layout_register_score,
)
from flagquantum.drawer.register_score_data import (
    export_register_score_json,
    register_score_data,
)
from flagquantum.drawer.register_score_html import export_register_score_html
from flagquantum.drawer.register_score_mpl import draw_register_score
from tests.unit.test_drawer_component_score import assert_source_geometry
from tests.unit.test_drawer_custom_symbols import assert_pixels
from tests.unit.test_drawer_register_symbols import native_symbols, source_symbols
from tests.unit.test_program_classical_registers import identity

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    raw = (
        Path(__file__).resolve().parents[3]
        / "ftqc-reference-env/qualtran/drawing/musical_score.py"
    ).read_bytes()
    assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == (
        "3863bb66cdba17c4654657e275a8485d94b8da75"
    )


def mixed_fixture(
    kind: str, shape: tuple[int, ...]
) -> tuple[ProgramRecord | RegisterScore, Any]:
    from attrs import frozen
    from qualtran import Bloq, BloqBuilder, CBit, CUInt, QAny, QBit
    from qualtran import Register as OriginalRegister
    from qualtran import Signature as OriginalSignature

    native_types = {
        "qzero": opaque_register_type(0),
        "czero": classical_unsigned_register_type(0),
        "cbit": classical_bit_type(),
        "cword": classical_unsigned_register_type(3),
    }
    original_types = {
        "qzero": QAny(0),
        "czero": CUInt(0),
        "cbit": CBit(),
        "cword": CUInt(3),
    }
    signature = Signature(
        (Register("q", quantum_bit_type()), Register("data", native_types[kind], shape))
    )
    if kind in {"czero", "cword"}:
        keys = ("q", *(f"data{idx}" for idx in signature[1].all_idxs()))
        widths = {key: int(signature[1].bitsize) for key in keys[1:]}
        widths["q"] = 1
        metadata = tuple(signature)
        record: ProgramRecord | RegisterScore = layout_register_score(
            widths,
            keys,
            (
                RegisterStep(
                    "mixed",
                    keys,
                    keys,
                    input_registers=metadata,
                    output_registers=metadata,
                ),
            )
            * 2,
            input_registers=metadata,
            output_registers=metadata,
        )
    else:
        leaf = identity("mixed", tuple(signature))
        builder = ComponentProgramBuilder(signature, (leaf,))
        values = builder.inputs
        for _ in range(2):
            values = builder.call_dict("mixed", values)
        record = builder.finalize(values)
    original_registers = (
        OriginalRegister("q", QBit()),
        OriginalRegister("data", original_types[kind], shape=shape),
    )

    def original_signature(self: Any) -> Any:
        return OriginalSignature(original_registers)

    mixed_identity = frozen(
        type("MixedIdentity", (Bloq,), {"signature": property(original_signature)})
    )

    bb, inputs = BloqBuilder.from_signature(OriginalSignature(original_registers))
    for _ in range(2):
        inputs = bb.add_d(mixed_identity(), **inputs)
    return record, bb.finalize(**inputs)


@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (0,), (0, 2)])
@pytest.mark.parametrize("kind", ["qzero", "czero", "cbit", "cword"])
def test_mixed_arrays_match_full_source_geometry_callbacks_and_exports(
    monkeypatch: pytest.MonkeyPatch, kind: str, shape: tuple[int, ...]
) -> None:
    from qualtran.drawing import get_musical_score_data

    record, original = mixed_fixture(kind, shape)
    before = hash(record)
    expected_events: list[Any] = []
    actual_events: list[Any] = []
    source_symbols(monkeypatch, original, expected_events)
    manager = RegisterLaneManager()
    score = (
        component_register_score(record, lane_manager=manager)
        if isinstance(record, ProgramRecord)
        else record
    )
    actual = register_score_data(
        score, register_wire_symbol=native_symbols(actual_events)
    )
    expected = get_musical_score_data(original)
    assert_source_geometry(actual, expected, ordered_ports=True)
    assert actual_events == expected_events
    assert hash(record) == before
    assert len(score.lines) == 1 + (0 if 0 in shape else prod(shape))
    assert {
        line.flavor for line in score.lines if line.register.startswith("data")
    } <= {"HLineFlavor.QUANTUM" if kind == "qzero" else "HLineFlavor.CLASSICAL"}
    if isinstance(record, ProgramRecord):
        assert sorted((line.json_dict() for line in manager.hlines), key=str) == sorted(
            actual["hlines"], key=str
        )
    assert json.loads(export_register_score_json(actual)) == actual
    page = export_register_score_html(actual)
    payload = page.split('type="application/json">', 1)[1].split("</script>", 1)[0]
    assert json.loads(payload)["sequential"] == actual


@pytest.mark.parametrize("kind", ["qzero", "czero", "cbit", "cword"])
@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (0,), (0, 2)])
def test_mixed_matplotlib_matches_source_after_boundary_text_adaptation(
    monkeypatch: pytest.MonkeyPatch, kind: str, shape: tuple[int, ...]
) -> None:
    import attrs
    from matplotlib import pyplot as plt
    from qualtran.drawing.musical_score import (
        Text,
        draw_musical_score,
        get_musical_score_data,
    )

    record, original = mixed_fixture(kind, shape)
    source_symbols(monkeypatch, original, [])
    actual = register_score_data(
        (
            component_register_score(record)
            if isinstance(record, ProgramRecord)
            else record
        ),
        register_wire_symbol=native_symbols([]),
    )
    expected = get_musical_score_data(original)
    by_position = {(p["seq_x"], p["y"]): p for p in actual["soqs"]}
    expected.soqs = [
        (
            attrs.evolve(
                port,
                symb=Text(
                    by_position[(port.rpos.seq_x, port.rpos.y)]["symb_attributes"][
                        "text"
                    ],
                    fontsize=8,
                ),
            )
            if port.ident.startswith("dang,")
            else port
        )
        for port in expected.soqs
    ]
    positions = [(port["seq_x"], port["y"]) for port in actual["soqs"]]
    original_positions = [(port.rpos.seq_x, port.rpos.y) for port in expected.soqs]
    assert positions == original_positions
    assert_source_geometry(actual, expected, ordered_ports=True)
    with plt.rc_context({"figure.dpi": 100, "font.family": "DejaVu Sans"}):
        reference, _ = draw_musical_score(expected)
        native, _ = draw_register_score(actual)
        try:
            assert_pixels(native, reference)
        finally:
            plt.close(reference)
            plt.close(native)


def test_explicit_classical_override_and_hand_authored_quantum_defaults() -> None:
    score = layout_register_score({"data": 0}, ("data",), ())
    assert score.lines[0].flavor == "HLineFlavor.QUANTUM"
    assert register_score_data(score)["hlines"][0]["flavor"] == "HLineFlavor.QUANTUM"
    assert (
        register_score_data(score, classical={"data"})["hlines"][0]["flavor"]
        == "HLineFlavor.CLASSICAL"
    )
    with pytest.raises(ValueError, match="Unknown classical"):
        register_score_data(score, classical={"missing"})


@pytest.mark.parametrize("width", [-1, True, 1.5])
def test_invalid_layout_width_fails_before_lane_mutation(width: Any) -> None:
    manager = RegisterLaneManager()
    with pytest.raises(ValueError, match="nonnegative"):
        layout_register_score({"bad": width}, ("bad",), (), lane_manager=manager)
    assert manager.hlines == set()
    assert manager.available == []


def test_measured_parity_and_discard_preserve_storage_flavor_on_reused_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np
    from qualtran import BloqBuilder, QBit, Side
    from qualtran import Register as OriginalRegister
    from qualtran import Signature as OriginalSignature
    from qualtran.bloqs.basic_gates import MeasureX
    from qualtran.bloqs.basic_gates.discard import Discard
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import Parity
    from qualtran.drawing import get_musical_score_data

    from flagquantum.compiler.registers.component_classical import (
        discard_word_function,
        parity_word_function,
    )
    from flagquantum.compiler.registers.component_measurement import (
        measure_x_word_function,
    )
    from flagquantum.compiler.registers.register_binding import bind_register_program

    signature = Signature(
        (
            Register("q", quantum_bit_type(), side=RegisterSide.LEFT),
            Register("idle", quantum_bit_type()),
        )
    )
    builder = ComponentProgramBuilder(
        signature,
        (
            measure_x_word_function("measure"),
            parity_word_function("parity", 1),
            discard_word_function("discard"),
        ),
    )
    bit = builder.call_dict("measure", {"q": builder.inputs["q"]})["c"]
    parity = builder.call_dict("parity", {"x": np.array([bit], dtype=object)})
    builder.call_dict("discard", {"c": np.asarray(parity["x"]).item()})
    builder.call_dict("discard", {"c": parity["parity"]})
    record = builder.finalize({"idle": builder.inputs["idle"]})
    original_builder, inputs = BloqBuilder.from_signature(
        OriginalSignature(
            [
                OriginalRegister("q", QBit(), side=Side.LEFT),
                OriginalRegister("idle", QBit()),
            ]
        )
    )
    outcome = original_builder.add(MeasureX(), q=inputs["q"])
    output = original_builder.add_d(Parity(1), x=np.array([outcome], dtype=object))
    original_builder.add(Discard(), c=output["x"].item())
    original_builder.add(Discard(), c=output["parity"])
    original = original_builder.finalize(idle=inputs["idle"])
    source_symbols(monkeypatch, original, [])
    before = bind_register_program(record, ((0,), (1,)), clean=(2,))
    score = component_register_score(record)
    actual = register_score_data(score, register_wire_symbol=native_symbols([]))
    assert_source_geometry(actual, get_musical_score_data(original))
    assert {line.flavor for line in score.lines if line.lane == 0} == {
        "HLineFlavor.QUANTUM",
        "HLineFlavor.CLASSICAL",
    }
    assert [line["flavor"] for line in actual["hlines"]].count(
        "HLineFlavor.CLASSICAL"
    ) == 2
    assert bind_register_program(record, ((0,), (1,)), clean=(2,)) == before
    assert Counter(g.name for g in before.circuit.instructions) == {
        "h": 1,
        "measure": 1,
        "classical_xor": 1,
    }
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    model = get_cost_value(MeasureX(), QECGatesCost())
    assert model.measurement == 1 and model.total_t_count() == 0


@pytest.mark.parametrize("kind", ["czero", "cword"])
def test_classical_word_metadata_does_not_expand_core_executable_admission(
    kind: str,
) -> None:
    from flagquantum.core.ir.program import verify_program_record
    from flagquantum.core.ir.program.register_verifier import register_port_types
    from flagquantum.core.ir.program.registers import RegisterSide

    width = 0 if kind == "czero" else 3
    leaf = identity(
        "main", (Register("word", classical_unsigned_register_type(width)),)
    )
    with pytest.raises(ValueError, match="unsupported executable"):
        register_port_types(leaf, RegisterSide.LEFT)
    assert verify_program_record(ProgramRecord((leaf,)))
