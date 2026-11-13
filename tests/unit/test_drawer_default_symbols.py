"""Default Core metadata symbols follow the frozen upstream default method."""

import csv
from dataclasses import replace
from hashlib import sha1
from pathlib import Path
from typing import Any

import pytest

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.register_score import (
    RegisterScore,
    RegisterStep,
    layout_register_score,
)
from flagquantum.drawer.register_score_data import (
    export_register_score_json,
    register_score_data,
)
from flagquantum.drawer.register_score_mpl import draw_register_score
from tests.unit.test_drawer_component_score import assert_source_geometry
from tests.unit.test_drawer_custom_symbols import assert_pixels
from tests.unit.test_program_classical_registers import identity

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    path = (
        Path(__file__).resolve().parents[3]
        / "ftqc-reference-env/qualtran/_infra/bloq.py"
    )
    raw = path.read_bytes()
    # Check the fixed inventory instead of modifying the upstream default method.
    inventory = path.parents[2].parent / "FlagQuantum-Qualtran-Kickmix-移植进度.csv"
    with inventory.open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    expected = next(
        row["源对象"] for row in rows if row["源文件"] == "qualtran/_infra/bloq.py"
    )
    assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == expected


def default_fixture(
    shape: tuple[int, ...], title: str, classical: bool
) -> tuple[RegisterScore, Any]:
    from attrs import frozen
    from qualtran import Bloq, BloqBuilder, CBit, QBit
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature

    dtype = RegisterType("bit", 0, 1) if classical else RegisterType("bit", 1)
    # Long identity prefixes and array brackets must not leak into local labels.
    signature = Signature(
        (Register("data", dtype, shape), Register("idle", RegisterType("bit", 1)))
    )
    leaf = identity(title, tuple(signature))
    builder = ComponentProgramBuilder(signature, (leaf,))
    values = builder.inputs
    for _ in range(2):
        values = builder.call_dict(title, values)
    record = builder.finalize(values)
    source_signature = SourceSignature(
        (
            SourceRegister("data", CBit() if classical else QBit(), shape=shape),
            SourceRegister("idle", QBit()),
        )
    )

    def get_signature(self: object) -> Any:
        return source_signature

    gate = frozen(type(title, (Bloq,), {"signature": property(get_signature)}))()
    bb, source_values = BloqBuilder.from_signature(source_signature)
    for _ in range(2):
        source_values = bb.add_d(gate, **source_values)
    return component_register_score(record), bb.finalize(**source_values)


@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (0,), (0, 2)])
@pytest.mark.parametrize(
    "title", ["Short", "TenLetters", "ElevenChars", "VeryLongComponentName"]
)
@pytest.mark.parametrize("classical", [False, True])
def test_default_component_symbols_match_unmodified_source_method(
    shape: tuple[int, ...], title: str, classical: bool
) -> None:
    from qualtran.drawing import get_musical_score_data

    score, original = default_fixture(shape, title, classical)
    before = hash(score)
    actual = register_score_data(score)
    expected = get_musical_score_data(original)
    assert_source_geometry(actual, expected, ordered_ports=True)
    assert hash(score) == before
    assert len(actual["vlines"][0]["label"]["text"]) <= 10
    assert export_register_score_json(actual) == export_register_score_json(
        register_score_data(score)
    )


@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (0,), (0, 2)])
def test_default_directional_ports_use_declarations_not_lifetime_keys(
    shape: tuple[int, ...],
) -> None:
    from qualtran import Bloq, BloqBuilder, QBit, Side
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.drawing import get_musical_score_data

    registers = tuple(
        Register("shared", RegisterType("bit", 1), shape, side)
        for side in (RegisterSide.LEFT, RegisterSide.RIGHT)
    )
    left, right = registers
    incoming = tuple(f"consumed{index}" for index in left.all_idxs())
    outgoing = tuple(f"created{index}" for index in right.all_idxs())
    score = layout_register_score(
        dict.fromkeys((*incoming, *outgoing), 1),
        incoming,
        (
            RegisterStep(
                "Directional",
                incoming,
                outgoing,
                input_registers=(left,),
                output_registers=(right,),
            ),
        ),
        input_registers=(left,),
        output_registers=(right,),
    )
    signature = SourceSignature(
        (
            SourceRegister("shared", QBit(), shape=shape, side=Side.LEFT),
            SourceRegister("shared", QBit(), shape=shape, side=Side.RIGHT),
        )
    )

    def get_signature(self: object) -> Any:
        return signature

    gate = type("Directional", (Bloq,), {"signature": property(get_signature)})()
    bb, values = BloqBuilder.from_signature(signature)
    original = bb.finalize(**bb.add_d(gate, **values))
    if 0 in shape:
        # The locked source has no RightDangle for this entirely empty graph.
        with pytest.raises(KeyError, match="RightDangle"):
            get_musical_score_data(original)
        assert register_score_data(score)["soqs"] == []
    else:
        assert_source_geometry(
            register_score_data(score),
            get_musical_score_data(original),
            ordered_ports=True,
        )


@pytest.mark.parametrize("shape", [(), (2,), (1, 2)])
def test_default_component_pixels_only_adapt_boundary_names(
    shape: tuple[int, ...],
) -> None:
    import attrs
    from matplotlib import pyplot as plt
    from qualtran.drawing.musical_score import (
        Text,
        draw_musical_score,
        get_musical_score_data,
    )

    score, original = default_fixture(shape, "VeryLongComponentName", True)
    actual = register_score_data(score)
    expected = get_musical_score_data(original)
    ports = {(port["seq_x"], port["y"]): port for port in actual["soqs"]}
    expected.soqs = [
        (
            attrs.evolve(
                port,
                symb=Text(
                    ports[(port.rpos.seq_x, port.rpos.y)]["symb_attributes"]["text"],
                    fontsize=8,
                ),
            )
            if port.ident.startswith("dang,")
            else port
        )
        for port in expected.soqs
    ]
    assert_source_geometry(actual, expected, ordered_ports=True)
    with plt.rc_context({"figure.dpi": 100, "font.family": "DejaVu Sans"}):
        native, _ = draw_register_score(actual)
        reference, _ = draw_musical_score(expected)
        try:
            assert_pixels(native, reference)
        finally:
            plt.close(native)
            plt.close(reference)


def test_plain_layout_identity_labels_and_explicit_style_remain_owned() -> None:
    step = RegisterStep("VeryLongComponentName", ("key",), ("key",))
    score = layout_register_score({"key": 1}, ("key",), (step,))
    data = register_score_data(score)
    assert data["vlines"][0]["label"]["text"] == step.name
    assert data["soqs"][1]["symb_attributes"]["text"] == "key"
    typed = replace(
        step,
        gate="x",
        input_registers=(Register("q", RegisterType("bit", 1)),),
        output_registers=(Register("q", RegisterType("bit", 1)),),
    )
    styled = layout_register_score({"key": 1}, ("key",), (typed,))
    result = register_score_data(styled)
    assert result["soqs"][1]["symb_cls"] == "ModPlus"
    assert result["vlines"][0]["label"]["text"] == "X"


@pytest.mark.parametrize("input_side", [False, True])
def test_default_metadata_validation_precedes_record_generation(
    input_side: bool,
) -> None:
    score, _ = default_fixture((), "Short", False)
    column = score.columns[0]
    registers = (Register("bad", RegisterType("bit", 1), (3,)),)
    invalid = (
        replace(column.step, input_registers=registers)
        if input_side
        else replace(column.step, output_registers=registers)
    )
    score = replace(score, columns=(replace(column, step=invalid), *score.columns[1:]))
    with pytest.raises(ValueError, match="metadata"):
        register_score_data(score)
