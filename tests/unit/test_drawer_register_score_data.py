"""Check native port glyphs and serialized geometry against the locked score."""

import json

import pytest

from flagquantum.drawer.register_score import (
    RegisterStep,
    layout_register_score,
    register_program_score,
)
from flagquantum.drawer.register_score_data import (
    export_register_score_json,
    register_score_data,
)
from tests.register_program_cases import Registers

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
def test_cnot_and_classical_passthrough_match_locked_json(tmp_path) -> None:
    pytest.importorskip("qualtran")
    from qualtran import BloqBuilder, CUInt, QBit, Register, Signature
    from qualtran.bloqs.basic_gates import CNOT
    from qualtran.drawing import dump_musical_score, get_musical_score_data

    signature = Signature(
        [
            Register("ctrl", QBit()),
            Register("target", QBit()),
            Register("bits", CUInt(3)),
        ]
    )
    builder, inputs = BloqBuilder.from_signature(signature)
    ctrl, target = builder.add(CNOT(), ctrl=inputs["ctrl"], target=inputs["target"])
    composite = builder.finalize(ctrl=ctrl, target=target, bits=inputs["bits"])
    path = tmp_path / "upstream"
    dump_musical_score(get_musical_score_data(composite), name=str(path))
    expected = json.loads(path.with_suffix(".json").read_text())

    score = layout_register_score(
        {"ctrl": 1, "target": 1, "bits": 3},
        ("ctrl", "target", "bits"),
        (RegisterStep("pair", ("ctrl", "target"), ("ctrl", "target"), "cx"),),
    )
    actual = json.loads(
        export_register_score_json(register_score_data(score, classical={"bits"}))
    )
    # The source lists soquets in graph traversal order. Compare stable identities.
    actual["soqs"].sort(key=lambda port: port["ident"])
    expected["soqs"].sort(key=lambda port: port["ident"])
    assert actual == expected


@pytest.mark.ftqc_reference
def test_alloc_free_direction_matches_locked_score() -> None:
    pytest.importorskip("qualtran")
    from qualtran import BloqBuilder, QAny
    from qualtran.drawing import get_musical_score_data

    builder = BloqBuilder()
    work = builder.allocate(dtype=QAny(3))
    builder.free(work)
    expected = get_musical_score_data(builder.finalize())
    score = layout_register_score(
        {"reg": 3},
        (),
        (
            RegisterStep("qreg.local", outputs=("reg",)),
            RegisterStep("qreg.release", inputs=("reg",)),
        ),
    )
    actual = register_score_data(score)
    assert actual["max_x"] == expected.max_x == 1
    assert actual["hlines"] == [line.json_dict() for line in expected.hlines]
    assert actual["soqs"] == [port.json_dict() for port in expected.soqs]
    assert actual["vlines"] == [line.json_dict() for line in expected.vlines]


def test_native_cnot_keeps_control_target_order_and_connector() -> None:
    registers = Registers()
    target = registers.declare("target", 1)
    ctrl = registers.declare("ctrl", 1)
    registers.emit("qreg.allocate", ctrl)
    registers.emit("qreg.allocate", target)
    registers.emit("qreg.pair_gate", ctrl, target, gate="cx")
    score = register_program_score(registers.finish())
    column = next(c for c in score.columns if c.step.gate == "cx")
    assert column.lanes == (1, 0)
    data = register_score_data(score)
    ports = [port for port in data["soqs"] if port["seq_x"] == column.sequence]
    assert [(port["y"], port["symb_cls"]) for port in ports] == [
        (1, "Circle"),
        (0, "ModPlus"),
    ]
    assert len({port["ident"] for port in data["soqs"]}) == len(data["soqs"])
    assert data["vlines"][-1] == {
        "x": column.sequence,
        "top_y": 1,
        "bottom_y": 0,
        "label": {"text": "", "fontsize": 10},
    }
