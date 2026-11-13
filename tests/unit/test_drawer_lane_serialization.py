"""Partial lane and existing score records match the frozen source encoder."""

import json
from dataclasses import replace
from hashlib import sha1
from pathlib import Path

import pytest

from flagquantum.core.ir.program.register_types import (
    classical_bit_type,
    classical_unsigned_register_type,
    opaque_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.drawer.register_lanes import LaneSegment, RegisterLaneManager
from flagquantum.drawer.register_score_data import (
    MusicalScoreData,
    SymbolClass,
    export_register_score_json,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_source() -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "ftqc-reference-env/qualtran/drawing/musical_score.py"
    ).read_bytes()
    assert (
        sha1(f"blob {len(source)}\0".encode() + source).hexdigest()
        == "3863bb66cdba17c4654657e275a8485d94b8da75"
    )


@pytest.mark.parametrize("end", [None, -1, 0, 7])
@pytest.mark.parametrize("classical", [False, True])
def test_partial_and_closed_records_match_source_and_own_their_mapping(
    end: int | None, classical: bool
) -> None:
    from qualtran.drawing.musical_score import HLine, HLineFlavor, MusicalScoreEncoder

    flavor = HLineFlavor.CLASSICAL if classical else HLineFlavor.QUANTUM
    native = LaneSegment(3, -1, end)
    if classical:
        native = replace(native, flavor="HLineFlavor.CLASSICAL")
    original = HLine(3, -1, end, flavor)
    record = native.json_dict()
    assert record == original.json_dict()
    assert list(record) == list(original.json_dict())
    expected = json.dumps(original, cls=MusicalScoreEncoder, indent=2)
    assert json.dumps(record, indent=2) == expected
    record["seq_x_end"] = 99
    record["flavor"] = "caller"
    assert native.json_dict() == original.json_dict()


@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (0,), (0, 2)])
@pytest.mark.parametrize("kind", ["qbit", "qzero", "cbit", "czero", "cword"])
def test_manager_snapshots_serialize_before_and_after_release(
    shape: tuple[int, ...], kind: str
) -> None:
    import attrs
    import numpy as np
    from qualtran import CBit, CUInt, LeftDangle, QAny, QBit, RightDangle
    from qualtran import Register as SourceRegister
    from qualtran.drawing.musical_score import LineManager, MusicalScoreEncoder

    native_types = {
        "qbit": quantum_bit_type(),
        "qzero": opaque_register_type(0),
        "cbit": classical_bit_type(),
        "czero": classical_unsigned_register_type(0),
        "cword": classical_unsigned_register_type(3),
    }
    source_types = {
        "qbit": QBit(),
        "qzero": QAny(0),
        "cbit": CBit(),
        "czero": CUInt(0),
        "cword": CUInt(3),
    }
    manager = RegisterLaneManager(8)
    original = LineManager(8)
    register = Register("word", native_types[kind], shape)
    source_register = SourceRegister("word", source_types[kind], shape)
    manager.seq_x = -1
    lanes = manager.new("input", register)
    positions = original.new(LeftDangle, source_register, seq_x=-1, topo_gen=0)
    partials = [line.json_dict() for line in sorted(manager.hlines)]
    source_partials = sorted(original.hlines)
    assert json.dumps(partials, indent=2) == json.dumps(
        source_partials, cls=MusicalScoreEncoder, indent=2
    )
    saved = json.dumps(partials)
    manager.seq_x = 5
    manager.free("output", register, lanes)
    if shape:
        for index in np.ndindex(shape):
            positions[index] = attrs.evolve(positions[index], seq_x=5)
    else:
        positions = attrs.evolve(positions, seq_x=5)
    original.free(RightDangle, source_register, positions)
    complete = [line.json_dict() for line in sorted(manager.hlines)]
    source_complete = sorted(original.hlines)
    assert json.dumps(complete, indent=2) == json.dumps(
        source_complete, cls=MusicalScoreEncoder, indent=2
    )
    assert json.dumps(partials) == saved
    assert all(line["seq_x_end"] is None for line in partials)
    assert all(line["seq_x_end"] == 5 for line in complete)
    assert json.loads(
        json.dumps({"partial": partials, "complete": complete})
    ) == json.loads(
        json.dumps(
            {"partial": source_partials, "complete": source_complete},
            cls=MusicalScoreEncoder,
        )
    )


@pytest.mark.parametrize(
    ("glyph", "attributes"),
    [
        ("Text", {"text": "θ†", "fontsize": 13}),
        ("TextBox", {"text": "θ†"}),
        ("RarrowTextBox", {"text": "state"}),
        ("LarrowTextBox", {"text": "effect"}),
        ("Circle", {"filled": True}),
        ("Circle", {"filled": False}),
        ("ModPlus", {}),
    ],
)
def test_existing_complete_record_export_matches_all_source_symbol_records(
    glyph: SymbolClass, attributes: dict[str, str | int | bool]
) -> None:
    import qualtran.drawing.musical_score as source

    symbol = getattr(source, glyph)(**attributes)
    original = source.MusicalScoreData(
        max_x=2,
        max_y=1,
        soqs=[source.SoqData(symbol, source.RegPosition(1, 0, 1), "port")],
        hlines=[source.HLine(1, -1, 2)],
        vlines=[source.VLine(0, 1, 0, source.Text("caption", fontsize=13))],
    )
    line = LaneSegment(1, -1, 2)
    # Completed score records retain their existing non-null end contract.
    data: MusicalScoreData = {
        "max_x": 2,
        "max_y": 1,
        "soqs": [
            {
                "symb_cls": glyph,
                "symb_attributes": attributes.copy(),
                "ident": "port",
                "y": 1,
                "seq_x": 0,
                "topo_gen": 1,
            }
        ],
        "hlines": [
            {
                "y": line.y,
                "seq_x_start": line.seq_x_start,
                "seq_x_end": 2,
                "flavor": line.flavor,
            }
        ],
        "vlines": [
            {
                "x": 0,
                "top_y": 1,
                "bottom_y": 0,
                "label": {"text": "caption", "fontsize": 13},
            }
        ],
    }
    actual = export_register_score_json(data)
    expected = json.loads(json.dumps(original, cls=source.MusicalScoreEncoder))
    assert json.loads(actual) == expected
    assert actual.endswith("\n")
    assert "\\u03b8" not in actual
