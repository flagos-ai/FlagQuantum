"""Canonical Core source encoding, independent of the future numeric plan 1.7."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace

import pytest

import flagquantum as fq
from flagquantum.core.ir.program.physical_codec import (
    native_program_from_dict,
    native_program_to_dict,
)
from flagquantum.errors import SerializationError
from flagquantum.experimental.simulation import RydbergProgram

pytestmark = pytest.mark.unit


@pytest.fixture
def source() -> RydbergProgram:
    program = RydbergProgram(
        atoms=(("a", "left"), ("b", "near")),
        sites={"left": (0, 0), "near": (2, 0), "far": (4, 0)},
        length_unit="um",
        time_unit="us",
        c6=64,
        loss_rate=0.2,
        minimum_separation=1,
        transport_channels=("m",),
        drive_channels=("d",),
    )
    program.move(
        "b", source="near", destination="far", start=0, duration=0.5, channel="m"
    )
    return program.pulse(
        "b",
        start=0,
        duration=0.5,
        amplitude=0.5 * fq.Parameter("omega") + 1,
        phase=0.7,
        channel="d",
    )


def test_native_source_round_trip_preserves_actual_graph_context_and_identity(
    source: RydbergProgram,
) -> None:
    original = source.to_ir()
    payload = native_program_to_dict(original)
    assert set(payload) == {"schema", "version", "functions", "physical_context"}
    restored = native_program_from_dict(json.loads(json.dumps(payload)))
    assert restored == original and hash(restored) == hash(original)
    assert restored.canonical_payload is None
    assert native_program_to_dict(restored) == payload
    assert (
        payload["functions"][0]["blocks"][0]["operations"][0]["opcode"]
        == "rydberg.move"
    )
    changed = deepcopy(payload)
    changed["physical_context"]["c6"] = -64.0
    assert native_program_from_dict(changed) != original
    source.pulse("a", start=0.5, duration=0.5, amplitude=1, channel="d")
    assert native_program_from_dict(payload) == original


@pytest.mark.parametrize(
    "location", ["program", "context", "function", "block", "operation", "data"]
)
def test_unknown_fields_at_every_native_depth_are_refused(
    source: RydbergProgram, location: str
) -> None:
    payload = native_program_to_dict(source.to_ir())
    nodes = {
        "program": payload,
        "context": payload["physical_context"],
        "function": payload["functions"][0],
        "block": payload["functions"][0]["blocks"][0],
        "operation": payload["functions"][0]["blocks"][0]["operations"][0],
        "data": payload["functions"][0]["blocks"][0]["operations"][0]["physical_data"],
    }
    nodes[location]["unreviewed"] = True
    with pytest.raises(SerializationError, match="exactly fields"):
        native_program_from_dict(payload)


@pytest.mark.parametrize(
    "malformed",
    [
        {"$parameter": 123},
        {"$parameter": "omega", "extra": 0},
        {
            "$expression": {
                "op": "mul",
                "args": [1, {"$parameter": "omega"}],
                "extra": 0,
            }
        },
        True,
        {"$complex": [1, 0]},
    ],
)
def test_native_scalar_decoder_rejects_coercing_or_noncanonical_encodings(
    source: RydbergProgram, malformed: object
) -> None:
    payload = native_program_to_dict(source.to_ir())
    payload["functions"][0]["blocks"][0]["operations"][1]["physical_data"][
        "amplitude"
    ] = malformed
    with pytest.raises(SerializationError):
        native_program_from_dict(payload)


@pytest.mark.parametrize(
    "field,value",
    [("kind", "ion"), ("time_unit", "ns"), ("loss_rate", -1.0), ("c6", float("nan"))],
)
def test_native_context_semantics_are_verified_after_decoding(
    source: RydbergProgram, field: str, value: object
) -> None:
    payload = native_program_to_dict(source.to_ir())
    payload["physical_context"][field] = value
    with pytest.raises(SerializationError):
        native_program_from_dict(payload)


def test_encoder_does_not_serialize_an_invalid_physical_source(
    source: RydbergProgram,
) -> None:
    record = source.to_ir()
    assert record.physical_context is not None
    with pytest.raises(SerializationError, match="physical.invalid_context"):
        native_program_to_dict(
            replace(
                record,
                physical_context=replace(record.physical_context, minimum_separation=0),
            )
        )


def test_disallowed_tensor_leaf_refuses_before_general_codec_allocation(
    source: RydbergProgram, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = native_program_to_dict(source.to_ir())
    payload["functions"][0]["blocks"][0]["operations"][1]["physical_data"][
        "amplitude"
    ] = {"$tensor": {"dtype": "float64", "shape": [10**12], "data": []}}

    def forbidden(value: object) -> None:
        raise AssertionError("general decoder was invoked for a disallowed tensor leaf")

    monkeypatch.setattr(
        "flagquantum.core.ir.program.physical_codec._decode_value", forbidden
    )
    with pytest.raises(SerializationError, match="not native control scalars"):
        native_program_from_dict(payload)


def test_stricter_native_encoding_does_not_change_existing_scalar_codec() -> None:
    from flagquantum.core.ir import _decode_value

    parameter = _decode_value({"$parameter": 123})
    assert isinstance(parameter, fq.Parameter)
    assert parameter.name == "123"
