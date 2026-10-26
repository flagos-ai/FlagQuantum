"""Source-only proposal 092 delivery; numerical admission remains closed."""

from copy import deepcopy

import pytest
import torch

import flagquantum.lindblad as fql
from flagquantum.core.ir import IRValidationError
from flagquantum.core.ir.program.physical_codec import (
    native_program_from_dict,
    native_program_to_dict,
)
from flagquantum.core.rydberg import RydbergProgram
from flagquantum.errors import CapabilityError, SerializationError

pytestmark = pytest.mark.unit


def source() -> RydbergProgram:
    p = RydbergProgram(
        atoms=(("a", "home"),),
        sites={"home": (0.0, 0.0)},
        length_unit="um",
        time_unit="us",
        c6=0.0,
        loss_rate=0.1,
        minimum_separation=1.0,
        drive_channels=("laser",),
    )
    return p.pulse("a", start=0.0, duration=1.0, amplitude=0.4, channel="laser")


def test_presence_source_roundtrip_and_closed_numeric_admission() -> None:
    p = source()
    assert p.measure_presence("a", at=0.5, efficiency=0.8) is p
    payload = native_program_to_dict(p.to_ir())
    assert payload["version"] == "1.4"
    assert native_program_to_dict(native_program_from_dict(payload)) == payload
    with pytest.raises(CapabilityError, match="explicit shots=1"):
        fql.plan(p, p.basis_state(("g",)), [0.0, 0.5, 1.0])


@pytest.mark.parametrize(
    "field,value",
    [
        ("at", True),
        ("at", torch.tensor(0.5)),
        ("efficiency", False),
        ("efficiency", torch.tensor(0.8)),
    ],
)
def test_presence_leaf_types(field: str, value: object) -> None:
    kwargs = {"at": 0.5, "efficiency": 0.8, field: value}
    with pytest.raises(TypeError):
        source().measure_presence("a", **kwargs)


@pytest.mark.parametrize("at", [0.0, -1.0, float("nan"), float("inf")])
def test_presence_invalid_time(at: float) -> None:
    with pytest.raises(IRValidationError):
        source().measure_presence("a", at=at, efficiency=0.8)


@pytest.mark.parametrize("efficiency", [-0.1, 1.1, float("nan"), float("inf")])
def test_presence_invalid_efficiency(efficiency: float) -> None:
    with pytest.raises(IRValidationError):
        source().measure_presence("a", at=0.5, efficiency=efficiency)


@pytest.mark.parametrize("at", [1.0, 2.0])
def test_presence_horizon_is_snapshot_checked(at: float) -> None:
    p = source().measure_presence("a", at=at, efficiency=0.8)
    with pytest.raises(IRValidationError, match="horizon"):
        p.to_ir()


def test_presence_unknown_atom_and_repeat_are_rejected() -> None:
    with pytest.raises(IRValidationError):
        source().measure_presence("missing", at=0.5, efficiency=0.8)
    p = source().measure_presence("a", at=0.5, efficiency=0.8)
    with pytest.raises(IRValidationError, match="one"):
        p.measure_presence("a", at=0.6, efficiency=0.8)


def test_presence_version_and_payload_are_strict() -> None:
    payload = native_program_to_dict(
        source().measure_presence("a", at=0.5, efficiency=0.8).to_ir()
    )
    wrong = deepcopy(payload)
    wrong["version"] = "1.0"
    with pytest.raises(SerializationError):
        native_program_from_dict(wrong)
    wrong = deepcopy(payload)
    wrong["functions"][0]["blocks"][0]["operations"][1]["physical_data"]["extra"] = 0
    with pytest.raises(SerializationError):
        native_program_from_dict(wrong)
    wrong = native_program_to_dict(source().to_ir())
    wrong["version"] = "1.4"
    with pytest.raises(SerializationError):
        native_program_from_dict(wrong)


@pytest.mark.parametrize(
    "mutation",
    [
        "operands",
        "results",
        "parameters",
        "literal",
        "condition_clauses",
        "null",
        "wrong_kind",
        "repeat",
        "reorder",
        "bool_time",
    ],
)
def test_presence_rejects_raw_source_tampering(mutation: str) -> None:
    payload = native_program_to_dict(
        source().measure_presence("a", at=0.5, efficiency=0.8).to_ir()
    )
    operations = payload["functions"][0]["blocks"][0]["operations"]
    observation = operations[1]
    if mutation in ("operands", "results", "parameters", "condition_clauses"):
        observation[mutation] = [0]
    elif mutation == "literal":
        observation[mutation] = 0
    elif mutation == "null":
        observation["physical_data"] = None
    elif mutation == "wrong_kind":
        observation["physical_data"]["kind"] = "pulse"
    elif mutation == "repeat":
        operations.append(deepcopy(observation))
    elif mutation == "reorder":
        operations.reverse()
    else:
        observation["physical_data"]["at"] = True
    with pytest.raises(SerializationError):
        native_program_from_dict(payload)
