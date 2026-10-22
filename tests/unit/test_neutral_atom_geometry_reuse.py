"""Preserve Remote parsing while reusing Core's finite coordinate semantics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

import flagquantum as fq
from flagquantum.errors import CapabilityError
from flagquantum.experimental.simulation import RydbergProgram
from flagquantum.remote.compute._native_job import NativeJiudingJobClient
from flagquantum.remote.compute.jiuding import JiudingClient, _measurement_program
from flagquantum.remote.qpu.neutral_atom import (
    _coordinate,
    neutral_atom_backend_profile,
)

pytestmark = pytest.mark.unit


class Integer(int):
    pass


class Real(float):
    pass


CASES: dict[str, object] = {
    "pair": (1, 2),
    "list": [1.5, -2],
    "three_coordinates": (1, 2, math.nan),
    "mapping_xy": {"x": 1, "y": 2},
    "mapping_upper": {"X": 1, "Y": 2},
    "mapping_position": {"position": [1, 2, 3]},
    "mapping_coord": {"coord": [1, 2]},
    "mapping_coordinate": {"coordinate": [1, 2]},
    "missing_y": {"x": 1},
    "empty_mapping": {},
    "text": "12",
    "bytes": b"12",
    "none": None,
    "short": [1],
    "empty": [],
    "true": [True, 2],
    "false": [1, False],
    "numeric_string": ["1", 2],
    "complex": [1j, 2],
    "infinite": [math.inf, 2],
    "nan": [1, math.nan],
    "negative_infinite": [-math.inf, 2],
    "integer_subclass": [Integer(1), 2],
    "float_subclass": [Real(1), 2],
    "huge_integer": [10**400, 2],
    "negative_zero": [-0.0, 0.0],
    "nonfinite_then_wrong_type": [math.nan, "x"],
    "wrong_type_then_nonfinite": ["x", math.nan],
    "overflow_then_wrong_type": [10**400, "x"],
    "wrong_type_then_overflow": ["x", 10**400],
    "nonfinite_then_overflow": [math.nan, 10**400],
    "overflow_then_nonfinite": [10**400, math.nan],
}
CAPTURE = json.loads(
    (
        Path(__file__).parents[1] / "fixtures/neutral_atom_geometry_compatibility.json"
    ).read_text()
)


@pytest.mark.parametrize("name", CASES)
def test_coordinate_result_or_refusal_matches_preimplementation_capture(
    name: str,
) -> None:
    try:
        actual = {
            "result": list(_coordinate(CASES[name], owner="Neutral atom", index=3))
        }
    except (ValueError, TypeError, OverflowError) as exc:
        actual = {"exception": type(exc).__name__, "message": str(exc)}
    assert actual == CAPTURE["geometry"][name]


@pytest.mark.parametrize("unit", ("um", "micrometers", "nm", "m"))
def test_register_profile_and_third_coordinate_policy_are_unchanged(unit: str) -> None:
    profile = neutral_atom_backend_profile(
        {"name": "register", "sites": [(0, 0, 99), (3, 4, -99)], "gates": ["x"]},
        interaction_radius=5,
        register_units=unit,
    )
    actual = {
        "n_qubits": profile.n_qubits,
        "edges": profile.coupling_map.edges,
        "metadata": profile.metadata,
        "basis_gates": profile.basis_gates,
    }
    assert json.loads(json.dumps(actual)) == CAPTURE["profiles"][unit]


@pytest.mark.parametrize("raw_record", (False, True))
def test_native_input_refuses_before_existing_remote_side_effects(
    raw_record: bool,
) -> None:
    program = RydbergProgram(
        atoms=(("a", "trap"),),
        sites={"trap": (0.0, 0.0)},
        length_unit="um",
        time_unit="us",
        c6=1.0,
        loss_rate=0.0,
        minimum_separation=1.0,
    )
    value = program.to_ir() if raw_record else program
    for client in (
        JiudingClient(workspace="test"),
        NativeJiudingJobClient(project="set.project"),
    ):
        guards = {
            name: Mock(side_effect=AssertionError("unexpected side effect: " + name))
            for name in (
                "workspace",
                "_resolve_compute_target",
                "submit",
                "_request",
                "_auth",
            )
        }
        with (
            patch.multiple(client, **guards),
            pytest.raises(CapabilityError, match="native Rydberg timelines"),
        ):
            client.submit_program(value, target="jiuding:cpu", image="unused")
        assert all(not spy.called for spy in guards.values())
    with pytest.raises(CapabilityError, match="native Rydberg timelines"):
        _measurement_program(value, outputs=fq.counts(), shots=10)
