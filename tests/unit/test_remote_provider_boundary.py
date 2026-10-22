"""Retire the generic provider without moving local physics into Remote."""

from __future__ import annotations

import importlib.util
from unittest.mock import Mock, patch

import pytest

import flagquantum as fq
import flagquantum.remote as remote
import flagquantum.remote.qpu as qpu
from flagquantum.errors import CapabilityError
from flagquantum.experimental.simulation import RydbergProgram
from flagquantum.remote.compute._native_job import NativeJiudingJobClient
from flagquantum.remote.compute.jiuding import JiudingClient, _measurement_program

pytestmark = pytest.mark.unit


def pair() -> RydbergProgram:
    return RydbergProgram(
        atoms=(("a", "left"), ("b", "near")),
        sites={"left": (0.0, 0.0), "near": (4.0, 0.0), "far": (8.0, 0.0)},
        length_unit="um",
        time_unit="us",
        c6=64.0,
        loss_rate=0.2,
        minimum_separation=1.0,
        transport_channels=("aod",),
    )


def test_retired_module_and_generic_exports_have_no_compatibility_shell() -> None:
    assert importlib.util.find_spec("flagquantum.remote.qpu.neutral_atom") is None
    for namespace in (remote, qpu):
        for name in (
            "NeutralAtomProvider",
            "interaction_edges",
            "neutral_atom_backend_profile",
        ):
            assert name not in namespace.__all__
            assert not hasattr(namespace, name)


def test_local_geometry_still_uses_the_existing_core_program() -> None:
    program = pair()
    program.move(
        "b", source="near", destination="far", start=0.0, duration=1.0, channel="aod"
    )
    record = program.to_ir()
    assert program.commanded_position("b", 0.5) == (6.0, 0.0)
    assert program.atom_ids == ("a", "b")
    assert program.dimensions == (3, 3)
    assert record.physical_context is not None
    assert program.survival_operator().shape == (9, 9)


@pytest.mark.parametrize("raw_record", (False, True))
def test_existing_remote_intakes_refuse_native_inputs_before_side_effects(
    raw_record: bool,
) -> None:
    program = pair()
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
