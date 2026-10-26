"""Circuit-location DEMs retain exact reference results without amplitude storage."""

from __future__ import annotations

import pytest

from flagquantum.noise import NoiseModel, bit_flip_channel, phase_flip_channel
from flagquantum.qec import (
    DetectorErrorModel,
    RepetitionCode,
    RotatedSurfaceCode,
    SteaneCode,
    ZxxzSurfaceCode,
    build_memory_circuit,
)
from flagquantum.qec.codes import StabilizerCode
from flagquantum.qec.dem import construction
from flagquantum.qec.execution import signatures
from flagquantum.qec.execution.signatures import (
    _gate_fault_signatures,
    _lowered_program,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("code", "basis"),
    [
        (RepetitionCode(distance=3), "z"),
        (RotatedSurfaceCode(distance=2), "z"),
        (SteaneCode(), "z"),
        (SteaneCode(), "x"),
        (ZxxzSurfaceCode(distance=3), "z"),
        (ZxxzSurfaceCode(distance=3), "x"),
    ],
)
def test_every_gate_location_matches_reference_execution(
    code: StabilizerCode, basis: str
) -> None:
    memory = build_memory_circuit(code, rounds=2, readout_basis=basis)
    program = _lowered_program(memory)
    noise = (
        NoiseModel()
        .add("cx", bit_flip_channel(0.015625))
        .add("cx", phase_flip_channel(0.03125))
    )
    if basis == "z":
        noise.add("h", phase_flip_channel(0.0625))
    mechanisms = construction._gate_bound_mechanisms(noise, program)
    faults = tuple(
        (m.after_instruction, m.wire, "X" if m.kind == "data" else "Z")
        for m in mechanisms
    )
    propagated = _gate_fault_signatures(memory, program, faults)
    reference = tuple(
        signatures._executed_flip_set(memory, construction._spliced_fault(program, m))
        for m in mechanisms
    )
    assert propagated == reference
    expected = DetectorErrorModel._merge_mechanisms(
        tuple(
            (m.probability, *result)
            for m, result in zip(mechanisms, reference, strict=True)
        ),
        num_detectors=len(memory.detectors),
        num_observables=len(memory.observables),
    )
    assert DetectorErrorModel.from_memory_circuit(memory, noise=noise) == expected


@pytest.mark.parametrize("distance", [5, 7])
def test_gate_bound_surface_model_needs_no_reference_execution(
    distance: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("a Clifford DEM must not allocate an amplitude store")

    monkeypatch.setattr(construction, "_executed_flip_set", forbidden)
    monkeypatch.setattr(construction, "_spliced_fault", forbidden)
    monkeypatch.setattr(signatures, "execute_hybrid_dynamic_session", forbidden)
    memory = build_memory_circuit(RotatedSurfaceCode(distance=distance), rounds=3)
    model = DetectorErrorModel.from_memory_circuit(
        memory,
        noise=NoiseModel().add("cx", bit_flip_channel(0.01)),
    )
    assert model.num_detectors == len(memory.detectors)
    assert model.num_observables == len(memory.observables)
    assert model.num_errors > 0
    assert all(
        0 <= d < model.num_detectors for error in model.errors for d in error.detectors
    )
