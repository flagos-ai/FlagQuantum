"""Surface memory evidence through existing code, maps, sampler and decoders."""

from __future__ import annotations

from typing import Literal

import pytest
import torch

pytest.importorskip("stim")
pytest.importorskip("pymatching")

from examples.qec.stim_pymatching_baseline import run_surface_baseline
from flagquantum.qec import (
    DetectorErrorModel,
    MemoryCircuit,
    MinimumWeightMatchingDecoder,
    Pauli,
    PhenomenologicalNoise,
    PyMatchingDecoder,
    RotatedSurfaceCode,
    build_memory_circuit,
    compile_memory_sampler,
    decoder_context_from_memory_circuit,
    sample_memory_measurements,
)
from flagquantum.qec.circuits.memory import MeasurementRef
from flagquantum.qec.execution.signatures import (
    _inject_data_flip,
    _inject_measurement_flip,
    _lowered_source,
)
from flagquantum.simulation.stabilizer import sample_noisy_measurements

pytestmark = pytest.mark.integration


def _memory(distance: int, basis: str, rounds: int = 3) -> MemoryCircuit:
    code = RotatedSurfaceCode(distance)
    product = Pauli(x_qubits=code.data_qubits) if basis == "x" else None
    return build_memory_circuit(code, rounds=rounds, product=product)


@pytest.mark.parametrize("distance", (3, 5, 7))
@pytest.mark.parametrize("basis", ("x", "z"))
@pytest.mark.parametrize("rounds", (1, 3))
def test_surface_measurement_maps_and_compiled_samples_agree(
    distance: int, basis: str, rounds: int
) -> None:
    memory = _memory(distance, basis, rounds)
    assert memory.x_readout_qubits == (memory.code.data_qubits if basis == "x" else ())
    noise = PhenomenologicalNoise(
        data_flip=0.005, phase_flip=0.007, measurement_flip=0.009
    )
    inputs = decoder_context_from_memory_circuit(memory, noise=noise).full_component()
    raw = sample_memory_measurements(memory, noise=noise, shots=17, seed=53)
    with compile_memory_sampler(memory, noise=noise, seed=53) as sampler:
        sample = sampler.sample(shots=17)
    bits = raw.outcomes.to(torch.int64)
    assert torch.equal(
        (bits @ inputs.measurement_to_detectors.dense().to(torch.int64).T) % 2,
        sample.detectors,
    )
    assert torch.equal(
        (bits @ inputs.measurement_to_observables.dense().to(torch.int64).T) % 2,
        sample.observables,
    )
    # Both uniform frames pin half the checks at preparation and readout.
    assert sample.detectors.shape == (17, rounds * (distance**2 - 1))
    assert sample.observables.shape == (17, 1)
    with compile_memory_sampler(
        memory, noise=PhenomenologicalNoise(), seed=53
    ) as sampler:
        clean = sampler.sample(shots=8)
    assert not clean.detectors.any() and not clean.observables.any()


@pytest.mark.parametrize("distance", (3, 5, 7))
@pytest.mark.parametrize("basis", ("x", "z"))
def test_surface_report_counts_tail_and_preserves_the_chosen_frame(
    distance: int, basis: Literal["x", "z"]
) -> None:
    report = run_surface_baseline(
        distance=distance, readout_basis=basis, shots=35, batch_size=16
    )
    assert report["requested_shots"] == report["shots"] == 35
    assert report["readout_basis"] == basis
    assert report["code"] == "RotatedSurfaceCode"
    assert report["detectors"] == 3 * (distance**2 - 1)
    assert report["logical_support"] == list(
        range(distance**2 if basis == "x" else distance)
    )
    assert report["observable_failures"] == [report["logical_failures"]]
    assert report["logical_error_rate"] == report["logical_failures"] / 35
    lower, upper = report["confidence_interval"]
    assert 0 <= lower <= report["logical_error_rate"] <= upper <= 1


def _assert_forced_flips(
    memory: MemoryCircuit, source: str, flipped: set[MeasurementRef]
) -> None:
    expected_detectors = torch.tensor(
        [
            len(set(detector.parity) & flipped) % 2
            for detector in memory.detectors.detectors
        ],
        dtype=torch.bool,
    )
    expected_observable = (
        len(set(memory.observables.observables[0].measurement_parity) & flipped) % 2
    )
    # A deliberately injected round is not an identical-round memory program.
    # Execute the existing Simulation seam directly for this independent oracle;
    # the production sampler must retain its identical-round placement guard.
    program = _lowered_source(
        source, rounds=memory.rounds, checks=len(memory.code.checks)
    )
    record = sample_noisy_measurements(
        program, terminal_qubits=memory.code.data_qubits, shots=3, seed=53
    )
    columns = {
        MeasurementRef(round_index, check.ancilla_qubit): round_index
        * len(memory.code.checks)
        + position
        for round_index in range(memory.rounds)
        for position, check in enumerate(memory.code.checks)
    }
    columns.update(
        {
            MeasurementRef(None, wire): memory.rounds * len(memory.code.checks)
            + position
            for position, wire in enumerate(memory.code.data_qubits)
        }
    )
    detectors = torch.stack(
        [
            record[:, [columns[ref] for ref in detector.parity]].sum(dim=1) % 2
            for detector in memory.detectors.detectors
        ],
        dim=1,
    )
    observable = (
        record[
            :,
            [
                columns[ref]
                for ref in memory.observables.observables[0].measurement_parity
            ],
        ].sum(dim=1)
        % 2
    )
    assert torch.equal(detectors, expected_detectors.expand(3, -1))
    assert torch.equal(
        observable, torch.full((3,), expected_observable, dtype=observable.dtype)
    )


@pytest.mark.parametrize("distance", (3, 5, 7))
@pytest.mark.parametrize("basis", ("x", "z"))
def test_first_and_last_round_data_faults_at_corner_and_interior(
    distance: int, basis: str
) -> None:
    memory = _memory(distance, basis)
    for round_index in (0, 2):
        for wire in (0, distance**2 // 2):
            for fault in ("x", "z"):
                pauli = (
                    Pauli(x_qubits=(wire,)) if fault == "x" else Pauli(z_qubits=(wire,))
                )
                # Propagate persistence using Pauli commutation against checks;
                # do not read the DEM construction's signature helper.
                flipped = {
                    MeasurementRef(later, check.ancilla_qubit)
                    for later in range(round_index, memory.rounds)
                    for check in memory.code.checks
                    if not pauli.commutes_with(check.stabilizer)
                }
                if (fault == "x") == (basis == "z"):
                    flipped.add(MeasurementRef(None, wire))
                source = _inject_data_flip(
                    memory, round_index=round_index, wire=wire, fault=fault
                )
                _assert_forced_flips(memory, source, flipped)


@pytest.mark.parametrize("distance", (3, 5, 7))
@pytest.mark.parametrize("basis", ("x", "z"))
def test_first_and_last_round_readout_faults_on_boundary_and_bulk_checks(
    distance: int, basis: str
) -> None:
    memory = _memory(distance, basis)
    for is_x in (False, True):
        for weight in (2, 4):
            check = next(
                c
                for c in memory.code.checks
                if c.stabilizer.weight == weight and bool(c.stabilizer.x_qubits) == is_x
            )
            for round_index in (0, 2):
                source = _inject_measurement_flip(
                    memory, round_index=round_index, ancilla_qubit=check.ancilla_qubit
                )
                _assert_forced_flips(
                    memory, source, {MeasurementRef(round_index, check.ancilla_qubit)}
                )


@pytest.mark.parametrize("basis", ("x", "z"))
def test_native_and_external_matchers_correct_the_same_single_fault_rows(
    basis: str,
) -> None:
    memory = _memory(3, basis)
    model = DetectorErrorModel.from_memory_circuit(
        memory,
        noise=PhenomenologicalNoise(
            data_flip=0.005, phase_flip=0.007, measurement_flip=0.009
        ),
    ).merge_duplicate_mechanisms()
    native = MinimumWeightMatchingDecoder.from_detector_error_model(model)
    external = PyMatchingDecoder.from_detector_error_model(model)
    rows = torch.zeros((len(model.errors) + 1, model.num_detectors), dtype=torch.bool)
    expected = torch.zeros((len(rows), 1), dtype=torch.bool)
    for index, error in enumerate(model.errors, start=1):
        rows[index, list(error.detectors)] = True
        expected[index, list(error.observables)] = True
    batched = external.decode_batch(rows)
    native_bits = torch.zeros_like(expected)
    row_bits = torch.zeros_like(expected)
    for index, row in enumerate(rows):
        syndrome = tuple(int(bit) for bit in row.nonzero().flatten())
        native_bits[index, list(native.decode(syndrome).observables)] = True
        row_bits[index, list(external.decode(syndrome).observables)] = True
    assert torch.equal(batched, expected)
    assert torch.equal(batched, row_bits)
    assert torch.equal(batched, native_bits)
