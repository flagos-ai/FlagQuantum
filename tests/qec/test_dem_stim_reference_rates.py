"""Cross-check a built detector error model against stim's own error analysis.

The distance-2 and distance-3 rotated-surface memory circuits are transcribed
gate for gate into stim, with the same phenomenological noise model, and stim
builds its detector error model from that circuit with its own error-analysis
pass. The two models were then compared: they declare the same number of
detectors, the same number of observables, the same number of mechanisms, and
their detector marginals agree within the sampling error of the stim side.

This is an independent re-derivation rather than a shared-helper cross-check:
stim never sees the FlagQuantum injection engine, the layouts, or the model, and
the FlagQuantum side never sees the stim circuit. What the two implementations
share is the circuit's gate sequence and the noise model, which is exactly the
input the comparison is about.

The marginals are only a usable comparison because a wrong rule misses the
tolerance by a wide margin. Measured on the distance-3, three-round
configuration at four hundred thousand shots, against the FlagQuantum model's
exact rates: the correct transcription differs by at most 0.00083 (2.3 standard
errors of the largest-rate detector, and the largest of twenty-four draws);
applying the data flip after the round's gates instead of before them differs by
0.038; and a terminal detector that omits the data readout differs by 0.107.
"""

from __future__ import annotations

import numpy as np
import pytest
import stim
from numpy.typing import NDArray

from flagquantum.qec import DetectorErrorModel, RotatedSurfaceCode, build_memory_circuit
from flagquantum.qec.circuit import MemoryCircuit
from flagquantum.qec.noise import PhenomenologicalNoise

pytestmark = pytest.mark.integration

_SHOTS = 400_000
_PROBABILITY = 0.01
_SIGMA = 4.0


def _transcribe(memory: MemoryCircuit, *, probability: float) -> stim.Circuit:
    """Return the stim circuit the memory circuit describes.

    The transcription follows the emitted program's conventions rather than
    re-deriving them: a data flip precedes each round's gates, a measurement
    flip precedes the syndrome measurement it corrupts, a Z-type ancilla is
    controlled by the data wires it checks, an X-type ancilla controls them and
    is prepared and read in the Hadamard basis, every ancilla is reset after it
    is measured, and the data wires are read once at the end.
    """

    code = memory.code
    data = list(code.data_wires)
    checks = list(code.checks)
    ancillas = [check.ancilla_wire for check in checks]
    z_checks = [check for check in checks if not check.stabilizer.x_wires]

    def rec(circuit: stim.Circuit, absolute: int) -> stim.GateTarget:
        """Return the record target of one absolute measurement index."""

        return stim.target_rec(absolute - circuit.num_measurements)

    circuit = stim.Circuit()
    circuit.append("R", data + ancillas)
    syndrome: list[dict[int, int]] = []
    for _ in range(memory.rounds):
        circuit.append("X_ERROR", data, probability)
        this_round: dict[int, int] = {}
        for check in checks:
            ancilla = check.ancilla_wire
            circuit.append("X_ERROR", [ancilla], probability)
            if check.stabilizer.x_wires:
                circuit.append("H", [ancilla])
                for wire in check.stabilizer.support:
                    circuit.append("CX", [ancilla, wire])
                circuit.append("H", [ancilla])
            else:
                for wire in check.stabilizer.support:
                    circuit.append("CX", [wire, ancilla])
            circuit.append("M", [ancilla])
            this_round[ancilla] = circuit.num_measurements - 1
            circuit.append("R", [ancilla])
        syndrome.append(this_round)

    # Detectors are declared in the layout's own order: round zero covers the
    # Z-type checks alone, each later round covers every check against the round
    # before it, and the terminal boundary compares the last syndrome of each
    # Z-type check with the data readout it protects.
    for check in z_checks:
        circuit.append("DETECTOR", [rec(circuit, syndrome[0][check.ancilla_wire])])
    for round_index in range(1, memory.rounds):
        for check in checks:
            ancilla = check.ancilla_wire
            circuit.append(
                "DETECTOR",
                [
                    rec(circuit, syndrome[round_index][ancilla]),
                    rec(circuit, syndrome[round_index - 1][ancilla]),
                ],
            )

    circuit.append("M", data)
    data_rec = {
        wire: circuit.num_measurements - len(data) + position
        for position, wire in enumerate(data)
    }
    for check in z_checks:
        targets = [rec(circuit, syndrome[memory.rounds - 1][check.ancilla_wire])]
        targets.extend(
            rec(circuit, data_rec[wire]) for wire in check.stabilizer.support
        )
        circuit.append("DETECTOR", targets)
    circuit.append(
        "OBSERVABLE_INCLUDE",
        [rec(circuit, data_rec[wire]) for wire in code.logical_observables[0].support],
        0,
    )
    return circuit


def _built(distance: int, rounds: int) -> MemoryCircuit:
    return build_memory_circuit(RotatedSurfaceCode(distance=distance), rounds=rounds)


@pytest.mark.parametrize(("distance", "rounds"), [(2, 2), (3, 3)])
def test_the_reference_circuit_declares_the_same_shape(
    distance: int, rounds: int
) -> None:
    memory = _built(distance, rounds)
    circuit = _transcribe(memory, probability=_PROBABILITY)

    assert circuit.num_detectors == len(memory.detectors.detectors)
    assert circuit.num_observables == len(memory.observables.observables)


@pytest.mark.parametrize(("distance", "rounds"), [(2, 2), (3, 3)])
def test_the_reference_model_has_the_same_mechanism_count(
    distance: int, rounds: int
) -> None:
    """Both implementations enumerate one mechanism per noise location.

    The count agreeing is a measured agreement, not an identity the test
    imposes: the FlagQuantum engine forces each location through the circuit and
    merges locations with the same signature, while stim's error analysis
    propagates each location symbolically and merges the same way.
    """

    memory = _built(distance, rounds)
    circuit = _transcribe(memory, probability=_PROBABILITY)

    built = DetectorErrorModel.from_memory_circuit(
        memory,
        noise=PhenomenologicalNoise(
            data_flip=_PROBABILITY, measurement_flip=_PROBABILITY
        ),
    )
    reference = circuit.detector_error_model(decompose_errors=False)

    assert len(built.errors) == reference.num_errors


@pytest.mark.parametrize(("distance", "rounds"), [(2, 2), (3, 3)])
def test_the_reference_marginals_agree_within_its_sampling_error(
    distance: int, rounds: int
) -> None:
    """Every detector marginal matches the independent implementation's.

    The tolerance is four standard errors of the largest measured rate, which is
    the widest sampling error any single detector in these configurations has.
    The largest deviation the correct transcription produced was 2.3 standard
    errors, while the mis-transcriptions described in the module docstring
    deviate by 0.038 and 0.107 — twenty-six and seventy-one times the tolerance.
    """

    memory = _built(distance, rounds)
    circuit = _transcribe(memory, probability=_PROBABILITY)
    built = DetectorErrorModel.from_memory_circuit(
        memory,
        noise=PhenomenologicalNoise(
            data_flip=_PROBABILITY, measurement_flip=_PROBABILITY
        ),
    )
    reference = circuit.detector_error_model(decompose_errors=False)
    sampled = reference.compile_sampler(seed=11).sample(_SHOTS)[0]
    obtained = np.asarray(sampled, dtype=np.float64).mean(axis=0)
    expected: NDArray[np.float64] = built.detector_rates().numpy()

    assert obtained.shape == expected.shape
    widest = float(np.max(np.sqrt(obtained * (1.0 - obtained) / _SHOTS)))
    tolerance = _SIGMA * widest
    deviation = float(np.max(np.abs(obtained - expected)))
    assert deviation < tolerance, (
        f"largest deviation {deviation:.6f} at detector "
        f"{int(np.argmax(np.abs(obtained - expected)))} exceeds {tolerance:.6f}"
    )


def test_the_reference_observable_marginal_agrees() -> None:
    memory = _built(3, 3)
    circuit = _transcribe(memory, probability=_PROBABILITY)
    built = DetectorErrorModel.from_memory_circuit(
        memory,
        noise=PhenomenologicalNoise(
            data_flip=_PROBABILITY, measurement_flip=_PROBABILITY
        ),
    )
    reference = circuit.detector_error_model(decompose_errors=False)
    sampled = reference.compile_sampler(seed=12).sample(_SHOTS)
    observed = float(np.asarray(sampled[1], dtype=np.float64).mean())
    expected = float(built.observable_rates().numpy()[0])

    widest = float(np.sqrt(observed * (1.0 - observed) / _SHOTS))
    assert abs(observed - expected) < _SIGMA * widest
