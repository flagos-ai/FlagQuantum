"""Cross-check the matrix route against stim's own error analysis.

The matrix route states a code-capacity experiment: a fault in round ``r`` shows
up in the syndrome row of round ``r`` and of round ``r + 1``, and nowhere else.
That is a claim about an experiment, so it can be handed to an independent
implementation. The circuit below is exactly that experiment and nothing more --
the faults live on their own qubits, which are reset each round, so a fault
cannot leak past the round it is injected in, and stim derives the detectors,
the observables and the rates from the circuit with its own error-analysis pass.

The transcription in :mod:`tests.qec.test_dem_stim_reference_rates` is a
different circuit on purpose. It is the memory circuit this package builds, and
its detectors include the terminal comparison between the last syndrome of each
Z-type check and the data readout. The matrix route has no terminal readout, so
the two routes disagree about the geometry and must be compared to stim
separately. Conflating them would hide which of the two a failure belongs to.

The comparison is on the declared shape, the number of mechanisms, and every
mechanism's full signature and rate. The rates are stim's own, so a rate this
package derived from the wrong fault family shows up as a difference rather than
being checked against itself.
"""

from __future__ import annotations

import pytest

from flagquantum.qec import (
    DetectorErrorModel,
    PhenomenologicalNoise,
    RepetitionCode,
    RotatedSurfaceCode,
    build_memory_circuit,
    code_matrices,
)

# Stim is an optional distribution and the reference side needs it, so the suite
# skips rather than fails when it is absent.
pytest.importorskip("stim")

import stim

pytestmark = pytest.mark.integration

# Distance, round count, data rate, measurement rate. The two rates are separate
# so a fault family the model confuses with the other one is caught, and the
# configurations cover both fault families alone as well as together.
_CONFIGURATIONS = [
    ("repetition-3", 3, 1, 0.05, 0.05),
    ("repetition-3", 3, 3, 0.02, 0.02),
    ("repetition-5", 5, 4, 0.01, 0.03),
    ("repetition-3-data-only", 3, 2, 0.04, 0.0),
    ("repetition-3-measurement-only", 3, 2, 0.0, 0.06),
    ("surface-2", 2, 2, 0.03, 0.02),
    ("surface-3", 3, 3, 0.01, 0.01),
]


def _code(name: str, distance: int) -> object:
    return (
        RotatedSurfaceCode(distance=distance)
        if name.startswith("surface")
        else RepetitionCode(distance=distance)
    )


def _reference_circuit(
    hz: list[list[int]],
    lz: list[list[int]],
    *,
    data_flip: float,
    measurement_flip: float,
    rounds: int,
) -> stim.Circuit:
    """Return the stim circuit the matrix route describes.

    One fault qubit per ``(round, data qubit)`` and one ancilla per
    ``(round, check)``, all reset in every round, so no fault survives its own
    round. A detector compares a check's ancilla with the same check one round
    earlier, which is the same statement as the model's two detector bands, and
    one observable wire accumulates the data qubits the logical operator holds.
    """

    num_checks = len(hz)
    num_qubits = len(hz[0])
    fault_base = 0
    ancilla_base = fault_base + rounds * num_qubits
    observable_wire = ancilla_base + rounds * num_checks

    circuit = stim.Circuit()
    circuit.append("R", [observable_wire])
    for round_index in range(rounds):
        faults = [fault_base + round_index * num_qubits + q for q in range(num_qubits)]
        ancillas = [
            ancilla_base + round_index * num_checks + k for k in range(num_checks)
        ]
        circuit.append("R", faults + ancillas)
        if data_flip:
            circuit.append("X_ERROR", faults, data_flip)
        if measurement_flip:
            circuit.append("X_ERROR", ancillas, measurement_flip)
        for check, row in enumerate(hz):
            for qubit, entry in enumerate(row):
                if entry:
                    circuit.append("CX", [faults[qubit], ancillas[check]])
        for qubit, entry in enumerate(lz[0]):
            if entry:
                circuit.append("CX", [faults[qubit], observable_wire])
        circuit.append("MR", ancillas)
        for check in range(num_checks):
            targets = [stim.target_rec(-num_checks + check)]
            if round_index:
                targets.append(stim.target_rec(-2 * num_checks + check))
            circuit.append("DETECTOR", targets)

    circuit.append("M", [observable_wire])
    if lz:
        circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)
    return circuit


def _stim_mechanisms(circuit: stim.Circuit) -> dict[tuple[tuple[int, ...], ...], float]:
    """Return stim's mechanisms keyed by their full signature."""

    model = circuit.detector_error_model(decompose_errors=False)
    mechanisms: dict[tuple[tuple[int, ...], ...], float] = {}
    for instruction in model.flattened():
        if instruction.type != "error":
            continue
        detectors: list[int] = []
        observables: list[int] = []
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                detectors.append(target.val)
            elif target.is_logical_observable_id():
                observables.append(target.val)
        key = (tuple(sorted(detectors)), tuple(sorted(observables)))
        mechanisms[key] = instruction.args_copy()[0]
    return mechanisms


def _model_mechanisms(
    model: DetectorErrorModel,
) -> dict[tuple[tuple[int, ...], ...], float]:
    return {
        (tuple(sorted(error.detectors)), tuple(sorted(error.observables))): (
            error.probability
        )
        for error in model.errors
    }


@pytest.mark.parametrize(
    ("name", "distance", "rounds", "data_flip", "measurement_flip"),
    _CONFIGURATIONS,
)
def test_matrix_route_matches_stim(
    name: str,
    distance: int,
    rounds: int,
    data_flip: float,
    measurement_flip: float,
) -> None:
    hz, lz = code_matrices(_code(name, distance))
    hz_list, lz_list = hz.tolist(), lz.tolist()

    model = DetectorErrorModel.from_code_matrices(
        hz=hz,
        lz=lz,
        noise=PhenomenologicalNoise(
            data_flip=data_flip, measurement_flip=measurement_flip
        ),
        num_rounds=rounds,
    )
    circuit = _reference_circuit(
        hz_list,
        lz_list,
        data_flip=data_flip,
        measurement_flip=measurement_flip,
        rounds=rounds,
    )

    assert circuit.num_detectors == model.num_detectors
    assert circuit.num_observables == model.num_observables
    ours, theirs = _model_mechanisms(model), _stim_mechanisms(circuit)
    assert set(ours) == set(theirs)
    for signature, rate in ours.items():
        # Both sides sum the same independent rates, so they differ only by the
        # last bit or two of the accumulation order.
        assert rate == pytest.approx(theirs[signature], abs=1e-15), signature


def test_matrix_route_geometry_is_not_the_memory_circuit_geometry() -> None:
    """The two routes describe different experiments, so they differ in shape."""

    code = RepetitionCode(distance=3)
    hz, lz = code_matrices(code)
    rounds = 3
    noise = PhenomenologicalNoise(data_flip=0.01)

    matrix = DetectorErrorModel.from_code_matrices(
        hz=hz, lz=lz, noise=noise, num_rounds=rounds
    )
    memory = DetectorErrorModel.from_memory_circuit(
        build_memory_circuit(code, rounds=rounds), noise=noise
    )

    # Code capacity: one detector band per round, no terminal readout.
    assert matrix.num_detectors == rounds * 2
    # The memory circuit measures its data qubits at the end, so each of the two
    # Z-type checks gains a terminal detector the matrix route does not have.
    assert memory.num_detectors == rounds * 2 + 2
    assert matrix.num_detectors != memory.num_detectors
