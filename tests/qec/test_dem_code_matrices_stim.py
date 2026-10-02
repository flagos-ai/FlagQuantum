"""Cross-check the matrix route against stim's own error analysis.

The matrix route states an experiment, so the statement can be handed to an
independent implementation. The circuit below is that experiment and nothing
more: one persisting data qubit per matrix column, one ancilla per check, one
syndrome extraction per round, the detectors declared as the difference between
an extraction and the extraction before it, and the data measured in one basis
at the end. stim derives the detectors, the observables and the rates from it
with its own error-analysis pass.

Two properties of the experiment make it a faithful translation of the model.
The extraction before round zero is noiseless and declares no detector: it
supplies the prior every round-zero detector is compared against, and it is the
only place a prior can come from for an X-type check, because an X-type ancilla
measures a random value on a register that has never been measured in that basis
and only becomes deterministic once the earlier measurement has prepared the
shared eigenstate. And the data qubits persist across rounds, which is what
makes a fault keep flipping the syndrome of every later extraction.

The readout is one basis because a single run of a memory experiment measures
one of the code's two logical operators: ``lz`` and ``lx`` anti-commute, so no
one state has both as a deterministic value. The model states both blocks, and
the reference side therefore runs the experiment twice -- once per readout
basis, against the model restricted to that basis -- instead of pretending to
read both at once.

The persisting data is also where the model and the experiment part company, and
the split is a *tested fact* here rather than an unexamined gap. The model gives
every data fault in round ``r`` the detector band of round ``r`` and the band of
round ``r + 1``. A fault that still reaches the logical measurement at the end
of the run is still in the data, so it flips the syndrome of every extraction
from ``r`` onward, and consecutive differences cancel everywhere except in band
``r``: a physical fault spans one band. The two agree exactly where the model's
extra band has nothing to claim -- in the final round, which has no band after
it, and in every round of a one-round run -- and they differ by that one extra
band for a data fault in an earlier round of a longer run.

The transcription in :mod:`tests.qec.test_dem_stim_reference_rates` is a
different circuit on purpose. It is the memory circuit this package builds, and
its detectors include the terminal comparison between the last syndrome of each
Z-type check and the data readout. The matrix route has no terminal readout, so
the two routes disagree about the geometry and are compared to stim separately.
Conflating them would hide which of the two a failure belongs to.

The comparison is on the declared shape and on every mechanism's full signature
and rate. The rates are stim's own, so a rate this package derived from the
wrong fault family shows up as a difference rather than being checked against
itself.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from flagquantum.qec import (
    CssCodeMatrices,
    DetectorErrorModel,
    PhenomenologicalNoise,
    RepetitionCode,
    RotatedSurfaceCode,
    SteaneCode,
    build_memory_circuit,
    css_code_matrices,
)

# Stim is an optional distribution and the reference side needs it, so the suite
# skips rather than fails when it is absent.
pytest.importorskip("stim")

import stim

pytestmark = pytest.mark.integration

# Name, distance, rounds, data rate, phase rate, Y rate, measurement rate. The
# three data rates are separate so a fault family the model confuses with either
# of the other two is caught. Every configuration below is one the model and a
# physical experiment agree on: a single round, where the model's extra band has
# nothing to claim, or a longer run whose only faults are measurement faults,
# which do span two bands in both.
_CONFIGURATIONS = [
    ("repetition-3", 3, 1, 0.05, 0.0, 0.0, 0.05),
    ("repetition-3", 3, 1, 0.02, 0.03, 0.01, 0.02),
    ("repetition-5", 5, 1, 0.01, 0.0, 0.0, 0.03),
    ("surface-2", 2, 1, 0.03, 0.02, 0.01, 0.02),
    ("surface-3", 3, 1, 0.01, 0.01, 0.005, 0.01),
    ("steane", 3, 1, 0.01, 0.02, 0.005, 0.03),
    ("repetition-3-measurement-only", 3, 3, 0.0, 0.0, 0.0, 0.06),
    ("surface-3-measurement-only", 3, 2, 0.0, 0.0, 0.0, 0.03),
    ("steane-measurement-only", 3, 2, 0.0, 0.0, 0.0, 0.02),
]


def _code(name: str, distance: int) -> Any:
    if name.startswith("surface"):
        return RotatedSurfaceCode(distance=distance)
    if name.startswith("steane"):
        return SteaneCode()
    return RepetitionCode(distance=distance)


def _readout_bases(matrices: CssCodeMatrices) -> list[str]:
    """Return the readout bases the code declares a logical operator in."""

    bases = []
    if matrices.num_z_logicals:
        bases.append("z")
    if matrices.num_x_logicals:
        bases.append("x")
    return bases


def _one_basis(matrices: CssCodeMatrices, basis: str) -> CssCodeMatrices:
    """Return the model's matrices with the other readout basis dropped."""

    if basis == "z":
        return CssCodeMatrices(hz=matrices.hz, hx=matrices.hx, lz=matrices.lz)
    return CssCodeMatrices(hz=matrices.hz, hx=matrices.hx, lx=matrices.lx)


def _reference_circuit(
    matrices: CssCodeMatrices,
    *,
    data_flip: float,
    phase_flip: float,
    both_flip: float,
    measurement_flip: float,
    rounds: int,
    basis: str,
) -> stim.Circuit:
    """Return the stim circuit the matrix route describes in one basis.

    A Z-type check couples its support into a ``|0>`` ancilla, an X-type check
    prepares ``|+>`` and couples out of it, and a measurement fault is a bit flip
    of the recorded outcome rather than of the state, so it flips this round's
    detector and the next round's and never a logical operator.
    """

    hz, hx = matrices.hz.tolist(), matrices.hx.tolist()
    logicals = (matrices.lz if basis == "z" else matrices.lx).tolist()
    num_qubits = matrices.num_qubits
    num_z_checks = len(hz)
    num_checks = matrices.num_checks
    z_ancilla = num_qubits
    x_ancilla = z_ancilla + num_z_checks

    circuit = stim.Circuit()
    circuit.append("R", list(range(num_qubits)))
    if basis == "x":
        # The readout basis is also the preparation basis: a register prepared
        # in ``|0>`` leaves the X-type logical operator a coin toss, so a run
        # that reads the X-type observable starts from ``|+>`` instead.
        circuit.append("H", list(range(num_qubits)))

    def extract(*, noisy: bool, detectors: bool) -> None:
        for check, row in enumerate(hz):
            ancilla = z_ancilla + check
            circuit.append("R", [ancilla])
            for qubit, entry in enumerate(row):
                if entry:
                    circuit.append("CX", [qubit, ancilla])
            if noisy and measurement_flip:
                circuit.append("X_ERROR", [ancilla], measurement_flip)
            circuit.append("M", [ancilla])
        for check, row in enumerate(hx):
            ancilla = x_ancilla + check
            circuit.append("R", [ancilla])
            circuit.append("H", [ancilla])
            for qubit, entry in enumerate(row):
                if entry:
                    circuit.append("CX", [ancilla, qubit])
            circuit.append("H", [ancilla])
            if noisy and measurement_flip:
                circuit.append("X_ERROR", [ancilla], measurement_flip)
            circuit.append("M", [ancilla])
        if not detectors:
            return
        # The record holds this extraction's checks in order, Z-type first, so
        # check ``k`` is ``num_checks`` back and its predecessor is twice that.
        for check in range(num_checks):
            targets = [stim.target_rec(-num_checks + check)]
            targets.append(stim.target_rec(-2 * num_checks + check))
            circuit.append("DETECTOR", targets)

    # The preparation extraction: it states the prior value of every check and
    # declares no detector, and it carries no fault of its own, because the
    # model's round-zero detectors are compared against a check that is right.
    extract(noisy=False, detectors=False)
    for _ in range(rounds):
        data = list(range(num_qubits))
        if data_flip:
            circuit.append("X_ERROR", data, data_flip)
        if phase_flip:
            circuit.append("Z_ERROR", data, phase_flip)
        if both_flip:
            # The Y member of the single-qubit Pauli channel, stated on its own
            # so the rate is the Y rate and not a mixture of the other two.
            for qubit in data:
                circuit.append("PAULI_CHANNEL_1", [qubit], [0.0, both_flip, 0.0])
        extract(noisy=True, detectors=True)

    # One product measurement per logical operator, in the model's row order.
    # Each readout is a single measurement, which is what
    # ``target_combined_paulis`` states -- a bare list of targets would be one
    # measurement per qubit instead.
    for row in logicals:
        support = [qubit for qubit, entry in enumerate(row) if entry]
        target = stim.target_z if basis == "z" else stim.target_x
        circuit.append(
            "MPP",
            stim.target_combined_paulis([target(qubit) for qubit in support]),
        )
    for index in range(len(logicals)):
        circuit.append(
            "OBSERVABLE_INCLUDE",
            [stim.target_rec(-len(logicals) + index)],
            index,
        )
    return circuit


def _signature(error: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return tuple(sorted(error.detectors)), tuple(sorted(error.observables))


def _stim_mechanisms(
    circuit: stim.Circuit,
) -> dict[tuple[tuple[int, ...], tuple[int, ...]], float]:
    """Return stim's mechanisms keyed by their full signature."""

    model = circuit.detector_error_model(decompose_errors=False)
    mechanisms: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}
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
) -> dict[tuple[tuple[int, ...], tuple[int, ...]], float]:
    return {_signature(error): error.probability for error in model.errors}


def _noise(
    data_flip: float, phase_flip: float, both_flip: float, measurement_flip: float
) -> PhenomenologicalNoise:
    return PhenomenologicalNoise(
        data_flip=data_flip,
        phase_flip=phase_flip,
        both_flip=both_flip,
        measurement_flip=measurement_flip,
    )


def _agreements(
    matrices: CssCodeMatrices,
    *,
    rounds: int,
    noise: PhenomenologicalNoise,
    basis: str,
    data_flip: float,
    phase_flip: float,
    both_flip: float,
    measurement_flip: float,
) -> tuple[
    dict[tuple[tuple[int, ...], tuple[int, ...]], float],
    dict[tuple[tuple[int, ...], tuple[int, ...]], float],
    int,
    int,
]:
    """Return both sides of the comparison for one code, run and readout basis."""

    restricted = _one_basis(matrices, basis)
    model = DetectorErrorModel.from_code_matrices(
        restricted, noise=noise, num_rounds=rounds
    )
    circuit = _reference_circuit(
        matrices,
        data_flip=data_flip,
        phase_flip=phase_flip,
        both_flip=both_flip,
        measurement_flip=measurement_flip,
        rounds=rounds,
        basis=basis,
    )
    return (
        _model_mechanisms(model),
        _stim_mechanisms(circuit),
        model.num_detectors,
        model.num_observables,
    )


@pytest.mark.parametrize(
    (
        "name",
        "distance",
        "rounds",
        "data_flip",
        "phase_flip",
        "both_flip",
        "measurement_flip",
    ),
    _CONFIGURATIONS,
)
def test_matrix_route_matches_a_physical_experiment(
    name: str,
    distance: int,
    rounds: int,
    data_flip: float,
    phase_flip: float,
    both_flip: float,
    measurement_flip: float,
) -> None:
    matrices = css_code_matrices(_code(name, distance))
    noise = _noise(data_flip, phase_flip, both_flip, measurement_flip)

    for basis in _readout_bases(matrices):
        ours, theirs, num_detectors, num_observables = _agreements(
            matrices,
            rounds=rounds,
            noise=noise,
            basis=basis,
            data_flip=data_flip,
            phase_flip=phase_flip,
            both_flip=both_flip,
            measurement_flip=measurement_flip,
        )
        assert set(ours) == set(theirs), basis
        for signature, rate in ours.items():
            # Both sides combine the same independent rates, so they differ only
            # by the last bit or two of the accumulation order.
            assert rate == pytest.approx(theirs[signature], abs=1e-15), (
                basis,
                signature,
            )
        # The shape the model declares is the shape stim derived, not a shape
        # this test read back off the model alone.
        assert num_observables == 1


def test_matrix_route_reads_each_basis_from_its_own_observable_block() -> None:
    """The Steane code needs two runs, and each one pins one observable block.

    ``lz`` and ``lx`` anti-commute, so a single state has no deterministic value
    for both and the model cannot be compared to one circuit. Each run reads one
    of them, and the fault family that run gives an observable row to is the one
    the Pauli character of that family predicts: an X fault is what a Z-type
    logical operator sees, a Z fault is what an X-type one sees, and a Y fault is
    both. The detector block a fault reaches is the same in either run, which is
    what makes the observable row the only thing the readout basis changes.
    """

    data_flip, phase_flip, both_flip = 0.01, 0.02, 0.005
    matrices = css_code_matrices(SteaneCode())
    seen: dict[str, dict[tuple[tuple[int, ...], tuple[int, ...]], float]] = {}
    for basis in ("z", "x"):
        ours, theirs, _, num_observables = _agreements(
            matrices,
            rounds=1,
            noise=_noise(data_flip, phase_flip, both_flip, 0.0),
            basis=basis,
            data_flip=data_flip,
            phase_flip=phase_flip,
            both_flip=both_flip,
            measurement_flip=0.0,
        )
        assert set(ours) == set(theirs)
        assert num_observables == 1
        seen[basis] = ours

    def family(run: dict[Any, float], rate: float) -> set[Any]:
        return {signature for signature, value in run.items() if value == rate}

    for basis in ("z", "x"):
        # An X fault carries the Z-type logical operator, a Z fault the X-type
        # one, and a Y fault both, so the readout basis decides which of them
        # finds an observable row and which one is detector-only.
        data = family(seen[basis], data_flip)
        phase = family(seen[basis], phase_flip)
        both = family(seen[basis], both_flip)
        assert data and phase and both
        # A Y fault is both families, so it carries an observable row wherever
        # its support meets the logical operator -- in either run -- and a fault
        # outside that support moves detectors only.
        assert {s[1] for s in both} == {(), (0,)}
        assert {s[1] for s in data} == ({(), (0,)} if basis == "z" else {()})
        assert {s[1] for s in phase} == ({(), (0,)} if basis == "x" else {()})

    # The detector block a fault family reaches is the same in both runs: the Z
    # run's copy of a family flips the detectors the X run's copy flips, and
    # only the observable rows differ.
    for rate in (data_flip, phase_flip, both_flip):
        assert {s[0] for s in family(seen["z"], rate)} == {
            s[0] for s in family(seen["x"], rate)
        }


def test_matrix_route_doubles_a_data_fault_into_the_band_after_it() -> None:
    """A fault that cannot persist is the one place the model adds a band.

    A data fault in an early round of a run longer than one round is the single
    case where the model and the experiment disagree, so it is pinned here with
    the exact relationship between them rather than left implicit. The model
    claims the band of the fault and the band after it; the experiment shows the
    band of the fault alone, because a fault that still reaches the final
    logical measurement is still in the data and therefore cancels out of every
    later difference.
    """

    matrices = css_code_matrices(RepetitionCode(distance=3))
    rounds = 2
    band = matrices.num_checks
    model = DetectorErrorModel.from_code_matrices(
        matrices, noise=_noise(0.04, 0.0, 0.0, 0.0), num_rounds=rounds
    )
    circuit = _reference_circuit(
        matrices,
        data_flip=0.04,
        phase_flip=0.0,
        both_flip=0.0,
        measurement_flip=0.0,
        rounds=rounds,
        basis="z",
    )
    ours, theirs = _model_mechanisms(model), _stim_mechanisms(circuit)

    # Every mechanism the experiment states lies inside one round's band.
    assert all(max(detectors) - min(detectors) < band for detectors, _ in theirs)
    assert theirs == {
        ((0,), (0,)): 0.04,
        ((0, 1), (0,)): 0.04,
        ((1,), (0,)): 0.04,
        ((2,), (0,)): 0.04,
        ((2, 3), (0,)): 0.04,
        ((3,), (0,)): 0.04,
    }
    # The model's round-zero faults are the experiment's round-zero faults with
    # the band after them added, and its final-round faults are unchanged
    # because the final round has no band after it.
    early = {(d, o) for d, o in theirs if max(d) < band}
    doubled = {
        (
            tuple(sorted(set(detectors) | {row + band for row in detectors})),
            observables,
        )
        for detectors, observables in early
    }
    assert set(ours) == (set(theirs) - early) | doubled
    assert set(ours) - set(theirs) == {
        ((0, 1, 2, 3), (0,)),
        ((0, 2), (0,)),
        ((1, 3), (0,)),
    }
    assert set(ours.values()) == {0.04}


def test_matrix_route_reaches_a_logical_operator_from_the_final_round() -> None:
    """The final round is where a fault and the logical readout meet.

    A fault in the final round has no later extraction to be cancelled by, so
    the model and the experiment agree there and both report the logical
    operator the fault flips. That is what makes the divergence above a
    statement about the extra band and not about the observable rows.
    """

    matrices = css_code_matrices(RepetitionCode(distance=3))
    model = DetectorErrorModel.from_code_matrices(
        matrices, noise=_noise(0.04, 0.0, 0.0, 0.0), num_rounds=2
    )
    final_band = {
        _signature(error)
        for error in model.errors
        if min(error.detectors) >= matrices.num_checks
    }

    # Each of the three data wires ends the run with the logical operator of the
    # repetition code, whichever check it touches, and none of them is empty.
    assert final_band
    assert all(observables == (0,) for _, observables in final_band)
    assert len(final_band) == matrices.num_qubits


def test_matrix_route_geometry_is_not_the_memory_circuit_geometry() -> None:
    """The two routes describe different experiments, so they differ in shape."""

    code = RepetitionCode(distance=3)
    rounds = 3
    noise = PhenomenologicalNoise(data_flip=0.01)

    matrix = DetectorErrorModel.from_code_matrices(
        css_code_matrices(code), noise=noise, num_rounds=rounds
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


def test_x_type_detectors_need_the_shared_preparation() -> None:
    """A round-zero X-type detector cannot be read against a fresh register.

    The X-type ancilla is prepared in ``|+>``, so on a register that has never
    been measured in that basis its outcome is a coin toss and a detector
    comparing that outcome with the preparation is not a detector at all. The
    reference circuit supplies the prior by extracting every check once before
    round zero, which is also how a real run prepares the shared eigenstate.
    """

    matrices = css_code_matrices(SteaneCode())
    naive = stim.Circuit()
    naive.append("R", list(range(7)))
    naive.append("R", [7])
    naive.append("H", [7])
    for qubit, entry in enumerate(matrices.hx.tolist()[0]):
        if entry:
            naive.append("CX", [7, qubit])
    naive.append("H", [7])
    naive.append("M", [7])
    naive.append("DETECTOR", [stim.target_rec(-1)])

    with pytest.raises(ValueError, match="deterministic"):
        naive.detector_error_model(decompose_errors=False)

    # The same check read against the preparation extraction is deterministic,
    # and it stays deterministic for an X-type check without any reset between
    # the two extractions: the first measurement is what prepared the eigenstate.
    shared = _reference_circuit(
        matrices,
        data_flip=0.0,
        phase_flip=0.0,
        both_flip=0.0,
        measurement_flip=0.0,
        rounds=1,
        basis="z",
    )
    assert shared.num_detectors == matrices.num_checks
    assert shared.detector_error_model(decompose_errors=False).num_detectors == 6


def test_one_basis_matrices_keep_the_other_check_block() -> None:
    """Dropping a readout basis drops an observable block and nothing else."""

    matrices = css_code_matrices(SteaneCode())
    restricted = _one_basis(matrices, "z")

    assert torch.equal(restricted.hz, matrices.hz)
    assert torch.equal(restricted.hx, matrices.hx)
    assert torch.equal(restricted.lz, matrices.lz)
    assert restricted.num_x_logicals == 0
    assert restricted.num_checks == matrices.num_checks
