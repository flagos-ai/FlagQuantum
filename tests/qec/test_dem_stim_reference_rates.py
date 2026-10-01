"""Cross-check a built detector error model against stim's own error analysis.

The rotated-surface memory circuits are transcribed gate for gate into stim, with
the same phenomenological noise model, and stim builds its detector error model
from that circuit with its own error-analysis pass. The two models are then
compared on four independent statistics: the declared shape, the number of
mechanisms, every detector marginal, and every detector pair rate.

This is an independent re-derivation rather than a shared-helper cross-check:
stim never sees the FlagQuantum injection engine, the layouts, or the model, and
the FlagQuantum side never sees the stim circuit. What the two implementations
share is the circuit's gate sequence and the noise model, which is exactly the
input the comparison is about.

The pair rates are not a restatement of the marginals. FlagQuantum's model is
built from independent mechanisms, so the exact pair rate of detectors ``i`` and
``j`` is ``(1 - c_i - c_j + c_ij) / 4`` with ``c_i`` the product of ``1 - 2p``
over the mechanisms flipping ``i`` and ``c_ij`` the same product over those
flipping exactly one of the two. A model with one mechanism per detector, each
carrying that detector's exact marginal, reproduces every marginal by
construction and has independent pairs by construction. Measured at the
distance-3, three-round configuration, that model lands between 0.29 and 0.65 of
the marginal tolerance while its pair rates miss by 10.7 to 23.6 times the pair
tolerance, as the noise strength runs over the tested range. The largest true
detector-pair covariance in those configurations runs from 0.0284 to 0.1794, so
the pairs carry structure the marginals cannot see.

The marginals are only a usable comparison because a wrong rule misses the
tolerance by a wide margin. Measured against the FlagQuantum model's exact rates
at four hundred thousand shots: over the eight round configurations below, the
worst correct deviation is 0.001031 at distance three with four rounds, which is
2.79 standard errors of the largest-rate detector and the largest of thirty-two
draws; applying the data flip after the round's gates instead of before them
differs by 0.038; and a terminal detector that omits the data readout differs by
0.107. Those last two are twenty-six and seventy-two times the tolerance.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.typing import NDArray

from flagquantum.qec import DetectorErrorModel, RotatedSurfaceCode, build_memory_circuit
from flagquantum.qec.circuit import MemoryCircuit
from flagquantum.qec.dem import DemError
from flagquantum.qec.noise import PhenomenologicalNoise

if TYPE_CHECKING:
    import stim

pytestmark = pytest.mark.integration

# Stim is an optional distribution. Without it the comparison has no reference
# side, so the suite skips rather than failing: the core environment installs no
# stim and still runs the integration tier.
stim = pytest.importorskip("stim")

_SHOTS = 400_000
_SIGMA = 4.0
_PROBABILITY = 0.01

# Round counts one through four at both distances that the model builder reaches:
# the distance-5 aggregate allocation is refused and the distance-4 build did not
# finish in forty-five minutes.
_ROUND_SWEEP = [
    (2, 1),
    (2, 2),
    (2, 3),
    (2, 4),
    (3, 1),
    (3, 2),
    (3, 3),
    (3, 4),
]
_PARAMETER_SWEEP = [0.005, 0.02, 0.05]
_MARGINAL_SEEDS = [11, 12]


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


@lru_cache(maxsize=None)
def _memory(distance: int, rounds: int) -> MemoryCircuit:
    return build_memory_circuit(RotatedSurfaceCode(distance=distance), rounds=rounds)


@lru_cache(maxsize=None)
def _flagquantum_model(
    distance: int, rounds: int, probability: float
) -> DetectorErrorModel:
    return DetectorErrorModel.from_memory_circuit(
        _memory(distance, rounds),
        noise=PhenomenologicalNoise(
            data_flip=probability, measurement_flip=probability
        ),
    )


@lru_cache(maxsize=None)
def _reference_model(distance: int, rounds: int, probability: float) -> stim.Circuit:
    return _transcribe(_memory(distance, rounds), probability=probability)


@lru_cache(maxsize=None)
def _reference_sample(
    distance: int, rounds: int, probability: float, seed: int
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    sampled = (
        _reference_model(distance, rounds, probability)
        .detector_error_model(decompose_errors=False)
        .compile_sampler(seed=seed)
        .sample(_SHOTS)
    )
    return np.asarray(sampled[0]), np.asarray(sampled[1])


def _rates(bits: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Return the empirical flip rate of every column."""

    return np.asarray(bits, dtype=np.float64).mean(axis=0)


def _joint_rates(bits: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Return the empirical rate at which every pair of columns fires together."""

    counts = np.asarray(bits, dtype=np.float64)
    return counts.T @ counts / _SHOTS


def _widest_marginal_error(rates: NDArray[np.float64]) -> float:
    """Return the largest standard error over the marginal rates."""

    error = np.sqrt(np.maximum(rates * (1.0 - rates), 0.0) / _SHOTS)
    return float(np.max(error))


def _widest_joint_error(rates: NDArray[np.float64]) -> float:
    """Return the largest standard error over the pair rates, diagonal excluded."""

    error = np.sqrt(np.maximum(rates * (1.0 - rates), 0.0) / _SHOTS)
    np.fill_diagonal(error, 0.0)
    return float(np.max(error))


def _tolerance(widest_error: float) -> float:
    """Return the four-standard-error tolerance around a measured rate.

    The bound is the standard error of a rate of one half, which no sampled rate
    can exceed at this shot count. A tolerance above it would no longer be a
    sampling error, so the bound stops the comparison from silently degrading
    into an assertion that accepts a divergence of any size. The measured
    marginal tolerance is 0.001477 at the default strength and 0.002684 at the
    strongest one tested, against 0.003162; the measured pair tolerance is
    0.000672, a factor of 4.7 below it.
    """

    tolerance = _SIGMA * widest_error
    assert tolerance <= _SIGMA * math.sqrt(0.25 / _SHOTS), tolerance
    return tolerance


def _exact_joint_rates(model: DetectorErrorModel) -> NDArray[np.float64]:
    """Return the exact pair flip rate of every detector pair.

    Every mechanism fires independently, so the sign product over the mechanisms
    flipping exactly one of a pair gives the pair's parity correlation and the
    rate follows from it. The diagonal of the result is the marginal rate, which
    the test below asserts so that the derivation cannot drift from the model's
    own marginal arithmetic without being noticed.
    """

    signature = model.detector_error_matrix().numpy() != 0
    factor = np.array(
        [1.0 - 2.0 * error.probability for error in model.errors], dtype=np.float64
    )

    def product(pattern: NDArray[np.bool_]) -> NDArray[np.float64]:
        return np.prod(np.where(pattern, factor, 1.0), axis=-1)

    single = product(signature)
    both = signature[:, None, :] != signature[None, :, :]
    pair = product(both)
    return np.asarray(
        (1.0 - single[:, None] - single[None, :] + pair) / 4.0, dtype=np.float64
    )


def _independent_model(model: DetectorErrorModel) -> DetectorErrorModel:
    """Return the model that keeps every marginal and drops every correlation."""

    marginals = model.detector_rates().numpy()
    return DetectorErrorModel(
        num_detectors=model.num_detectors,
        num_observables=model.num_observables,
        errors=tuple(
            DemError(float(marginal), (index,), ())
            for index, marginal in enumerate(marginals)
        ),
    )


@pytest.mark.parametrize(("distance", "rounds"), _ROUND_SWEEP)
def test_the_reference_circuit_declares_the_same_shape(
    distance: int, rounds: int
) -> None:
    memory = _memory(distance, rounds)
    circuit = _reference_model(distance, rounds, _PROBABILITY)

    assert circuit.num_detectors == len(memory.detectors.detectors)
    assert circuit.num_observables == len(memory.observables.observables)


@pytest.mark.parametrize(("distance", "rounds"), _ROUND_SWEEP)
def test_the_reference_model_has_the_same_mechanism_count(
    distance: int, rounds: int
) -> None:
    """Both implementations enumerate one mechanism per noise location.

    The count agreeing is a measured agreement, not an identity the test
    imposes: the FlagQuantum engine forces each location through the circuit and
    merges locations with the same signature, while stim's error analysis
    propagates each location symbolically and merges the same way.
    """

    built = _flagquantum_model(distance, rounds, _PROBABILITY)
    reference = _reference_model(distance, rounds, _PROBABILITY).detector_error_model(
        decompose_errors=False
    )

    assert len(built.errors) == reference.num_errors


@pytest.mark.parametrize(("distance", "rounds"), _ROUND_SWEEP)
def test_the_reference_marginals_agree_within_its_sampling_error(
    distance: int, rounds: int
) -> None:
    """Every detector marginal matches the independent implementation's.

    The tolerance is four standard errors of the largest measured rate, which is
    the widest sampling error any single detector in these configurations has.
    Over the eight round configurations below the largest deviation the correct
    transcription produced was 2.79 standard errors, while the mis-transcriptions
    described in the module docstring deviate by 0.038 and 0.107 — twenty-six and
    seventy-two times the tolerance measured at distance three with three rounds.
    """

    built = _flagquantum_model(distance, rounds, _PROBABILITY)
    obtained = _rates(_reference_sample(distance, rounds, _PROBABILITY, 11)[0])
    expected: NDArray[np.float64] = built.detector_rates().numpy()

    assert obtained.shape == expected.shape
    tolerance = _tolerance(_widest_marginal_error(obtained))
    deviation = float(np.max(np.abs(obtained - expected)))
    assert deviation < tolerance, (
        f"largest deviation {deviation:.6f} at detector "
        f"{int(np.argmax(np.abs(obtained - expected)))} exceeds {tolerance:.6f}"
    )


@pytest.mark.parametrize(("distance", "rounds"), _ROUND_SWEEP)
def test_the_reference_detector_pair_rates_agree(distance: int, rounds: int) -> None:
    """Every detector pair fires together at the rate the independent build says.

    Pair rates are where a model that reproduced the marginals by construction
    stops agreeing, so this is the statistic that carries the comparison beyond
    the marginals.
    """

    built = _flagquantum_model(distance, rounds, _PROBABILITY)
    detectors = _reference_sample(distance, rounds, _PROBABILITY, 11)[0]
    obtained = _joint_rates(detectors)
    expected = _exact_joint_rates(built)

    assert obtained.shape == expected.shape
    assert np.allclose(np.diag(expected), built.detector_rates().numpy(), atol=1e-12)
    # The pair statistic has to resolve strictly finer than the marginal one it
    # extends; carrying the diagonal would hand it the marginal rates instead.
    assert _widest_joint_error(obtained) < _widest_marginal_error(_rates(detectors))
    tolerance = _tolerance(_widest_joint_error(obtained))
    deviation = np.abs(obtained - expected)
    np.fill_diagonal(deviation, 0.0)
    assert float(np.max(deviation)) < tolerance, (
        f"largest pair deviation {float(np.max(deviation)):.6f} at "
        f"{np.unravel_index(int(np.argmax(deviation)), deviation.shape)} "
        f"exceeds {tolerance:.6f}"
    )


@pytest.mark.parametrize("probability", [_PROBABILITY, *_PARAMETER_SWEEP])
def test_the_reference_agrees_across_seeds_and_noise_strengths(
    probability: float,
) -> None:
    """The agreement is not an artifact of one seed or one noise strength."""

    built = _flagquantum_model(3, 3, probability)
    expected_marginals: NDArray[np.float64] = built.detector_rates().numpy()
    expected_pairs = _exact_joint_rates(built)
    expected_observable = float(built.observable_rates().numpy()[0])

    for seed in _MARGINAL_SEEDS:
        detectors, observables = _reference_sample(3, 3, probability, seed)
        obtained_marginals = _rates(detectors)
        obtained_pairs = _joint_rates(detectors)

        marginal_tolerance = _tolerance(_widest_marginal_error(obtained_marginals))
        assert (
            float(np.max(np.abs(obtained_marginals - expected_marginals)))
            < marginal_tolerance
        )
        pair_tolerance = _tolerance(_widest_joint_error(obtained_pairs))
        pair_deviation = np.abs(obtained_pairs - expected_pairs)
        np.fill_diagonal(pair_deviation, 0.0)
        assert float(np.max(pair_deviation)) < pair_tolerance

        observed = float(_rates(observables)[0])
        tolerance = _tolerance(
            float(np.sqrt(max(observed * (1.0 - observed), 0.0) / _SHOTS))
        )
        assert abs(observed - expected_observable) < tolerance


@pytest.mark.parametrize("probability", [_PROBABILITY, *_PARAMETER_SWEEP])
def test_the_pair_rates_separate_a_model_the_marginals_cannot(
    probability: float,
) -> None:
    """A model that keeps every marginal and drops every correlation.

    One mechanism per detector carrying that detector's exact marginal reproduces
    the marginals by construction and treats the detectors as independent. The
    marginals therefore cannot separate it from the built model, and the pair
    rates do. The measured margins are recorded in the module docstring.
    """

    built = _flagquantum_model(3, 3, probability)
    independent = _independent_model(built)
    detectors, _ = _reference_sample(3, 3, probability, 11)
    obtained_marginals = _rates(detectors)
    obtained_pairs = _joint_rates(detectors)

    marginal_deviation = float(
        np.max(np.abs(obtained_marginals - independent.detector_rates().numpy()))
    )
    assert marginal_deviation < _tolerance(_widest_marginal_error(obtained_marginals))

    pair_deviation = np.abs(obtained_pairs - _exact_joint_rates(independent))
    np.fill_diagonal(pair_deviation, 0.0)
    tolerance = _tolerance(_widest_joint_error(obtained_pairs))
    assert float(np.max(pair_deviation)) > 2.0 * tolerance
