"""Execution coverage for the ZXXZ surface patch, and the limit it runs into.

``test_toric_memory_execution.py`` runs the same machinery on a family whose
checks are pure, where the memory experiment has a detector anchored to the
prepared state and to the terminal readout, and where a matching decoder turns a
raw logical rate of a few per cent into a small fraction of one. This file runs it
on the family whose every check is mixed, and the difference is the subject.

The record's own claim -- same wires, same weights, same distance as the rotated
patch -- is a statement about the *code*. It is not a statement about the memory
experiment built on it, and this file is where the two come apart. A mixed check
measures the product of an X factor and a Z factor on one ancilla, so its outcome
is deterministic neither at the all-zero preparation nor at a Z-basis readout. A
check that is deterministic in neither place gets no detector comparing it with
the prepared state, and a check with no such anchor also gets no terminal detector
comparing it with the readout, so the Z-basis memory experiment of this family
carries detectors only *between* syndrome rounds and none at all from a syndrome
round to the data. That is measured below rather than argued, and it is the reason
the decoded rate is dominated by something no decoder can see.

What this file proves
---------------------

1. ``test_the_memory_experiment_declares_one_observable_and_its_own_detectors``:
   the Z-basis experiment reads out one observable whose support is the declared
   Z-type logical operator, and it carries exactly ``(d ** 2 - 1) * (rounds - 1)``
   detectors -- the round-to-round comparisons alone, with no round-zero and no
   terminal detector. The X-basis experiment on the same code, which is anchored
   by the two checks the readout plan leaves in their own basis, carries strictly
   more, and the difference is the measurement rather than the arithmetic.
2. ``test_the_readout_is_compared_with_no_syndrome_round``: no detector of the
   Z-basis experiment references a data readout at all, while the X-basis
   experiment's detectors do. This is the defect stated in the one form that does
   not depend on a count.
3. ``test_the_model_carries_a_mechanism_that_fires_no_detector``: exactly one
   mechanism of every Z-basis model fires no detector and flips the observable, at
   three distances and three rates, and its probability is the parity of the ``d``
   readout wires on the logical operator's support, ``(1 - (1 - 2 * p) ** d) / 2``
   rather than ``d * p``, which agrees only to first order. It is the readout, and
   its probability grows with the distance.
4. ``test_the_noise_declaration_decides_whether_the_model_is_graphlike``: the same
   code is graphlike without a Y fault and not graphlike with one, at every
   distance and round count in this file, and the graphlike side is handed to the
   matching decoder so that "graphlike" is the decoder's acceptance rather than a
   histogram read.
5. ``test_the_circuit_sampler_and_the_model_agree_on_the_observable``: the model's
   per-detector and per-observable marginals match the executed circuit, so the
   model is the program a shot actually runs and the rows above are about that
   program.
6. ``test_every_single_mechanism_is_explained_and_some_are_misread``: every
   mechanism's syndrome is reproduced by the correction, and the observable
   predictions of some are not. The mechanism counts are recorded and the two
   halves are asserted against each other, so a decoder that reproduced nothing
   and one that misread nothing would both fail here. The misreads are located:
   some come from two mechanisms sharing a syndrome and disagreeing about the
   observable, and at least one does not.
7. ``test_the_decoder_always_returns_a_correction_that_reproduces_the_syndrome``:
   every syndrome the model itself samples is decodable and every correction
   reproduces it exactly, at two distances and a large shot budget.
8. ``test_matching_barely_moves_the_logical_rate_on_this_patch``: the measured
   finding, with the rotated patch at the same distance as the control. On the
   ZXXZ patch the decoded rate stays above ``raw / 1.5`` at both distances, and on
   the rotated patch it falls below ``raw / 4`` at both. The assertion is that
   contrast rather than a threshold, and it is the evidence that the shortfall is
   a property of this record's memory experiment rather than of the harness.

What this file does not prove
-----------------------------

It makes no threshold, suppression or scaling claim, and it records point 8 as a
limitation rather than as a result: the number of shots at which the ZXXZ patch's
decoded rate exceeds the rotated patch's by an order of magnitude is a statement
about an unprotected readout, not about a decoder's quality. Neither patch is
compared against one at a different distance to argue about scaling.

The blind mechanism in point 3 is a property of the *record as built*, and this
file does not claim it is irreducible: a detector comparing a mixed check's
Z factor with a Z-basis readout would compare the readout with the syndrome, and
building one is a change to the detector layout that this file measures the need
for rather than making. Until that exists the Z-basis memory experiment of this
family is not a distance-``d`` experiment in the sense the rotated patch's is, and
the record says so.

Nothing here is executed through the trajectory simulator. The patch carries
``d ** 2`` data wires and ``d ** 2 - 1`` ancillas, which is 17 qubits at the
smallest distance this file builds and 97 at the next one, so a distribution over
its full output is not materialised. Every execution row goes through the
stabilizer sampler, which does not.

"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from functools import lru_cache

import pytest

from flagquantum.errors import CapabilityError
from flagquantum.qec import RotatedSurfaceCode, ZxxzSurfaceCode
from flagquantum.qec.circuit import MemoryCircuit, build_memory_circuit
from flagquantum.qec.decoding_graph import DecodingGraph, DecodingGraphEdge
from flagquantum.qec.dem import DetectorErrorModel
from flagquantum.qec.matching import MinimumWeightMatchingDecoder
from flagquantum.qec.noise import PhenomenologicalNoise
from flagquantum.qec.sampling import sample_memory_circuit

pytestmark = pytest.mark.integration

# Seed for every draw in this file, so the sampled rows are reproducible.
_SEED = 0

# The fault rate the profile, the blind-mechanism and the sampler rows run at.
_PROBABILITY = 0.05

# The faults are declared on both data bases and on the measurements, so a data
# wire is wrong in either basis at this rate and so is a syndrome bit.
_UNIFORM = (0.05, 0.05, 0.05)

# A sampled rate is accepted when it sits within this many binomial standard
# errors of the modelled rate. Four covers a per-index miss probability of about
# 6e-5, and the comparison is per index, so one wrong mechanism is located by the
# failure message rather than hidden in a summary statistic.
_RATE_TOLERANCE_SIGMAS = 4.0

# Shot budgets. The sampler lowers and executes once per call on the stabilizer
# engine, and the decode loop is one exact pairing per shot.
_SAMPLED_SHOTS = 4000
_DECODING_SHOTS = 4000

# The noise the decoded-rate contrast is measured at, and the shot budget for it.
# The budget is larger than the other rows' because this row has to separate two
# decoded rates that differ by about a third rather than by an order of magnitude.
_LOW_NOISE = 0.01
_CONTRAST_SHOTS = 20000

# The two distances that row is measured at, and the margin the ZXXZ patch does
# *not* clear. The measured values at this rate and budget are 0.0593 raw against
# 0.0527 decoded at distance three and 0.0917 against 0.0728 at distance five, so
# the patch sits inside a factor of 1.13 and 1.26 of its own raw rate; the rotated
# patch is 0.0573 against 0.0053 and 0.0921 against 0.0013, which is a factor of
# 10.9 and 70.8. The margins bound that contrast rather than those six numbers.
_CONTRAST_DISTANCES = (3, 5)
_ZXXZ_CEILING = 1.5
_ROTATED_FLOOR = 4.0

# The model profile: distance, round count, detector count, the mechanism weight
# histogram of the model whose noise declares a Y fault, and the same histogram
# for the model whose noise does not. The detector count is the check count times
# the number of round boundaries, which is asserted against the record's own check
# count as well. The histograms are recorded measurements rather than formulas:
# the weight-zero mechanism is the readout, the odd weights are the merged
# mechanisms of the last round's boundary, and the weight-four mechanisms appear
# only once the Y fault's two halves land on different round boundaries.
_MODEL_PROFILE = (
    (3, 2, 8, {0: 1, 1: 10, 2: 10, 3: 4, 4: 1}, {0: 1, 1: 10, 2: 6}),
    (3, 3, 16, {0: 1, 1: 20, 2: 28, 3: 8, 4: 2}, {0: 1, 1: 20, 2: 20}),
    (5, 2, 24, {0: 1, 1: 26, 2: 34, 3: 12, 4: 9}, {0: 1, 1: 26, 2: 30}),
    (5, 3, 48, {0: 1, 1: 52, 2: 92, 3: 24, 4: 18}, {0: 1, 1: 52, 2: 84}),
)


@lru_cache(maxsize=None)
def _patch(distance: int) -> ZxxzSurfaceCode:
    """Return the patch once per distance, because the checks are rebuilt per use."""

    return ZxxzSurfaceCode(distance)


@lru_cache(maxsize=None)
def _rotated(distance: int) -> RotatedSurfaceCode:
    """Return the rotated patch of the same distance."""

    return RotatedSurfaceCode(distance)


def _noise(*, y_fault: bool, rate: float = _PROBABILITY) -> PhenomenologicalNoise:
    """Return the depolarizing noise the profile rows are recorded under.

    ``both_flip`` is the Y data fault. It is the only field that separates the two
    sides of the profile, so the helper makes that explicit rather than hiding it
    in two nearly identical literals.
    """

    return PhenomenologicalNoise(
        data_flip=rate,
        phase_flip=rate,
        both_flip=rate if y_fault else 0.0,
        measurement_flip=rate,
    )


def _weights(model: DetectorErrorModel) -> dict[int, int]:
    """Return the mechanism weight histogram of one model."""

    return dict(sorted(Counter(len(error.detectors) for error in model.errors).items()))


def _readout_references(memory: MemoryCircuit) -> int:
    """Return how many detector parity terms read the data rather than a syndrome.

    A detector's parity is a list of measurement references, and a reference with
    no round index is the terminal data readout rather than a syndrome bit. A
    detector carrying none of them cannot see a readout fault at all.
    """

    return sum(
        1
        for detector in memory.detectors.detectors
        for reference in detector.parity
        if reference.round_index is None
    )


def _tolerance(predicted: float, *, shots: int) -> float:
    """Return the band a sampled rate may sit in, in binomial standard errors."""

    rate = max(predicted, 1.0 / shots)
    return _RATE_TOLERANCE_SIGMAS * math.sqrt(rate * (1.0 - rate) / shots)


def _odd_degree(graph: DecodingGraph, edges: Iterable[DecodingGraphEdge]) -> set[int]:
    """The detectors an odd number of the given edges flip.

    A matching correction is an edge set rather than a mechanism list, so this is
    how a caller checks that it reproduces a syndrome; it is the same reader
    ``test_matching_decoder.py`` uses.
    """

    parity: set[int] = set()
    for edge in edges:
        for node in edge.detectors:
            if node == graph.boundary_node:
                continue
            parity ^= {node}
    return parity


def _decoded_rate(
    code: ZxxzSurfaceCode | RotatedSurfaceCode,
    *,
    distance: int,
    rounds: int,
    shots: int,
) -> tuple[float, float]:
    """Return the raw and decoded logical rates of one patch in its Z frame.

    The decoded side flips the raw side exactly when the correction and the draw
    disagree about the observable, which is the number of predicted observable
    flips taken modulo two against the draw's own bit. Reading the correction's
    *count* of flipped observables is what makes that a parity: the sum of the
    flipped indices would be identically zero on a one-observable layout and would
    report the raw rate as its own decoded rate.
    """

    memory = build_memory_circuit(code, rounds=rounds)
    noise = _noise(y_fault=False, rate=_LOW_NOISE)
    model = DetectorErrorModel.from_memory_circuit(memory, noise=noise)
    decoder = MinimumWeightMatchingDecoder.from_detector_error_model(model)
    sample = sample_memory_circuit(memory, noise=noise, shots=shots, seed=_SEED)
    detectors: list[list[int]] = sample.detectors.tolist()
    observables: list[list[int]] = sample.observables.tolist()
    raw = decoded = 0
    for detector_row, observable_row in zip(detectors, observables, strict=True):
        events = tuple(index for index, bit in enumerate(detector_row) if bit)
        result = decoder.decode(events)
        raw += observable_row[0]
        decoded += (len(result.observables) % 2) ^ observable_row[0]
    assert model.num_observables == 1
    return raw / shots, decoded / shots


@pytest.mark.parametrize("distance", (3, 5))
def test_the_memory_experiment_declares_one_observable_and_its_own_detectors(
    distance: int,
) -> None:
    """The Z frame reads one observable and carries no anchored detector.

    The detector count is the record's own check count times the number of round
    boundaries, and the observable's support is the declared Z-type logical
    operator rather than a weight-``d`` string of the right length. The X frame of
    the same code is measured alongside: the readout plan leaves two of its checks
    in the basis they are measured in, those two are anchored, and the count is
    therefore strictly larger. Without the second frame the first count would be
    consistent with a layout that simply has fewer checks.
    """

    code = _patch(distance)
    checks = len(code.checks)
    z_observable = code.logical_observables[0]
    x_observable = code.logical_observables[1]

    for rounds in (2, 3):
        z_frame = build_memory_circuit(code, rounds=rounds)
        x_frame = build_memory_circuit(code, rounds=rounds, product=x_observable)

        assert len(z_frame.detectors) == checks * (rounds - 1)
        assert len(x_frame.detectors) == 2 * (rounds + 1) + (checks - 2) * (rounds - 1)
        assert len(x_frame.detectors) > len(z_frame.detectors)
        assert len(z_frame.observables) == len(x_frame.observables) == 1
        assert [
            {
                reference.wire
                for reference in observable.measurement_parity
                if reference.round_index is None
            }
            for observable in z_frame.observables.observables
        ] == [set(z_observable.support)]
        assert [
            {
                reference.wire
                for reference in observable.measurement_parity
                if reference.round_index is None
            }
            for observable in x_frame.observables.observables
        ] == [set(x_observable.support)]

    with pytest.raises(ValueError, match="deterministic detector"):
        build_memory_circuit(code, rounds=1)
    assert (
        len(build_memory_circuit(code, rounds=1, product=x_observable).detectors) == 4
    )


@pytest.mark.parametrize("distance", (3, 5))
def test_the_readout_is_compared_with_no_syndrome_round(distance: int) -> None:
    """No detector of the Z frame reads the data, and the X frame's do.

    This is the same fact as the count above in the form that does not depend on
    an arithmetic identity: a detector that mentions no readout reference cannot
    fire for a readout fault, whatever the layout's detector count is. The X frame
    is the control, because the layout machinery is capable of emitting such a
    reference and does so exactly for the checks it leaves in their own basis --
    one terminal detector per protected check, whose parity is the check's final
    syndrome round together with the readout of every wire the check carries. The
    reference count is read off the record's own protected checks rather than
    pinned as a constant, so it is the rule being measured and not the number.
    """

    code = _patch(distance)
    rounds = 3
    z_frame = build_memory_circuit(code, rounds=rounds)
    x_frame = build_memory_circuit(
        code, rounds=rounds, product=code.logical_observables[1]
    )
    protected = tuple(
        check
        for check in code.checks
        if all(
            (wire in code.logical_observables[1].x_wires)
            == (wire in check.stabilizer.x_wires)
            for wire in check.stabilizer.support
        )
    )

    assert _readout_references(z_frame) == 0
    assert 0 < len(protected) < len(code.checks)
    assert _readout_references(x_frame) == sum(
        len(check.stabilizer.support) for check in protected
    )
    assert sum(len(check.stabilizer.support) for check in protected) == 4


@pytest.mark.parametrize("distance", (3, 5, 7))
@pytest.mark.parametrize("rate", (0.005, 0.01, 0.02))
def test_the_model_carries_a_mechanism_that_fires_no_detector(
    distance: int, rate: float
) -> None:
    """Exactly one mechanism flips the observable and fires no detector.

    It is the readout: the observable is the product of the ``d`` data wires on the
    logical operator's support, each read out at this rate, and the ``d`` faults
    merge into the single mechanism that fires when an odd number of them do --
    probability ``(1 - (1 - 2 * p) ** d) / 2``. Both sides of the assertion are
    stated, because a model with no such mechanism and a model with two of them are
    different failures, and the closed form is read off the record's own observable
    support rather than a distance the record declares.

    The rate is checked exactly rather than sampled, since the model is exact, and
    the merged probability is the merge rule's rather than ``d * p``: the two agree
    only to first order, and it is the merged form a decoder would be handed. The
    three distances make the growth in ``d`` the measurement: a channel whose
    probability rises with the distance is not one a decoder can suppress by
    pairing detectors, because no detector is involved.
    """

    memory = build_memory_circuit(_patch(distance), rounds=2)
    model = DetectorErrorModel.from_memory_circuit(
        memory, noise=_noise(y_fault=False, rate=rate)
    )
    support = set(_patch(distance).logical_observables[0].support)
    blind = [
        error
        for error in model.errors
        if not len(error.detectors) and len(error.observables)
    ]

    assert model.num_detectors == len(_patch(distance).checks)
    assert len(blind) == 1
    assert tuple(int(index) for index in blind[0].observables) == (0,)
    assert float(blind[0].probability) == pytest.approx(
        (1.0 - (1.0 - 2.0 * rate) ** len(support)) / 2.0, abs=1e-12
    )
    assert float(blind[0].probability) > rate
    assert not any(
        reference.round_index is None
        for detector in memory.detectors.detectors
        for reference in detector.parity
    )


@pytest.mark.parametrize(
    "distance, rounds, detectors, with_y, without_y", _MODEL_PROFILE
)
def test_the_noise_declaration_decides_whether_the_model_is_graphlike(
    distance: int,
    rounds: int,
    detectors: int,
    with_y: dict[int, int],
    without_y: dict[int, int],
) -> None:
    """The same code is graphlike or not depending only on the declared faults.

    Nothing about the code changes between the two sides of this test: the same
    record, the same round count and the same fault rate produce both models. What
    changes is whether the noise declares a Y data fault, and the consequence is
    that the model gains weight-four mechanisms -- the Y fault's X half lighting
    the checks whose Z factor it meets and its Z half lighting the others, which
    sit on different round boundaries.

    Unlike the torus, this family has no round count at which the two sides agree,
    because a mixed check has no deterministic round to merge across; that the
    smallest round count is already split is the second measurement here. The
    graphlike side is handed to the matching decoder so that "graphlike" is a
    property of the decoder's acceptance rather than of a weight read off a
    histogram.
    """

    memory = build_memory_circuit(_patch(distance), rounds=rounds)
    declared = DetectorErrorModel.from_memory_circuit(
        memory, noise=_noise(y_fault=True)
    )
    quiet = DetectorErrorModel.from_memory_circuit(memory, noise=_noise(y_fault=False))

    assert declared.num_detectors == quiet.num_detectors == detectors
    assert declared.num_detectors == len(_patch(distance).checks) * (rounds - 1)
    assert declared.num_observables == quiet.num_observables == 1
    assert _weights(declared) == with_y
    assert _weights(quiet) == without_y
    assert max(_weights(quiet)) == 2
    assert max(_weights(declared)) == 4
    MinimumWeightMatchingDecoder.from_detector_error_model(quiet)
    with pytest.raises(CapabilityError, match="hyperedge"):
        MinimumWeightMatchingDecoder.from_detector_error_model(declared)


@pytest.mark.parametrize(
    "distance, rounds, detectors, with_y, without_y", _MODEL_PROFILE
)
def test_the_circuit_sampler_and_the_model_agree_on_the_observable(
    distance: int,
    rounds: int,
    detectors: int,
    with_y: dict[int, int],
    without_y: dict[int, int],
) -> None:
    """The model's marginals match what the circuit actually does, per index.

    Every row of the profile is compared, so the merged mechanisms of the last
    round's boundary are covered rather than the single faults alone. The
    comparison uses the model whose noise declares no Y fault, so both sides are
    graphlike and the row is about the sampler rather than about a decoder.

    The observable is compared as well as the detectors, and it is the observable
    that carries the blind mechanism: a model that put the readout's probability in
    the wrong column would agree on every detector and fail here.
    """

    memory = build_memory_circuit(_patch(distance), rounds=rounds)
    noise = _noise(y_fault=False)
    model = DetectorErrorModel.from_memory_circuit(memory, noise=noise)
    sample = sample_memory_circuit(
        memory, noise=noise, shots=_SAMPLED_SHOTS, seed=_SEED
    )

    assert sample.detectors.shape == (shots := _SAMPLED_SHOTS, detectors)
    assert sample.observables.shape == (shots, 1)
    for label, counts, rates in (
        ("detector", sample.detectors.float().mean(dim=0), model.detector_rates()),
        (
            "observable",
            sample.observables.float().mean(dim=0),
            model.observable_rates(),
        ),
    ):
        assert rates.shape == counts.shape
        for index in range(counts.numel()):
            observed = float(counts[index])
            modelled = float(rates[index])
            assert abs(observed - modelled) <= _tolerance(modelled, shots=shots), (
                f"{label} {index} sampled {observed}, which is outside "
                f"{_tolerance(modelled, shots=shots)} of the modelled {modelled}"
            )


@pytest.mark.parametrize("distance", (3, 5))
def test_every_single_mechanism_is_explained_and_some_are_misread(
    distance: int,
) -> None:
    """Every mechanism's syndrome is reproduced, and not all of them are explained.

    Both halves are asserted against each other, because either one alone is
    satisfied by a decoder that is useless in the other direction: one that
    returned the whole edge set would reproduce nothing, and one that returned the
    empty correction would reproduce nothing else. The counts are recorded as the
    code's own mechanism structure rather than as a decoder's score.

    The misreads are then located rather than counted. A syndrome shared by two
    mechanisms that disagree about the observable accounts for some of them,
    because the matcher can only undo one of the pair, and the assertion is that
    the remaining misreads are not all of that kind -- one mechanism with its own
    syndrome is misread too, which is the readout of the row above seen from the
    decoder's side.
    """

    memory = build_memory_circuit(_patch(distance), rounds=2)
    model = DetectorErrorModel.from_memory_circuit(
        memory, noise=_noise(y_fault=False, rate=0.02)
    )
    decoder = MinimumWeightMatchingDecoder.from_detector_error_model(model)
    by_syndrome: dict[tuple[int, ...], list[int]] = {}
    for index, error in enumerate(model.errors):
        by_syndrome.setdefault(
            tuple(sorted(int(node) for node in error.detectors)), []
        ).append(index)

    ambiguous = ambiguous_wrong = unambiguous = unambiguous_wrong = 0
    for syndrome, members in by_syndrome.items():
        shared = (
            len(
                {
                    frozenset(int(i) for i in model.errors[m].observables)
                    for m in members
                }
            )
            > 1
        )
        result = decoder.decode(syndrome)
        assert _odd_degree(decoder.graph, result.error_edges) == set(syndrome)
        for member in members:
            agrees = {int(i) for i in model.errors[member].observables} == set(
                result.observables
            )
            if shared:
                ambiguous += 1
                ambiguous_wrong += not agrees
            else:
                unambiguous += 1
                unambiguous_wrong += not agrees

    assert len(by_syndrome) < len(model.errors)
    assert ambiguous and unambiguous
    assert 0 < ambiguous_wrong < ambiguous
    assert 0 < unambiguous_wrong < unambiguous
    assert unambiguous_wrong + ambiguous_wrong < len(model.errors)


@pytest.mark.parametrize("distance", (3, 5))
def test_the_decoder_always_returns_a_correction_that_reproduces_the_syndrome(
    distance: int,
) -> None:
    """Every sampled syndrome is explained exactly.

    The syndromes are the model's own draws at a large shot budget, so the row
    covers the merged mechanisms and every detector count the noise reaches rather
    than the single faults alone. The assertion is reproduction rather than a rate:
    the number of syndromes a decoder explains is not a performance figure, but a
    correction that does not reproduce its syndrome is not a correction.
    """

    memory = build_memory_circuit(_patch(distance), rounds=2)
    model = DetectorErrorModel.from_memory_circuit(memory, noise=_noise(y_fault=False))
    sample = model.dem_sampling(shots=_DECODING_SHOTS, seed=_SEED)
    decoder = MinimumWeightMatchingDecoder.from_detector_error_model(model)

    for row in range(_DECODING_SHOTS):
        events = tuple(
            int(index) for index in sample.detectors[row].nonzero().flatten().tolist()
        )
        result = decoder.decode(events)
        assert _odd_degree(decoder.graph, result.error_edges) == set(events)
        assert all(0 <= index < model.num_observables for index in result.observables)

    assert model.num_observables == 1
    assert model.num_detectors == len(_patch(distance).checks)


@pytest.mark.parametrize("distance", _CONTRAST_DISTANCES)
def test_matching_barely_moves_the_logical_rate_on_this_patch(distance: int) -> None:
    """The decoded rate stays near the raw rate here and falls far below it there.

    The rotated patch is run through the identical harness at the same distance,
    the same rate, the same shot budget and the same seed, and its checks are pure,
    so the difference between the two rows is the mixedness of the checks and
    nothing else. The ZXXZ patch's decoded rate is required to stay above
    ``raw / 1.5`` and the rotated patch's below ``raw / 4``: the first is the
    limitation stated as an inequality, and the second is the control that says the
    harness can produce a suppressed rate at all.

    Larger distances are not compared against each other. A decoded rate that rose
    with the distance is consistent with an unprotected readout and would be a
    scaling claim if it were asserted, so it is recorded in the record's own
    documentation rather than as an assertion here.

    The decoded rate must also be below the raw rate, so a decoder that returned
    the empty correction for every syndrome -- one that "explains" nothing -- is
    not what produces the ceiling.
    """

    raw, decoded = _decoded_rate(
        _patch(distance), distance=distance, rounds=2, shots=_CONTRAST_SHOTS
    )
    control_raw, control_decoded = _decoded_rate(
        _rotated(distance), distance=distance, rounds=2, shots=_CONTRAST_SHOTS
    )

    assert 0.0 < decoded < raw
    assert decoded > raw / _ZXXZ_CEILING
    assert 0.0 < control_decoded < control_raw / _ROTATED_FLOOR
    assert decoded > control_decoded * _ROTATED_FLOOR
