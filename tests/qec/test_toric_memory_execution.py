"""End-to-end coverage for the square-lattice toric code as a family record.

``test_toric_code.py`` pins the algebra of the torus this package derives from a
linear size. This file takes that record through the rest of the path a code has
to walk: memory circuit, detector error model, sampling, and decoding.

The family is worth walking because it is the first one here that splits the
decoder call on the **noise declaration** rather than on the code. A torus has no
boundary, so its checks are all weight four and its single-round model is
graphlike whatever the noise says. What makes it not graphlike is a **Y data
fault seen across a round boundary**: the X part of such a fault lights X-type
checks, which have interior detectors only, and the Z part lights Z-type checks,
which have detectors at every round boundary. At one round the two parts land on
the same detectors and the mechanism has weight two. From two rounds on they land
on different ones and the mechanism has weight four, so the matching decoder
refuses a model it accepted one round earlier. That is measured here rather than
asserted, and it is also the family's practical warning: adding a Y fault to a
noise model can move a code out of the matching decoder's reach without changing
the code.

What this file proves
---------------------

1. ``test_the_memory_experiment_declares_one_observable_per_logical_qubit``: the
   detector count the memory circuit derives is the one the record's check classes
   and the round count imply, and the layout declares **two** observables whose
   supports are the record's two declared Z-type logical operators. Every other
   family here declares one, so the count is derived from the code rather than
   compared against a table.
2. ``test_the_noiseless_torus_flips_no_detector_or_observable``: the smallest
   torus, executed as a real circuit at one and three rounds, flips nothing when no
   noise is configured. At three rounds the X-type checks have their interior
   detectors and an X-type ancilla's raw round-zero bit is random, so a layout that
   read raw bits would fail the second row and pass the first.
3. ``test_the_noise_declaration_decides_whether_the_model_is_graphlike``: with the
   same code, round count and fault rate, the model is graphlike when the noise
   declares no Y fault and carries weight-four mechanisms when it does -- and only
   from two rounds on, because at one round the Y fault's two halves share their
   detectors. Both sides are asserted at every row, so the row is a comparison
   rather than a refusal on its own, and the accepted side is handed to the
   matching decoder so that "graphlike" is not read off a weight alone.
4. ``test_the_circuit_sampler_and_the_model_agree_on_both_observables``: the
   detector and observable marginals the model states match what the stabilizer
   sampler observes, per index, on a layout with two observables rather than one.
   The two sides are independent -- the model is derived from the matrices and the
   circuit's layouts, the sampler lowers the same circuit, places the noise at the
   instructions it belongs to, and executes -- so this is what pins the model's
   mechanism signatures to the program a shot actually runs.
5. ``test_a_single_fault_is_detected_and_corrected_above_distance_two``: every
   mechanism of the one-round model is decoded, the correction reproduces the
   syndrome it was given, and the observable flips it predicts are the ones the
   mechanism causes. The distance-two torus is measured alongside and does **not**
   correct all of them, so the row is about the distance rather than about the
   decoder.
6. ``test_the_decoder_always_returns_a_correction_that_reproduces_the_syndrome``:
   at three linear sizes, every syndrome the model itself samples is decodable and
   every correction reproduces its syndrome exactly. A decoder that returned a
   plausible-looking edge set without reproducing the syndrome would fail here,
   which is the gap between a decoder and a heuristic.
7. ``test_decoding_reduces_the_logical_rate_at_low_noise``: at a noise rate low
   enough that the distance-three torus corrects every single fault, the decoded
   logical rate sits well below the model's own raw observable rate, per
   observable index. This is what makes the correction usable rather than merely
   valid.

What this file does not prove
-----------------------------

It makes no threshold, suppression or scaling claim. Linear sizes three, four and
five are three tori, not a family of growing capacity, and the file compares each
torus's decoded rate against its own raw rate rather than one size against
another: at these shot budgets a small difference between two sizes is not
separable from sampling error, so it is not asserted. Nothing here is a statement
about a toric-code decoder's real-world performance, and the weight-four
mechanisms that decide the matching refusal are fed to nothing but that refusal --
the belief-propagation decoder is not run against this family at all.

Execution is limited to the linear size two torus. The trajectory simulator draws
each shot from the full output distribution, so the size-three torus, which is
thirty-six qubits, is 2**36 amplitudes per shot and does not fit; the size-three
and larger rows of this file run through the model and the stabilizer sampler,
both of which are polynomial in the lattice size.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from functools import lru_cache

import pytest

from flagquantum.compiler._hybrid import INDEX, capture_source, lower_dynamic_program
from flagquantum.errors import CapabilityError
from flagquantum.qec import toric_code
from flagquantum.qec.circuit import MeasurementRef, MemoryCircuit, build_memory_circuit
from flagquantum.qec.codes import CssCode
from flagquantum.qec.decoding_graph import DecodingGraph, DecodingGraphEdge
from flagquantum.qec.dem import DetectorErrorModel
from flagquantum.qec.matching import MinimumWeightMatchingDecoder
from flagquantum.qec.noise import PhenomenologicalNoise
from flagquantum.qec.sampling import sample_memory_circuit
from flagquantum.runtime.dynamic.hybrid_session import execute_hybrid_dynamic_session

pytestmark = pytest.mark.integration

# Seed for every draw in this file, so the sampled rows are reproducible.
_SEED = 0

# The fault rate the profile and the sampler comparison run at. The decoded-rate
# row uses a lower rate, where the distance-three torus corrects every single
# fault and the comparison is therefore about correction rather than the limit.
_PROBABILITY = 0.05
_LOW_NOISE = 0.01

# A sampled rate is accepted when it sits within this many binomial standard
# errors of the modelled rate. Four covers a per-index miss probability of about
# 6e-5, and the comparison is per index, so one wrong mechanism is located by the
# failure message rather than hidden in a summary statistic.
_RATE_TOLERANCE_SIGMAS = 4.0

# Shot budgets. The sampler lowers and executes once per call on the stabilizer
# engine, and the decode loop is one exact pairing per shot.
_SAMPLED_SHOTS = 4000
_DECODING_SHOTS = 4000

# The noise at which a decoded rate is compared against the model's raw rate. The
# measured values at this rate and 4000 shots are 0.0294 raw against 0.0010
# decoded at size three, 0.0388 against 0.0013 at size four and 0.0480 against
# 0.0000 and 0.0008 at size five, so the margin asserted below is a factor of
# four, which every index of every row clears by a factor of seven or more. The
# assertion is an inequality rather than those numbers because the numbers are a
# sample at one seed and the inequality is the claim.
_DECODING_MARGIN = 4.0

# The model profile: linear size, round count, detector count, the mechanism
# weight histogram of the model whose noise declares a Y fault, and the same
# histogram for the model whose noise does not. The detector count is affine in
# the round count and is asserted against the code's own check counts as well.
# The histograms are recorded measurements rather than formulas: a weight-one
# mechanism appears only once a fault at the last round boundary has somewhere to
# merge, and the weight-four mechanisms appear only once the Y fault's two halves
# land on different round boundaries.
_MODEL_PROFILE = (
    (3, 1, 18, {2: 27}, {2: 27}),
    (3, 2, 36, {1: 9, 2: 72, 4: 18}, {1: 9, 2: 72}),
    (3, 3, 54, {1: 18, 2: 126, 4: 36}, {1: 18, 2: 126}),
    (5, 1, 50, {2: 75}, {2: 75}),
    (5, 2, 100, {1: 25, 2: 200, 4: 50}, {1: 25, 2: 200}),
)


@lru_cache(maxsize=None)
def _torus(linear_size: int) -> CssCode:
    """Return the torus once per size, because the distance search is the cost."""

    return toric_code(linear_size)


def _noise(*, y_fault: bool) -> PhenomenologicalNoise:
    """Return the depolarizing noise the profile rows are recorded under.

    ``both_flip`` is the Y data fault. It is the only field that separates the two
    sides of the profile, so the helper makes that explicit rather than hiding it
    in two nearly identical literals.
    """

    return PhenomenologicalNoise(
        data_flip=_PROBABILITY,
        phase_flip=_PROBABILITY,
        both_flip=_PROBABILITY if y_fault else 0.0,
        measurement_flip=_PROBABILITY,
    )


def _flips(memory: MemoryCircuit, source: str, *, shots: int = 1) -> list[tuple]:
    """Execute ``source`` and read its detection events and observable flips off.

    This is a reader, not a second derivation of the model: it lowers the given
    source, runs it, and takes the parity of the bits each declared parity
    references. A syndrome bit is read at ``round_index * len(checks) +
    position``, which is the stride the lowerer emits, so the position is the
    check's place in the code's ``checks`` tuple and never its declared ``index``.
    It is written here rather than shared with the model so that the two do not
    read one layout implementation.
    """

    lowered = lower_dynamic_program(
        capture_source(source, (INDEX,)),
        (memory.rounds,),
        max_dynamic_measurements=memory.rounds * len(memory.code.checks),
    )
    execution = execute_hybrid_dynamic_session(
        lowered.circuit, shots=shots, seed=_SEED, strategy="trajectory"
    )
    classical: list[list[int]] = execution.classical_bits.tolist()
    samples: list[list[int]] = execution.samples.tolist()
    checks = len(memory.code.checks)
    positions = {
        check.ancilla_wire: position
        for position, check in enumerate(memory.code.checks)
    }

    def bit(
        classical_row: list[int], sample_row: list[int], reference: MeasurementRef
    ) -> int:
        if reference.round_index is None:
            return int(sample_row[reference.wire])
        return int(
            classical_row[reference.round_index * checks + positions[reference.wire]]
        )

    def parity(classical_row: list[int], sample_row: list[int], references) -> int:
        return (
            sum(bit(classical_row, sample_row, reference) for reference in references)
            % 2
        )

    return [
        (
            tuple(
                detector.index
                for detector in memory.detectors.detectors
                if parity(classical_row, sample_row, detector.parity)
            ),
            tuple(
                observable.index
                for observable in memory.observables.observables
                if parity(classical_row, sample_row, observable.measurement_parity)
            ),
        )
        for classical_row, sample_row in zip(classical, samples, strict=True)
    ]


def _weights(model: DetectorErrorModel) -> dict[int, int]:
    """Return the mechanism weight histogram of one model."""

    return dict(sorted(Counter(len(error.detectors) for error in model.errors).items()))


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


def _syndrome_of(
    model: DetectorErrorModel, mechanisms: tuple[int, ...]
) -> tuple[int, ...]:
    """Return the detection events a selected mechanism set flips, ascending."""

    fired: set[int] = set()
    for index in mechanisms:
        for detector in model.errors[index].detectors:
            fired ^= {int(detector)}
    return tuple(sorted(fired))


@pytest.mark.parametrize("linear_size, rounds", ((3, 1), (3, 3), (5, 2)))
def test_the_memory_experiment_declares_one_observable_per_logical_qubit(
    linear_size: int, rounds: int
) -> None:
    """The layout's counts are the ones the record's checks and rounds imply.

    The formula is the protocol's, not this file's: every Z-type check has a
    detector at each round boundary and every X-type check has one in each interior
    round, because a Z-type ancilla's round-zero bit is deterministic and an X-type
    ancilla's is not. The torus splits its checks evenly, so the two terms are one
    number here, and the assertion says that rather than evaluating an expression
    twice.

    The observable count is the family's own: the torus declares two Z-type
    logical operators, so the default Z-basis memory experiment reads out two, and
    each observable's support is the declared operator rather than a weight-``d``
    string of the right length.
    """

    code = _torus(linear_size)
    memory = build_memory_circuit(code, rounds=rounds)
    z_checks = sum(1 for check in code.checks if check.stabilizer.z_wires)
    x_checks = sum(1 for check in code.checks if check.stabilizer.x_wires)

    assert z_checks == x_checks == len(code.checks) // 2 == linear_size * linear_size
    assert len(memory.detectors) == z_checks * (rounds + 1) + x_checks * (rounds - 1)

    declared = [
        observable for observable in code.logical_observables if observable.z_wires
    ]
    assert len(memory.observables) == len(declared) == 2
    assert [
        {
            reference.wire
            for reference in observable.measurement_parity
            if reference.round_index is None
        }
        for observable in memory.observables.observables
    ] == [set(observable.support) for observable in declared]


@pytest.mark.parametrize("rounds", (1, 3))
def test_the_noiseless_torus_flips_no_detector_or_observable(rounds: int) -> None:
    """The executed program is the experiment the layouts describe.

    Only the linear size two torus is executed here, and the reason is capacity
    rather than convenience: the trajectory simulator draws each shot from the full
    output distribution, so a thirty-six-qubit torus is 2**36 amplitudes per shot.
    The larger rows below run through the model and the stabilizer sampler, neither
    of which materialises a distribution.
    """

    memory = build_memory_circuit(_torus(2), rounds=rounds)
    read = _flips(memory, memory.source, shots=8)

    assert len(read) == 8
    assert {flip[0] for flip in read} == {()}
    assert {flip[1] for flip in read} == {()}


@pytest.mark.parametrize(
    "linear_size, rounds, detectors, with_y, without_y", _MODEL_PROFILE
)
def test_the_noise_declaration_decides_whether_the_model_is_graphlike(
    linear_size: int,
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
    X-type checks and its Z half lighting Z-type checks, which sit on different
    round boundaries.

    At one round there is no such boundary between the halves, so the two models
    agree, and that row is why the test is a comparison: a row that only asserted
    the weight-four mechanisms would not show that the code is graphlike at all,
    and one that only asserted the graphlike side would not show what breaks it.
    The graphlike side is handed to the matching decoder so that "graphlike" is a
    property of the decoder's acceptance rather than of a weight read off a
    histogram.
    """

    memory = build_memory_circuit(_torus(linear_size), rounds=rounds)
    declared = DetectorErrorModel.from_memory_circuit(
        memory, noise=_noise(y_fault=True)
    )
    quiet = DetectorErrorModel.from_memory_circuit(memory, noise=_noise(y_fault=False))

    assert declared.num_detectors == quiet.num_detectors == detectors
    assert declared.num_observables == quiet.num_observables == 2
    assert _weights(declared) == with_y
    assert _weights(quiet) == without_y
    assert max(_weights(quiet)) == 2
    MinimumWeightMatchingDecoder.from_detector_error_model(quiet)

    if rounds == 1:
        assert max(_weights(declared)) == 2
        MinimumWeightMatchingDecoder.from_detector_error_model(declared)
    else:
        assert max(_weights(declared)) == 4
        with pytest.raises(CapabilityError, match="hyperedge"):
            MinimumWeightMatchingDecoder.from_detector_error_model(declared)


@pytest.mark.parametrize(
    "linear_size, rounds, detectors, with_y, without_y", _MODEL_PROFILE
)
def test_the_circuit_sampler_and_the_model_agree_on_both_observables(
    linear_size: int,
    rounds: int,
    detectors: int,
    with_y: dict[int, int],
    without_y: dict[int, int],
) -> None:
    """The model's marginals match what the circuit actually does, per index.

    Every row of the profile is compared, and the two-round rows are the
    load-bearing ones: a mechanism reached by one round's fault and not the next
    is invisible at one round, and exactly those merged mechanisms are what the
    recorded histogram is about. Both observables are compared index by index, so
    a mechanism attributed to the wrong observable is located by the failure
    message rather than hidden in a two-observable summary.

    The comparison uses the model whose noise declares no Y fault, so both sides
    are graphlike and the row is about the sampler rather than about a decoder.
    """

    memory = build_memory_circuit(_torus(linear_size), rounds=rounds)
    noise = _noise(y_fault=False)
    model = DetectorErrorModel.from_memory_circuit(memory, noise=noise)
    sample = sample_memory_circuit(
        memory, noise=noise, shots=_SAMPLED_SHOTS, seed=_SEED
    )

    assert sample.detectors.shape == (shots := _SAMPLED_SHOTS, detectors)
    assert sample.observables.shape == (shots, 2)
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


@pytest.mark.parametrize("linear_size", (2, 3, 4))
def test_a_single_fault_is_detected_and_corrected_above_distance_two(
    linear_size: int,
) -> None:
    """Every one-round mechanism is detected, explained, and undone above distance two.

    The mechanisms are the model's own, so the syndrome each one is handed is the
    one it declares rather than one this file computed. Both halves of the
    correction are asserted: it reproduces the syndrome it was given, and the
    observable flips it predicts are the ones the mechanism causes. A decoder that
    returned an edge set spanning the syndrome without explaining it would pass the
    first assertion and fail nothing; a decoder that explained the syndrome but
    misread an observable would fail the second.

    The distance-two torus is the control, and the shape of its failure is the
    evidence: twelve mechanisms land on eight distinct syndromes, so two of them
    share a syndrome and the matcher can only undo one of the pair. Above distance
    two the mechanisms and the syndromes are in bijection, and the assertion is
    that identity rather than a corrected count. That is what keeps this row a
    statement about the distance rather than a decoder's score.
    """

    model = DetectorErrorModel.from_memory_circuit(
        build_memory_circuit(_torus(linear_size), rounds=1),
        noise=_noise(y_fault=False),
    )
    decoder = MinimumWeightMatchingDecoder.from_detector_error_model(model)
    syndromes = Counter(
        tuple(sorted(int(node) for node in error.detectors)) for error in model.errors
    )
    corrected = 0

    for error in model.errors:
        detection_events = tuple(sorted(int(node) for node in error.detectors))
        result = decoder.decode(detection_events)
        assert _odd_degree(decoder.graph, result.error_edges) == set(detection_events)
        assert all(0 <= index < model.num_observables for index in result.observables)
        if set(result.observables) == {int(item) for item in error.observables}:
            corrected += 1

    if linear_size == 2:
        assert len(syndromes) < len(model.errors)
        assert max(syndromes.values()) > 1
        assert 0 < corrected < len(model.errors)
    else:
        assert set(syndromes.values()) == {1}
        assert corrected == len(model.errors)


@pytest.mark.parametrize("linear_size", (3, 4, 5))
def test_the_decoder_always_returns_a_correction_that_reproduces_the_syndrome(
    linear_size: int,
) -> None:
    """At three sizes every sampled syndrome is explained exactly.

    The syndromes are the model's own draws at a large shot budget, so the row
    covers the merged mechanisms and every detector count the noise reaches rather
    than the single faults alone. The assertion is reproduction rather than a rate:
    the number of syndromes a decoder explains is not a performance figure, but a
    correction that does not reproduce its syndrome is not a correction.

    Both observables are range-checked on every row, because a correction that
    named an observable outside the layout would still reproduce the syndrome and
    would then be read against a column that does not exist.
    """

    model = DetectorErrorModel.from_memory_circuit(
        build_memory_circuit(_torus(linear_size), rounds=1),
        noise=_noise(y_fault=False),
    )
    sample = model.dem_sampling(shots=_DECODING_SHOTS, seed=_SEED)
    decoder = MinimumWeightMatchingDecoder.from_detector_error_model(model)

    for row in range(_DECODING_SHOTS):
        detection_events = tuple(
            int(index) for index in sample.detectors[row].nonzero().flatten().tolist()
        )
        result = decoder.decode(detection_events)
        assert _odd_degree(decoder.graph, result.error_edges) == set(detection_events)
        assert all(0 <= index < model.num_observables for index in result.observables)

    assert model.num_observables == 2
    assert model.num_detectors == 2 * linear_size * linear_size


@pytest.mark.parametrize("linear_size", (3, 4, 5))
def test_decoding_reduces_the_logical_rate_at_low_noise(linear_size: int) -> None:
    """Correcting the faults a code of this size corrects lowers the rate.

    The reference is the model's own raw observable rate, which is exact rather
    than sampled, and the measured side is the fraction of sampled syndromes whose
    correction predicts a different set of observable flips from the one the draw
    caused. The margin is a factor of four, which the measured rows clear by more
    than a factor of seven, so a decoder that returned the empty correction for
    every syndrome -- one that "explains" nothing -- would fail here.

    Every observable index is compared, and the torus declares two, so a decoder
    that collapsed the pair into one label would fail on the second index rather
    than pass on the first.
    """

    model = DetectorErrorModel.from_memory_circuit(
        build_memory_circuit(_torus(linear_size), rounds=1),
        noise=PhenomenologicalNoise(data_flip=_LOW_NOISE, measurement_flip=_LOW_NOISE),
    )
    sample = model.dem_sampling(shots=_DECODING_SHOTS, seed=_SEED)
    decoder = MinimumWeightMatchingDecoder.from_detector_error_model(model)
    wrong = [0] * model.num_observables

    for row in range(_DECODING_SHOTS):
        detection_events = tuple(
            int(index) for index in sample.detectors[row].nonzero().flatten().tolist()
        )
        caused = {
            int(index) for index in sample.observables[row].nonzero().flatten().tolist()
        }
        result = decoder.decode(detection_events)
        assert _odd_degree(decoder.graph, result.error_edges) == set(detection_events)
        for index in range(model.num_observables):
            if (index in caused) != (index in set(result.observables)):
                wrong[index] += 1

    raw = [float(rate) for rate in model.observable_rates()]
    decoded = [count / _DECODING_SHOTS for count in wrong]

    assert model.num_observables == 2
    assert all(0.0 < rate < 1.0 for rate in raw)
    for index, (before, after) in enumerate(zip(raw, decoded, strict=True)):
        assert (
            after < before / _DECODING_MARGIN
        ), f"observable {index} decoded to {after} against a raw rate of {before}"
