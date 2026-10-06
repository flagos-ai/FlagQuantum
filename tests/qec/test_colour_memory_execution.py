"""End-to-end coverage for the triangular colour code as a family record.

``test_colour_code.py`` pins the algebra of the patch this package derives from a
distance. This file takes that record through the rest of the path a code has to
walk: memory circuit, detector error model, sampling, and decoding. The family is
worth walking because it is the first one here whose checks are not all the same
weight, and the consequence is visible at the decoder rather than in the
matrices: a single data fault on a six-vertex face lights three detectors, the
model is not graphlike, and the matching decoder refuses it. What opens the
family is the belief-propagation decoder, and this file is where that is
measured rather than asserted.

What this file proves
---------------------

1. ``test_the_memory_experiment_follows_the_two_check_classes``: the detector and
   observable counts the memory circuit derives are the ones the record's checks
   and the round count imply, and the record's checks split evenly into the two
   families, which is what self-duality means at the protocol level. The count is
   derived from the code here rather than compared against a table.
2. ``test_the_noiseless_patch_flips_no_detector_or_observable``: the distance-three
   patch, executed as a real circuit at one and three rounds, flips nothing when no
   noise is configured. At three rounds the X-type checks have their interior
   detectors, and an X-type ancilla's raw round-zero bit is random, so a layout
   that read raw bits would fail the second row and pass the first.
3. ``test_the_model_is_not_graphlike_at_the_smallest_distance``: the model's
   mechanism-weight histogram at the recorded profile, its maximum weight of three
   at every distance and round count, and the matching decoder's refusal. The
   refusal is contrasted against the repetition and rotated surface codes, whose
   models the same decoder accepts at the same round count, so the refusal is a
   property of this family rather than of the decoder.
4. ``test_the_circuit_sampler_and_the_model_agree``: the detector and observable
   marginals the model states match what the stabilizer sampler observes, per
   index, at every row of the profile including the two-round rows. The two sides
   are independent -- the model is derived from the matrices and the circuit's
   layouts, the sampler lowers the same circuit, places the noise at the
   instructions it belongs to, and executes -- so this is what pins the model's
   mechanism signatures to the program a shot actually runs.
5. ``test_a_single_data_fault_is_detected_and_corrected``: every one of the
   distance-three patch's seven data wires is faulted in the executed circuit, the
   syndrome that comes back is decoded by belief propagation with ordered
   statistics, the correction reproduces the syndrome it was given, and the
   observable flip it predicts equals the one the fault caused. This is the
   strongest statement available for the code: the path ends in a usable
   correction for a family the matching decoder declines.
6. ``test_the_decoder_always_returns_a_correction_that_reproduces_the_syndrome``:
   at distance five, every syndrome the model itself samples is decodable and every
   correction reproduces its syndrome exactly. A decoder that returned a
   plausible-looking mechanism set without reproducing the syndrome would fail
   here, which is the gap between a decoder and a heuristic.
7. ``test_decoding_reduces_the_logical_rate_at_low_noise``: at a noise rate low
   enough that the distance-three patch corrects every single fault, the decoded
   logical rate sits well below the model's own raw observable rate at the same
   distance. This is what makes the correction usable rather than merely valid.

What this file does not prove
-----------------------------

It makes no threshold, suppression or scaling claim. Distance three and distance
five are two patches, not a family of growing capacity, and the file compares each
patch's decoded rate against its own raw rate rather than one distance against the
other: at these shot budgets a two-fold difference between the two distances is
not separable from sampling error, so it is not asserted. The decoder is the
package's own belief propagation with an order-zero ordered-statistics pass, and
nothing here is a statement about a colour-code decoder's real-world performance.

Execution is limited to the distance-three patch. The trajectory simulator draws
each shot from the full output distribution, so a distance-five colour patch, which
is thirty-seven qubits, is 2**37 amplitudes per shot and does not fit; the
distance-five rows of this file therefore run through the model and the stabilizer
sampler, both of which are polynomial in the patch size.
"""

from __future__ import annotations

import math

import pytest

from flagquantum.compiler._hybrid import INDEX, capture_source, lower_dynamic_program
from flagquantum.errors import CapabilityError
from flagquantum.qec import triangular_colour_code
from flagquantum.qec.bposd import BeliefPropagationOsdDecoder
from flagquantum.qec.circuit import MeasurementRef, MemoryCircuit, build_memory_circuit
from flagquantum.qec.codes import RepetitionCode
from flagquantum.qec.dem import DetectorErrorModel
from flagquantum.qec.dem_construction import _inject_data_flip
from flagquantum.qec.matching import MinimumWeightMatchingDecoder
from flagquantum.qec.noise import PhenomenologicalNoise
from flagquantum.qec.sampling import sample_memory_circuit
from flagquantum.qec.surface import RotatedSurfaceCode
from flagquantum.runtime.dynamic.hybrid_session import execute_hybrid_dynamic_session

pytestmark = pytest.mark.integration

# Seed for every draw in this file, so the sampled rows are reproducible.
_SEED = 0

# The noise the profile and the sampler comparison run at, which is a depolarizing
# data fault and a measurement fault. The decoded-rate row uses a lower rate, where
# the distance-three patch corrects every single fault and the comparison is
# therefore about correction rather than about the code's limit.
_PROBABILITY = 0.05
_LOW_NOISE = 0.02

# A sampled rate is accepted when it sits within this many binomial standard
# errors of the modelled rate. Four covers a per-index miss probability of about
# 6e-5, and the comparison is per index, so one wrong mechanism is located by the
# failure message rather than hidden in a summary statistic.
_RATE_TOLERANCE_SIGMAS = 4.0

# Shot budgets. The sampler lowers and executes once per call on the stabilizer
# engine, and the decode loop is one ordered-statistics pass per shot.
_SAMPLED_SHOTS = 4000
_DECODING_SHOTS = 4000

# The noise at which a decoded rate is compared against the model's raw rate. The
# measured values at this rate and 4000 shots are 0.0545 raw against 0.0070 decoded
# at distance three and 0.0877 against 0.0035 at distance five, so the margin
# asserted below is a factor of four, which both rows clear by a wide factor. The
# assertion is an inequality rather than those four numbers because the numbers are
# a sample at one seed and the inequality is the claim.
_DECODING_MARGIN = 4.0

# The model profile: distance, round count, detector count, and the mechanism
# weight histogram at the noise above. The histogram is a recorded measurement
# rather than a formula -- the weight-one mechanisms merge as rounds are added
# only when their signatures coincide -- while the detector count is affine in the
# round count and is asserted against the code's own check counts as well. The
# maximum weight is the structural part: it is three at every row, which is what
# makes the model non-graphlike at the smallest patch there is.
_MODEL_PROFILE = (
    (3, 1, 6, {1: 3, 2: 6, 3: 1}),
    (3, 2, 12, {1: 9, 2: 15, 3: 3}),
    (3, 3, 18, {1: 15, 2: 27, 3: 5}),
    (5, 1, 18, {1: 3, 2: 18, 3: 7}),
    (5, 2, 36, {1: 15, 2: 45, 3: 21}),
    (5, 3, 54, {1: 27, 2: 81, 3: 35}),
)


def _depolarizing(probability: float) -> PhenomenologicalNoise:
    """Return the data and measurement noise the profile rows are recorded under."""

    return PhenomenologicalNoise(
        data_flip=probability,
        phase_flip=probability,
        measurement_flip=probability,
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


def _tolerance(predicted: float, *, shots: int) -> float:
    """Return the band a sampled rate may sit in, in binomial standard errors."""

    rate = max(predicted, 1.0 / shots)
    return _RATE_TOLERANCE_SIGMAS * math.sqrt(rate * (1.0 - rate) / shots)


def _syndrome_of(
    model: DetectorErrorModel, mechanisms: tuple[int, ...]
) -> tuple[int, ...]:
    """Return the detection events a selected mechanism set flips, ascending."""

    fired: set[int] = set()
    for index in mechanisms:
        for detector in model.errors[index].detectors:
            fired ^= {int(detector)}
    return tuple(sorted(fired))


@pytest.mark.parametrize("distance, rounds", ((3, 1), (3, 3), (5, 2)))
def test_the_memory_experiment_follows_the_two_check_classes(
    distance: int, rounds: int
) -> None:
    """The layout's counts are the ones the record's checks and rounds imply.

    The formula is the protocol's, not this file's: every Z-type check has a
    detector at each round boundary and every X-type check has one in each interior
    round, because a Z-type ancilla's round-zero bit is deterministic and an X-type
    ancilla's is not. The record's checks split evenly between the two classes, so
    the two terms are one number here, and the assertion says that rather than
    evaluating an expression twice.
    """

    code = triangular_colour_code(distance)
    memory = build_memory_circuit(code, rounds=rounds)
    z_checks = sum(1 for check in code.checks if check.stabilizer.z_wires)
    x_checks = sum(1 for check in code.checks if check.stabilizer.x_wires)

    assert z_checks == x_checks == len(code.checks) // 2
    assert len(memory.detectors) == z_checks * (rounds + 1) + x_checks * (rounds - 1)
    assert len(memory.observables) == 1
    # One logical qubit, so one observable, and it is the declared side of the
    # triangle: a readout support that named another weight-d string would still
    # pass a count assertion, which is why the support is compared.
    assert memory.observables.observables[0].measurement_parity
    assert {
        wire
        for reference in memory.observables.observables[0].measurement_parity
        for wire in (reference.wire,)
    } == set(code.logical_observables[0].support)


@pytest.mark.parametrize("rounds", (1, 3))
def test_the_noiseless_patch_flips_no_detector_or_observable(rounds: int) -> None:
    """The executed program is the experiment the layouts describe.

    Only the distance-three patch is executed here, and the reason is capacity
    rather than convenience: the trajectory simulator draws each shot from the full
    output distribution, so a thirty-seven-qubit patch is 2**37 amplitudes per
    shot. The distance-five rows below run through the model and the stabilizer
    sampler, neither of which materialises a distribution.
    """

    memory = build_memory_circuit(triangular_colour_code(3), rounds=rounds)
    read = _flips(memory, memory.source, shots=8)

    assert len(read) == 8
    assert {flip[0] for flip in read} == {()}
    assert {flip[1] for flip in read} == {()}


@pytest.mark.parametrize("distance, rounds, detectors, histogram", _MODEL_PROFILE)
def test_the_model_is_not_graphlike_at_the_smallest_distance(
    distance: int, rounds: int, detectors: int, histogram: dict[int, int]
) -> None:
    """A face carries one fault into three detectors, so matching refuses the model.

    The recorded histogram is asserted as an equality because it is the evidence
    this file's other rows are read against, and the maximum weight is asserted
    separately because it is the structural part: a face in the bulk of the patch
    touches six qubits, so a data fault inside it can light three detectors. The
    contrast is what keeps the refusal from reading as the decoder failing on
    everything.
    """

    code = triangular_colour_code(distance)
    memory = build_memory_circuit(code, rounds=rounds)
    model = DetectorErrorModel.from_memory_circuit(
        memory, noise=_depolarizing(_PROBABILITY)
    )
    counts: dict[int, int] = {}
    for error in model.errors:
        counts[len(error.detectors)] = counts.get(len(error.detectors), 0) + 1

    assert model.num_detectors == detectors
    assert model.num_observables == 1
    assert counts == histogram
    assert max(counts) == 3

    with pytest.raises(CapabilityError, match="hyperedge"):
        MinimumWeightMatchingDecoder.from_detector_error_model(model)

    for graphlike in (RepetitionCode(distance=3), RotatedSurfaceCode(distance=3)):
        accepted = DetectorErrorModel.from_memory_circuit(
            build_memory_circuit(graphlike, rounds=rounds),
            noise=_depolarizing(_PROBABILITY),
        )
        assert max(len(error.detectors) for error in accepted.errors) == 2
        MinimumWeightMatchingDecoder.from_detector_error_model(accepted)


@pytest.mark.parametrize("distance, rounds, detectors, histogram", _MODEL_PROFILE)
def test_the_circuit_sampler_and_the_model_agree(
    distance: int, rounds: int, detectors: int, histogram: dict[int, int]
) -> None:
    """The model's marginals match what the circuit actually does, per index.

    Every row of the profile is compared, and the two-round rows are the
    load-bearing ones: a mechanism reached by one round's fault and not the next
    is invisible at one round, and it is exactly the merged mechanism that the
    recorded histogram is about. The comparison is per index, so a mechanism
    attributed to the wrong detector or the wrong round is located by the failure
    message.
    """

    code = triangular_colour_code(distance)
    memory = build_memory_circuit(code, rounds=rounds)
    model = DetectorErrorModel.from_memory_circuit(
        memory, noise=_depolarizing(_PROBABILITY)
    )
    sample = sample_memory_circuit(
        memory, noise=_depolarizing(_PROBABILITY), shots=_SAMPLED_SHOTS, seed=_SEED
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


def test_a_single_data_fault_is_detected_and_corrected() -> None:
    """Every single fault of the distance-three patch is detected and undone.

    The fault is injected into the executed circuit, so the syndrome is the one a
    real run produces rather than one the model states. Both halves of the
    correction are asserted: it reproduces the syndrome it was handed, and the
    observable flip it predicts is the one the fault caused. A decoder that
    returned a mechanism set spanning the syndrome without explaining it would
    pass the first assertion and fail nothing; a decoder that explained the
    syndrome but misread the observable would fail the second.
    """

    code = triangular_colour_code(3)
    memory = build_memory_circuit(code, rounds=1)
    model = DetectorErrorModel.from_memory_circuit(
        memory, noise=_depolarizing(_PROBABILITY)
    )
    decoder = BeliefPropagationOsdDecoder(model)
    corrected: set[int] = set()

    for wire in code.data_wires:
        detection_events, observable_flips = _flips(
            memory, _inject_data_flip(memory, round_index=0, wire=wire)
        )[0]
        assert detection_events, f"a fault on wire {wire} light no detector"
        result = decoder.decode(detection_events)
        assert _syndrome_of(model, result.mechanisms) == detection_events
        if set(observable_flips) ^ set(result.observables) == set():
            corrected.add(wire)

    # A distance-three code corrects every single fault, so no fault may be left
    # uncorrected. Asserting the set rather than its size names the offending wire
    # in the failure.
    assert corrected == set(code.data_wires)


def test_the_decoder_always_returns_a_correction_that_reproduces_the_syndrome() -> None:
    """At distance five every sampled syndrome is explained exactly.

    The syndromes are the model's own draws at a large shot budget, so the row
    covers the merged mechanisms and the hyperedges that the matching decoder
    refuses rather than the single faults alone. The assertion is reproduction
    rather than a rate: the number of syndromes a decoder explains is not a
    performance figure, but a correction that does not reproduce its syndrome is
    not a correction.
    """

    memory = build_memory_circuit(triangular_colour_code(5), rounds=1)
    model = DetectorErrorModel.from_memory_circuit(
        memory, noise=_depolarizing(_PROBABILITY)
    )
    sample = model.dem_sampling(shots=_DECODING_SHOTS, seed=_SEED)
    decoder = BeliefPropagationOsdDecoder(model)
    explained = 0

    for row in range(_DECODING_SHOTS):
        detection_events = tuple(
            int(index) for index in sample.detectors[row].nonzero().flatten().tolist()
        )
        result = decoder.decode(detection_events)
        assert _syndrome_of(model, result.mechanisms) == detection_events
        assert all(0 <= index < model.num_observables for index in result.observables)
        explained += 1

    assert explained == _DECODING_SHOTS
    assert model.num_detectors > 0


@pytest.mark.parametrize("distance", (3, 5))
def test_decoding_reduces_the_logical_rate_at_low_noise(distance: int) -> None:
    """Correcting the faults that a code of this distance corrects lowers the rate.

    The reference is the model's own raw observable rate, which is exact rather
    than sampled, and the measured side is the fraction of sampled syndromes whose
    correction predicts a different observable flip from the one the draw caused.
    The margin is a factor of four, which the measured rows clear by eight and
    twenty-six respectively, so a decoder that returned the empty correction for
    every syndrome -- one that "explains" nothing -- would fail here.
    """

    memory = build_memory_circuit(triangular_colour_code(distance), rounds=1)
    noise = _depolarizing(_LOW_NOISE)
    model = DetectorErrorModel.from_memory_circuit(memory, noise=noise)
    sample = model.dem_sampling(shots=_DECODING_SHOTS, seed=_SEED)
    decoder = BeliefPropagationOsdDecoder(model)
    wrong = 0

    for row in range(_DECODING_SHOTS):
        detection_events = tuple(
            int(index) for index in sample.detectors[row].nonzero().flatten().tolist()
        )
        caused = {
            int(index) for index in sample.observables[row].nonzero().flatten().tolist()
        }
        result = decoder.decode(detection_events)
        assert _syndrome_of(model, result.mechanisms) == detection_events
        if caused ^ set(result.observables):
            wrong += 1

    raw = float(model.observable_rates()[0])
    decoded = wrong / _DECODING_SHOTS

    assert 0.0 < raw < 1.0
    assert decoded < raw / _DECODING_MARGIN
