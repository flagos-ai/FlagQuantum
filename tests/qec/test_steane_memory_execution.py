"""End-to-end coverage for the Steane code as one vertical slice.

The unit file ``test_steane_code.py`` pins the code's algebra. This file takes
the same record through the rest of the path a code has to walk: memory-circuit
construction, detector and observable layouts, hybrid lowering, circuit
execution, detector-error-model construction, and detector-error-model sampling.

What this file proves
---------------------

1. ``test_the_noiseless_memory_circuit_flips_no_detector_or_observable``: the
   noiseless memory circuit's detectors are deterministic. An X-type check
   ancilla is prepared in ``|+>``, so its raw round-zero bit is random and
   asserting the raw register would fail; what the layouts define is the
   detector parity, and every detector of that parity is zero on every shot.
2. ``test_a_single_data_error_is_detected_and_its_syndrome_identifies_the_wire``:
   all seven single-qubit X errors are detected, their seven syndromes are
   pairwise distinct, and their weight spectrum is ``{1: 3, 2: 3, 3: 1}``. That
   is the seven non-zero three-bit vectors, which is exactly what a perfect
   distance-three code produces: a distinct, non-empty syndrome for every
   correctable error. The observable flips on precisely the wires in the
   declared logical Z support and on no others, which is the statement that a
   single X error becomes a logical error exactly when it anticommutes with the
   observable.
3. ``test_the_detector_error_model_has_a_weight_three_mechanism``: the
   phenomenological model built from the memory circuit contains a mechanism
   that flips three detectors at every round count, and the matching decoder
   refuses it. The refusal is contrasted against the repetition and rotated
   surface codes, whose models the same decoder accepts at the same round count,
   so the refusal is a property of this code rather than a blanket failure of
   the decoder.
4. ``test_sampled_rates_agree_with_the_circuit_simulator``: every mechanism is
   drawn independently, the fired set is injected into one program, and the
   program is executed. The empirical detector and observable rates are compared
   against the model's own marginals. This is the step that makes the model
   usable for the Steane code at all: a hyperedge cannot reach the matching
   decoder, so the model's marginal rates are the strongest quantitative claim
   available, and they are checked against the circuit rather than against a
   second statement of the same arithmetic.

What this file does not prove
-----------------------------

It does not decode. The Steane code is quantum-perfect and therefore degenerate,
so a minimum-weight matching decoder is the wrong instrument for it, and this
file records that as a fail-closed boundary instead of adding a hyperedge-aware
decoder, which the parity contract defers. It also makes no threshold,
suppression, or scaling claim: the sampled comparison fixes one noise
probability and one round count, and a distance-three perfect code cannot
demonstrate that a logical error rate falls with distance. Reports must not cite
this file for either.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace

import pytest

from flagquantum.compiler._hybrid import INDEX, capture_source, lower_dynamic_program
from flagquantum.errors import CapabilityError
from flagquantum.qec.circuit import MeasurementRef, MemoryCircuit, build_memory_circuit
from flagquantum.qec.codes import (
    RepetitionCode,
    RotatedSurfaceCode,
    StabilizerCode,
    SteaneCode,
)
from flagquantum.qec.dem import DetectorErrorModel
from flagquantum.qec.dem_construction import (
    _inject_data_flip,
    _inject_measurement_flip,
    _Mechanism,
    _mechanisms,
)
from flagquantum.qec.matching import MinimumWeightMatchingDecoder
from flagquantum.qec.noise import PhenomenologicalNoise
from flagquantum.runtime.dynamic.hybrid_session import execute_hybrid_dynamic_session

pytestmark = pytest.mark.integration

# Seed for every draw in this file, so the sampled case is reproducible.
_SEED = 0

# The sampled rate is accepted when it sits within this many binomial standard
# errors of the modelled rate. Four covers a per-index miss probability of about
# 6e-5, and the observable's modelled rate is 0.1355 at the noise below, whose
# band is 0.0086 at this shot budget -- narrow enough that a model which dropped
# the weight-three mechanism's contribution, or added mechanism probabilities
# instead of merging them, moves the rate outside it.
_RATE_TOLERANCE_SIGMAS = 4.0

# Noise and shot budgets for the sampled comparison. The measured cost of one
# circuit execution at these settings is about 2 ms (lowering dominates), which
# keeps the file inside the ordinary ``integration`` lane.
_NOISE_PROBABILITY = 0.05
_SAMPLED_SHOTS = 300
_DEM_SAMPLING_SHOTS = 20_000

# Round counts the detector error model is built at, with the mechanism-weight
# histogram each one produces at the noise below. The histogram is a recorded
# measurement rather than a formula: it is not affine in the round count,
# because two mechanisms merge only when their signatures coincide, and at one
# round the three X-type round-zero syndrome measurements carry no detection
# event at all while at two rounds each of them coincides with the next round's.
# A fourth round therefore gives ``{1: 18, 2: 30, 3: 4}``, not the affine
# continuation of the three rows below. The count of weight-three mechanisms is
# the one part that is structural, and it is asserted separately.
_MODEL_ROUNDS = (
    (1, {1: 3, 2: 6, 3: 1}),
    (2, {1: 9, 2: 12, 3: 2}),
    (3, {1: 15, 2: 21, 3: 3}),
)


@dataclass(frozen=True)
class _Flips:
    """The detectors and observables one executed program flips."""

    detectors: tuple[int, ...]
    observables: tuple[int, ...]


def _flips(memory: MemoryCircuit, source: str, *, shots: int = 1) -> list[_Flips]:
    """Execute ``source`` and read its flips off the layouts.

    A detector is flipped when its parity over the measurements it references is
    odd, and an observable when the terminal samples over its readout support are
    odd. A syndrome bit is read at ``round_index * len(checks) + position``,
    which is the stride the lowerer emits, so the position is the check's place
    in the code's ``checks`` tuple and never its declared ``index``.

    This is a reader, not a second derivation of the model: it is written here
    rather than imported so that the model and this file do not share one
    layout-reading implementation.
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

    def measurement_bit(
        classical_row: list[int], sample_row: list[int], reference: MeasurementRef
    ) -> int:
        if reference.round_index is None:
            return int(sample_row[reference.wire])
        return int(
            classical_row[reference.round_index * checks + positions[reference.wire]]
        )

    def parity(classical_row: list[int], sample_row: list[int], references) -> int:
        return (
            sum(
                measurement_bit(classical_row, sample_row, reference)
                for reference in references
            )
            % 2
        )

    read: list[_Flips] = []
    for classical_row, sample_row in zip(classical, samples, strict=True):
        read.append(
            _Flips(
                detectors=tuple(
                    detector.index
                    for detector in memory.detectors.detectors
                    if parity(classical_row, sample_row, detector.parity)
                ),
                observables=tuple(
                    observable.index
                    for observable in memory.observables.observables
                    if parity(classical_row, sample_row, observable.measurement_parity)
                ),
            )
        )
    return read


def _injected_source(memory: MemoryCircuit, fired: Sequence[_Mechanism]) -> str:
    """Return one source string carrying every fired mechanism's forced error.

    Each mechanism is injected by the detector-error-model injector for its kind,
    into the *current* source rather than into the original one, because an
    injector rewrites one anchor and returns a new string.
    """

    current = memory
    for record in fired:
        if record.kind == "data":
            source = _inject_data_flip(
                current, round_index=record.round_index, wire=record.wire
            )
        elif record.kind == "measurement":
            source = _inject_measurement_flip(
                current, round_index=record.round_index, ancilla_wire=record.wire
            )
        else:
            raise ValueError(f"unknown mechanism kind {record.kind!r}")
        current = replace(current, source=source)
    return current.source


def _rate_tolerance(predicted: float, *, shots: int) -> float:
    """Return the band a sampled rate may sit in, in binomial standard errors."""

    rate = max(predicted, 1.0 / shots)
    return _RATE_TOLERANCE_SIGMAS * math.sqrt(rate * (1.0 - rate) / shots)


@pytest.mark.parametrize("rounds", (1, 3))
def test_the_noiseless_memory_circuit_flips_no_detector_or_observable(
    rounds: int,
) -> None:
    memory = build_memory_circuit(SteaneCode(), rounds=rounds)
    read = _flips(memory, memory.source, shots=8)

    assert len(read) == 8
    assert {flip.detectors for flip in read} == {()}
    assert {flip.observables for flip in read} == {()}
    # The assertion is only meaningful if the layout has one detector per Z-type
    # check per round boundary plus one per X-type check per interior round, so
    # the three-round case is what puts X-type detectors in front of it. An
    # X-type ancilla is prepared in ``|+>``, so its raw round-zero bit is random
    # and a layout that read raw bits instead of parities would fail here.
    assert len(memory.detectors) == 3 * (rounds + 1) + 3 * (rounds - 1)
    assert len(memory.observables) == 1


def test_a_single_data_error_is_detected_and_its_syndrome_identifies_the_wire() -> None:
    """Every single X error is detected, and the seven syndromes are distinct.

    A perfect distance-three code maps each of its seven correctable errors to a
    distinct non-empty syndrome, and the seven non-zero three-bit vectors have
    weight spectrum ``{1: 3, 2: 3, 3: 1}``. Both facts are asserted here against
    the executed circuit rather than against the check matrix, so a layout that
    numbered or ordered its detectors differently would still pass.
    """

    code = SteaneCode()
    memory = build_memory_circuit(code, rounds=1)
    syndromes: dict[int, tuple[int, ...]] = {}
    logicals: set[int] = set()
    for wire in code.data_wires:
        read = _flips(memory, _inject_data_flip(memory, round_index=0, wire=wire))[0]
        syndromes[wire] = read.detectors
        if read.observables:
            logicals.add(wire)

    assert all(syndrome for syndrome in syndromes.values())
    assert len(set(syndromes.values())) == len(code.data_wires)
    assert Counter(len(syndrome) for syndrome in syndromes.values()) == {
        1: 3,
        2: 3,
        3: 1,
    }
    # A single X error becomes a logical error exactly when it anticommutes with
    # the declared logical Z operator, which is exactly when it lies in that
    # operator's support.
    (observable,) = tuple(item for item in code.logical_observables if not item.x_wires)
    assert logicals == set(observable.support)
    assert len(logicals) == code.distance


def test_a_single_measurement_error_flips_the_pair_its_round_compares() -> None:
    """Each Z-type ancilla's syndrome bit is compared against the terminal data.

    At one round a Z-type check has two detectors: the round-zero syndrome bit
    and the terminal parity. Flipping that one measurement therefore flips both,
    and an X-type check has no round-zero detector to flip at one round, so the
    same injection on an X-type ancilla is invisible. Both halves of that
    statement are asserted, because a layout that gave the two classes the same
    detector structure would pass the first and fail the second.
    """

    code = SteaneCode()
    memory = build_memory_circuit(code, rounds=1)
    z_ancillas = {
        check.ancilla_wire for check in code.checks if check.stabilizer.z_wires
    }
    x_ancillas = {
        check.ancilla_wire for check in code.checks if check.stabilizer.x_wires
    }

    for ancilla in sorted(z_ancillas):
        read = _flips(
            memory,
            _inject_measurement_flip(memory, round_index=0, ancilla_wire=ancilla),
        )[0]
        assert len(read.detectors) == 2
        assert not read.observables

    for ancilla in sorted(x_ancillas):
        read = _flips(
            memory,
            _inject_measurement_flip(memory, round_index=0, ancilla_wire=ancilla),
        )[0]
        assert read.detectors == ()
        assert not read.observables


@pytest.mark.parametrize("rounds,weight_histogram", _MODEL_ROUNDS)
def test_the_detector_error_model_has_a_weight_three_mechanism(
    rounds: int, weight_histogram: dict[int, int]
) -> None:
    """A hyperedge appears once per round, and matching refuses the model."""

    memory = build_memory_circuit(SteaneCode(), rounds=rounds)
    model = DetectorErrorModel.from_memory_circuit(
        memory,
        noise=PhenomenologicalNoise(
            data_flip=_NOISE_PROBABILITY, measurement_flip=_NOISE_PROBABILITY
        ),
    )
    hyperedges = tuple(error for error in model.errors if len(error.detectors) == 3)
    mechanisms = _mechanisms(
        memory,
        PhenomenologicalNoise(
            data_flip=_NOISE_PROBABILITY, measurement_flip=_NOISE_PROBABILITY
        ),
    )

    # One mechanism per data wire and one per check ancilla, per round, before
    # any signature is merged.
    assert len(mechanisms) == rounds * (
        memory.code.num_data_qubits + memory.code.num_ancilla_qubits
    )
    assert len(model.errors) <= len(mechanisms)
    assert Counter(len(error.detectors) for error in model.errors) == weight_histogram
    assert set(weight_histogram) <= {1, 2, 3}
    assert len(hyperedges) == rounds
    assert all(error.observables == () for error in hyperedges)

    with pytest.raises(CapabilityError, match="hyperedge"):
        MinimumWeightMatchingDecoder.from_detector_error_model(model)


@pytest.mark.parametrize(
    "code", (RepetitionCode(distance=3), RotatedSurfaceCode(distance=3))
)
def test_the_same_decoder_accepts_the_other_two_codes(
    code: StabilizerCode,
) -> None:
    """The refusal above is a property of the Steane code, not of the decoder.

    Both of these codes carry a weight-three data error into at most two
    detectors, so their detector error models are graphlike, the matching decoder
    builds a graph from them, and it decodes a syndrome the model itself sampled.
    Without this contrast the refusal asserted above could be read as the decoder
    failing on any code.
    """

    memory = build_memory_circuit(code, rounds=1)
    model = DetectorErrorModel.from_memory_circuit(
        memory,
        noise=PhenomenologicalNoise(
            data_flip=_NOISE_PROBABILITY, measurement_flip=_NOISE_PROBABILITY
        ),
    )

    assert max(len(error.detectors) for error in model.errors) == 2
    decoder = MinimumWeightMatchingDecoder.from_detector_error_model(model)
    sample = model.dem_sampling(shots=1, seed=_SEED)
    syndrome = tuple(
        int(index) for index in sample.detectors[0].nonzero().flatten().tolist()
    )
    result = decoder.decode(syndrome)

    assert len(result.observables) <= model.num_observables
    assert result.weight >= 0.0


def test_sampled_rates_agree_with_the_circuit_simulator() -> None:
    """Compare the model's marginals against programs the simulator executed.

    Every mechanism fires independently per shot, the fired set is injected into
    one program, and that program is executed once. The draw is
    ``random.Random`` at the module's fixed seed, so the counts are reproducible.
    The comparison is per index rather than on a summary statistic, so a single
    wrong mechanism is located by the failure message.
    """

    memory = build_memory_circuit(SteaneCode(), rounds=1)
    noise = PhenomenologicalNoise(
        data_flip=_NOISE_PROBABILITY, measurement_flip=_NOISE_PROBABILITY
    )
    model = DetectorErrorModel.from_memory_circuit(memory, noise=noise)
    mechanisms = _mechanisms(memory, noise)
    generator = random.Random(_SEED)
    detector_counts = [0] * model.num_detectors
    observable_counts = [0] * model.num_observables
    for _ in range(_SAMPLED_SHOTS):
        fired = [
            record for record in mechanisms if generator.random() < record.probability
        ]
        read = _flips(memory, _injected_source(memory, fired))[0]
        for index in read.detectors:
            detector_counts[index] += 1
        for index in read.observables:
            observable_counts[index] += 1

    for label, counts, rates in (
        ("detector", detector_counts, model.detector_rates()),
        ("observable", observable_counts, model.observable_rates()),
    ):
        assert rates.shape == (len(counts),)
        for index, count in enumerate(counts):
            rate = count / _SAMPLED_SHOTS
            modelled = float(rates[index])
            tolerance = _rate_tolerance(modelled, shots=_SAMPLED_SHOTS)
            assert abs(rate - modelled) <= tolerance, (
                f"{label} {index} sampled {count}/{_SAMPLED_SHOTS} = {rate}, which "
                f"is outside {tolerance} of the modelled {modelled}"
            )


def test_the_model_reproduces_its_observable_rate_when_sampled() -> None:
    """The observable's marginal rate is a merged rate, not one mechanism's.

    Three of the seven data errors are logical, and one of them flips all three
    Z-type detectors, so the observable's rate is assembled from several
    mechanisms including a hyperedge. Sampling the model at a large shot budget
    reproduces the rate the model states, which is what licenses quoting that
    marginal as the Steane code's quantitative claim while the decoder boundary
    stands.
    """

    memory = build_memory_circuit(SteaneCode(), rounds=1)
    model = DetectorErrorModel.from_memory_circuit(
        memory,
        noise=PhenomenologicalNoise(
            data_flip=_NOISE_PROBABILITY, measurement_flip=_NOISE_PROBABILITY
        ),
    )
    sample = model.dem_sampling(shots=_DEM_SAMPLING_SHOTS, seed=_SEED)
    counts = [int(value) for value in sample.observables.sum(dim=0).tolist()]
    rates = model.observable_rates()

    assert sample.shots == _DEM_SAMPLING_SHOTS
    assert rates.shape == (model.num_observables,)
    for index, count in enumerate(counts):
        modelled = float(rates[index])
        tolerance = _rate_tolerance(modelled, shots=_DEM_SAMPLING_SHOTS)
        assert abs(count / _DEM_SAMPLING_SHOTS - modelled) <= tolerance, (
            f"observable {index} sampled {count}/{_DEM_SAMPLING_SHOTS}, which is "
            f"outside {tolerance} of the modelled {modelled}"
        )


def test_the_model_states_the_logical_rate_a_matching_decoder_would_need() -> None:
    """More rounds raise the logical rate, because more rounds add more chances.

    A distance-three perfect code cannot suppress its logical error rate by
    adding rounds: the phenomenological model's logical rate grows with the round
    count. The assertion is directional and records the shape of the model rather
    than a threshold claim, which this file is not able to make.
    """

    rates = []
    for rounds in (1, 2, 3):
        memory = build_memory_circuit(SteaneCode(), rounds=rounds)
        model = DetectorErrorModel.from_memory_circuit(
            memory,
            noise=PhenomenologicalNoise(
                data_flip=_NOISE_PROBABILITY, measurement_flip=_NOISE_PROBABILITY
            ),
        )
        (rate,) = model.observable_rates().tolist()
        rates.append(float(rate))

    assert rates == sorted(rates)
    assert rates[0] < rates[-1]
