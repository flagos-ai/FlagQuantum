"""Cross-check the detector error model against the circuit simulator.

The model's rates are compared here against what the real hybrid-compiler
program does, not against a second statement of the same arithmetic. Two
comparisons run, both deterministic: a fixed seed draws every sample, and the
pair sweep takes no sample at all.

1. ``test_mechanism_pairs_compose_by_xor`` forces every *pair* of the fifteen
   mechanisms a distance-three circuit has into one program, executes it, and
   asserts the flipped detectors and observables are the symmetric difference of
   the two single-mechanism signatures. The sweep is exhaustive
   (``C(15, 2) = 105`` executions), so no statistic and no flakiness enter, and
   it is the stage's strongest structural evidence: XOR composition is the
   assumption ``_merge_mechanisms`` rests on.
2. ``test_sampled_detector_and_observable_rates_agree`` draws an independent
   Bernoulli per mechanism per shot, injects the fired set, executes, and
   compares the empirical detector and observable rates against
   ``detector_rates()`` and ``observable_rates()`` at ``d = 3`` and ``d = 5``.
   ``test_merged_mechanisms_agree_with_sampled_rates`` runs the same comparison
   on a code whose mechanisms collide, which is the only construction here that
   puts the merge arithmetic in front of the simulator: on the repetition code
   every mechanism has its own signature (measured at ``d = 3`` and ``d = 5``),
   so merging and discarding never fire there.

What this file does and does not prove
--------------------------------------

It proves that mechanisms compose by XOR in the simulator, and that a model
built by merging identical signatures predicts the marginal rates the circuit
actually produces — at two distances, and on a code where merging fires. The
merge rule ``p1 * (1 - p2) + p2 * (1 - p1)`` is exactly what the collision case
measures: three mechanisms that share a signature flip their detector when an
odd number of them fires, so a model that added their probabilities instead
would disagree with the sample.

It does **not** independently re-derive the signatures. Every source in this
file is built by the Task 6 injectors, which is what keeps injection in one
place, but the single-mechanism signatures the pair sweep compares against come
from ``_forced_signature`` — the same engine ``from_memory_circuit`` builds the
model from. A signature that engine computes wrongly is therefore wrong on both
sides of the comparison, and this file cannot see it. What this file does read
for itself is the *combined* result, through its own layout reader, so an engine
that read the layouts differently for one mechanism than for several would fail
the XOR assertion rather than agree with itself.

The one genuinely independent check available — parsing the emitted stim text
with the real ``stim`` package — was run at developer time in Task 5 and is not
a committed test here, because ``stim`` is not a dependency of this repository.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, replace

import pytest

from flagquantum.compiler._hybrid import INDEX, capture_source, lower_dynamic_program
from flagquantum.qec.circuit import (
    MeasurementRef,
    MemoryCircuit,
    build_memory_circuit,
)
from flagquantum.qec.codes import CodeCheck, RepetitionCode
from flagquantum.qec.dem import (
    DetectorErrorModel,
    _forced_signature,
    _inject_data_flip,
    _inject_measurement_flip,
    _Mechanism,
    _mechanisms,
)
from flagquantum.qec.noise import PhenomenologicalNoise
from flagquantum.qec.pauli import Pauli
from flagquantum.runtime.dynamic.hybrid_session import execute_hybrid_dynamic_session

pytestmark = pytest.mark.integration

# Seed for every draw in this file, so the sampled cases are reproducible.
_SEED = 0

# The sampled rate is accepted when it sits within this many binomial standard
# errors of the modelled rate. Four covers a per-detector miss probability of
# about 6e-5, so a single wrong mechanism is located rather than averaged away,
# while a correct model is not failed by its own sample. The collision case
# below is what this number protects: a plain-sum merge moves its modelled
# detector rate by about 2.5 bands, so a tolerance of 12 sigmas or more would
# accept that wrong rule, and no other assertion in the file would notice.
_RATE_TOLERANCE_SIGMAS = 4.0

# Noise and shot budgets. The measured per-shot cost is about 9 ms at ``d = 3``
# and 25 ms at ``d = 5`` (lowering dominates), which is what keeps the whole
# file inside the ordinary ``integration`` lane's budget.
_REPETITION_NOISE_PROBABILITY = 0.05
_COLLIDING_NOISE_PROBABILITY = 0.25
_DISTANCE_THREE_SHOTS = 1000
_DISTANCE_FIVE_SHOTS = 500
_COLLIDING_SHOTS = 1000


def _injected_source(memory: MemoryCircuit, fired: Sequence[_Mechanism]) -> str:
    """Return one source string carrying every fired mechanism's forced error.

    Each mechanism is injected by the Task 6 injector for its kind, into the
    *current* source rather than into the original one, because an injector
    rewrites one anchor and returns a new string. This helper only chains the
    two injectors that exist; it is not a second injection implementation.
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
            # Unreachable through ``_mechanisms``; stated so a record whose kind
            # is neither flip is refused here rather than injected as one.
            raise ValueError(f"unknown mechanism kind {record.kind!r}")
        current = replace(current, source=source)
    return current.source


def _measured_flips(
    memory: MemoryCircuit, source: str
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Execute ``source`` once and read its flips off the layouts.

    A detector is flipped when its parity over the measurements it references is
    odd, and an observable when the terminal samples over its readout support
    are odd. A syndrome bit is read at ``round_index * len(checks) + position``,
    which is the stride the lowerer emits, so the position is the check's place
    in the code's tuple and never its declared ``index``.
    """

    program = capture_source(source, (INDEX,))
    lowered = lower_dynamic_program(
        program,
        (memory.rounds,),
        max_dynamic_measurements=memory.rounds * len(memory.code.checks),
    )
    execution = execute_hybrid_dynamic_session(
        lowered.circuit, shots=1, seed=_SEED, strategy="trajectory"
    )
    classical = execution.classical_bits.tolist()[0]
    sample = execution.samples.tolist()[0]
    checks = len(memory.code.checks)
    positions = {
        check.ancilla_wire: position
        for position, check in enumerate(memory.code.checks)
    }

    def measurement_bit(reference: MeasurementRef) -> int:
        if reference.round_index is None:
            return int(sample[reference.wire])
        return int(
            classical[reference.round_index * checks + positions[reference.wire]]
        )

    def parity(references: Sequence[MeasurementRef]) -> int:
        return sum(measurement_bit(reference) for reference in references) % 2

    flipped = [0] * len(memory.detectors.detectors)
    for detector in memory.detectors.detectors:
        flipped[detector.index] = parity(detector.parity)
    detectors = tuple(index for index, bit in enumerate(flipped) if bit)
    observables = tuple(
        observable.index
        for observable in memory.observables.observables
        if parity(observable.measurement_parity)
    )
    return detectors, observables


def _xor(left: Sequence[int], right: Sequence[int]) -> tuple[int, ...]:
    """Return the ascending symmetric difference of two ascending index tuples."""

    return tuple(sorted(set(left) ^ set(right)))


def _rate_tolerance(predicted: float, *, shots: int) -> float:
    """Return the band a sampled rate may sit in, in binomial standard errors.

    ``_RATE_TOLERANCE_SIGMAS`` standard errors of a binomial rate estimated from
    ``shots`` draws. A detector the model says never fires would give a
    zero-width band, which is the unfalsifiable-check shape, so the rate that
    enters the formula is floored at one success in the budget: the smallest
    rate such a sample can resolve at all. No ceiling is needed, because these
    noise records predict no rate near one.
    """

    rate = max(predicted, 1.0 / shots)
    return _RATE_TOLERANCE_SIGMAS * math.sqrt(rate * (1.0 - rate) / shots)


def _sampled_flip_counts(
    memory: MemoryCircuit, noise: PhenomenologicalNoise, *, shots: int
) -> tuple[list[int], list[int]]:
    """Count the shots in which the simulator flips each detector and observable.

    Every mechanism fires independently per shot with its own probability, the
    fired set is injected into one program, and that program is executed once.
    The draw is ``random.Random`` at the module's fixed seed, so the counts are
    reproducible; a different seed would change the sample but not, at these
    budgets and tolerances, a correct run's verdict.
    """

    mechanisms = _mechanisms(memory, noise)
    generator = random.Random(_SEED)
    detector_counts = [0] * len(memory.detectors.detectors)
    observable_counts = [0] * len(memory.observables.observables)
    for _ in range(shots):
        fired = [
            record for record in mechanisms if generator.random() < record.probability
        ]
        detectors, observables = _measured_flips(
            memory, _injected_source(memory, fired)
        )
        for index in detectors:
            detector_counts[index] += 1
        for index in observables:
            observable_counts[index] += 1
    return detector_counts, observable_counts


def _assert_rates_agree(
    counts: Sequence[int], predicted, *, shots: int, label: str
) -> None:
    """Assert every index's sampled rate against its modelled rate.

    The assertion is per index, not on a summary statistic, so a single wrong
    mechanism is located by the failure message instead of being averaged into a
    total that still looks close.
    """

    assert predicted.shape == (len(counts),)
    for index, count in enumerate(counts):
        rate = count / shots
        modelled = float(predicted[index])
        tolerance = _rate_tolerance(modelled, shots=shots)
        assert abs(rate - modelled) <= tolerance, (
            f"{label} {index} sampled {count}/{shots} = {rate}, which is outside "
            f"{tolerance} of the modelled {modelled}"
        )


def test_mechanism_pairs_compose_by_xor() -> None:
    """Every mechanism pair flips the symmetric difference of its members.

    Two mechanisms that share a signature are merged by the parity rule because
    a decoder only sees their combined parity, and that rule is correct only if
    forced errors compose by XOR. Each pair is injected into one program and
    executed once, and the pair's flips are compared against the two
    single-mechanism signatures the model is built from. The sweep is exhaustive
    at ``d = 3``, so nothing here depends on a draw.
    """

    memory = build_memory_circuit(RepetitionCode(3), rounds=3)
    noise = PhenomenologicalNoise(
        data_flip=_REPETITION_NOISE_PROBABILITY,
        measurement_flip=_REPETITION_NOISE_PROBABILITY,
    )
    mechanisms = _mechanisms(memory, noise)
    assert len(mechanisms) == 15
    signatures = [
        _forced_signature(memory, _injected_source(memory, (record,)))
        for record in mechanisms
    ]

    checked = 0
    for first, second in itertools.combinations(range(len(mechanisms)), 2):
        measured = _measured_flips(
            memory, _injected_source(memory, (mechanisms[first], mechanisms[second]))
        )
        expected = (
            _xor(signatures[first][0], signatures[second][0]),
            _xor(signatures[first][1], signatures[second][1]),
        )
        assert measured == expected, (
            f"mechanisms {first} and {second} flip {measured}, not the XOR of "
            f"their signatures {expected}"
        )
        checked += 1
    assert checked == 15 * 14 // 2


@pytest.mark.parametrize(
    ("distance", "shots"),
    ((3, _DISTANCE_THREE_SHOTS), (5, _DISTANCE_FIVE_SHOTS)),
    ids=("distance-3", "distance-5"),
)
def test_sampled_detector_and_observable_rates_agree(distance: int, shots: int) -> None:
    """Empirical rates from the simulator match the model's marginal rates.

    The sample is drawn per mechanism, so a shot exercises whatever set fires
    together, and the model it is compared against was built by forced execution
    of one mechanism at a time. Agreement is what ties the merge arithmetic and
    the marginal-rate formula to the circuit.
    """

    memory = build_memory_circuit(RepetitionCode(distance), rounds=distance)
    noise = PhenomenologicalNoise(
        data_flip=_REPETITION_NOISE_PROBABILITY,
        measurement_flip=_REPETITION_NOISE_PROBABILITY,
    )
    model = DetectorErrorModel.from_memory_circuit(memory, noise=noise)
    detector_counts, observable_counts = _sampled_flip_counts(
        memory, noise, shots=shots
    )

    _assert_rates_agree(
        detector_counts, model.detector_rates(), shots=shots, label="detector"
    )
    _assert_rates_agree(
        observable_counts, model.observable_rates(), shots=shots, label="observable"
    )


@dataclass(frozen=True)
class _SharedSupportCode:
    """A code whose single check spans every data wire.

    A data flip on wire 0, 1 or 2 in one round therefore flips the same
    detectors and the same observable, so the three locations share one
    signature and the model must merge them into a single mechanism. The
    repetition code gives every mechanism its own signature, so this is the
    construction that lets the merge rule reach the simulator at all.
    """

    distance: int = 3

    @property
    def num_data_qubits(self) -> int:
        return 3

    @property
    def num_ancilla_qubits(self) -> int:
        return 1

    @property
    def data_wires(self) -> tuple[int, ...]:
        return (0, 1, 2)

    @property
    def ancilla_wires(self) -> tuple[int, ...]:
        return (3,)

    @property
    def checks(self) -> tuple[CodeCheck, ...]:
        return (
            CodeCheck(
                index=0,
                stabilizer=Pauli(z_wires=(0, 1, 2)),
                ancilla_wire=3,
                cnot_wires=((0, 3), (1, 3), (2, 3)),
            ),
        )

    @property
    def stabilizers(self) -> tuple[Pauli, ...]:
        return tuple(check.stabilizer for check in self.checks)

    @property
    def logical_observables(self) -> tuple[Pauli, ...]:
        return (Pauli(z_wires=(0, 1, 2)),)


def test_merged_mechanisms_agree_with_sampled_rates() -> None:
    """Three colliding mechanisms merge by parity, and the sample says so.

    The noise is raised for this case so the two candidate rules separate far
    beyond the tolerance band: three 0.25 flips merge to 0.46875 by parity and
    to 0.75 by a plain sum, which moves a modelled detector rate by twice the
    band. A model that added the probabilities would fail here and nowhere else
    in this file.
    """

    memory = build_memory_circuit(_SharedSupportCode(), rounds=2)
    noise = PhenomenologicalNoise(
        data_flip=_COLLIDING_NOISE_PROBABILITY,
        measurement_flip=_COLLIDING_NOISE_PROBABILITY,
    )
    assert len(_mechanisms(memory, noise)) == 8
    model = DetectorErrorModel.from_memory_circuit(memory, noise=noise)
    # Two rounds of three colliding data flips plus two measurement flips.
    assert model.num_errors == 4

    detector_counts, observable_counts = _sampled_flip_counts(
        memory, noise, shots=_COLLIDING_SHOTS
    )
    _assert_rates_agree(
        detector_counts,
        model.detector_rates(),
        shots=_COLLIDING_SHOTS,
        label="detector",
    )
    _assert_rates_agree(
        observable_counts,
        model.observable_rates(),
        shots=_COLLIDING_SHOTS,
        label="observable",
    )
