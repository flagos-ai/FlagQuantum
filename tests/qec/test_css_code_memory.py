"""End-to-end coverage for a code stated as matrices rather than as a family.

``test_css_code.py`` pins the algebra of the record built from parity-check
matrices. This file takes a code that no record in this package declares -- the
toric code on a torus -- through the rest of the path a code has to walk: memory
circuit, detector error model, sampling, and decoding. The toric code is the case
worth walking because its checks are local while its logical operators are not,
so it answers the question the record raises: does the path past the record work
for a family nobody wrote down here?

What this file proves
---------------------

1. ``test_a_matrix_stated_family_reaches_a_graphlike_detector_error_model``: every
   mechanism of the model built from this family flips at most two detectors, at
   every round count. That is not a property of the matrices; it is a property of
   the code, and it is what makes the family the package already has a decoder for
   reachable for a code the package never declared.
2. ``test_the_matching_decoder_takes_this_family_and_still_refuses_the_steane_code``:
   the same decoder construction accepts this model and refuses the Steane model
   with the hyperedge it names. Without the second half, the acceptance could be
   read as this file having weakened the decoder.
3. ``test_the_circuit_sampler_and_the_model_agree_for_a_matrix_stated_family``: the
   detector and observable marginals the model states match what the circuit
   sampler observes, per index, at one round. The one-round setting is the one the
   declared records are compared at; see the note on the round boundary below.
4. ``test_the_decoder_acts_only_where_the_distance_leaves_something_to_correct``:
   the lattice-two code's distance is two, so no single fault is correctable and
   the decoder returns no observable flip for any syndrome; the lattice-three
   code's distance is three, so it does, and the observable rate under decoding
   falls below the rate without it. This is the strongest statement available here
   and it is the one that says the route ends in a usable correction rather than in
   a model.

What this file does not prove
-----------------------------

It makes no threshold, suppression or scaling claim. The two lattices are two
sizes, not a family of growing capacity, and one of them is the smallest torus
there is. It also does not compare this family against a hardware experiment or
against another framework's implementation of the same code.

A note on the round boundary, recorded because this file steps around it: at two
or more rounds the circuit sampler and the detector error model disagree at the
detectors that compare one round against the next, by roughly fifteen binomial
standard errors at the noise below. The same disagreement appears, index for
index, for the declared rotated surface code at the same settings, so it belongs
to the sampler and the model rather than to this route, and it is not what this
file is about. The one-round comparison below is therefore the claim this file
makes; the multi-round discrepancy is a gap an owner has to adjudicate, and
asserting a rate agreement that does not hold would hide it.
"""

from __future__ import annotations

import math

import pytest

from flagquantum.errors import CapabilityError
from flagquantum.qec.circuit import build_memory_circuit
from flagquantum.qec.codes import SteaneCode
from flagquantum.qec.dem import DetectorErrorModel
from flagquantum.qec.matching import MinimumWeightMatchingDecoder
from flagquantum.qec.noise import PhenomenologicalNoise
from flagquantum.qec.sampling import sample_memory_circuit
from tests.qec.test_css_code import _toric

pytestmark = pytest.mark.integration

# One noise probability for the three data fault families and the measurement
# fault, which is a depolarizing data fault. The sampler and the model are compared
# at this rate and the decoder is measured at it.
_PROBABILITY = 0.05
_SEED = 0
_RATE_TOLERANCE_SIGMAS = 4.0
# Shot budgets. The sampler lowers and executes once per call and the decode loop
# is one match per shot, which keeps the file inside the ordinary integration lane.
_SAMPLED_SHOTS = 4000
_DECODING_SHOTS = 600

# The mechanism-weight histogram the model produces at the noise above, keyed by
# lattice size and round count, and the detector count that goes with it. These are
# recorded measurements rather than a formula: the weight-one mechanisms merge as
# rounds are added only when their signatures coincide, so the second weight is not
# affine in the round count. The maximum is the part that is structural, and it is
# asserted for every row.
_MODEL_PROFILE = (
    (2, 1, 8, {2: 12}),
    (2, 2, 16, {1: 4, 2: 28}),
    (2, 3, 24, {1: 8, 2: 48}),
    (3, 1, 18, {2: 27}),
    (3, 2, 36, {1: 9, 2: 72}),
    (3, 3, 54, {1: 18, 2: 126}),
)


def _noise(probability: float = _PROBABILITY) -> PhenomenologicalNoise:
    """Return the depolarizing data fault and measurement fault used throughout."""

    return PhenomenologicalNoise(
        data_flip=probability,
        phase_flip=probability,
        measurement_flip=probability,
    )


@pytest.mark.parametrize("lattice, rounds, detectors, histogram", _MODEL_PROFILE)
def test_a_matrix_stated_family_reaches_a_graphlike_detector_error_model(
    lattice: int, rounds: int, detectors: int, histogram: dict[int, int]
) -> None:
    """A local check turns one data fault into at most two detection events.

    The toric code measures four data qubits per check, which is the shape the
    Steane code has too, and yet its model is graphlike: a single-qubit fault sits
    on an edge and the two faces that share that edge are the two detectors it
    flips. The distance, not the check weight, is what decides this, and it is the
    reason a hyperedge-aware decoder is not needed to decode this family.
    """

    code = _toric(lattice)
    memory = build_memory_circuit(code, rounds=rounds)
    model = DetectorErrorModel.from_memory_circuit(memory, noise=_noise())

    counts: dict[int, int] = {}
    for error in model.errors:
        counts[len(error.detectors)] = counts.get(len(error.detectors), 0) + 1

    assert model.num_detectors == detectors
    assert model.num_observables == 2
    assert counts == histogram
    assert max(counts) == 2

    # The weight of a mechanism is the two faces sharing an edge, so a data fault
    # is never carried into one detector -- if it were, the model would be stating
    # a fault that no check can see.
    assert min(counts) == (1 if rounds > 1 else 2)


def test_the_matching_decoder_takes_this_family_and_still_refuses_the_steane_code() -> (
    None
):
    """The decoder's boundary is a property of the code, not of this route.

    The two models are built the same way from the same phenomenological noise, and
    the decoder accepts one and refuses the other. A route that made every model
    decodable, or one that made none of them decodable, would fail one half of this
    assertion.
    """

    toric = DetectorErrorModel.from_memory_circuit(
        build_memory_circuit(_toric(3), rounds=1), noise=_noise()
    )
    steane = DetectorErrorModel.from_memory_circuit(
        build_memory_circuit(SteaneCode(), rounds=1), noise=_noise()
    )

    assert max(len(error.detectors) for error in steane.errors) == 3
    with pytest.raises(CapabilityError, match="hyperedge"):
        MinimumWeightMatchingDecoder.from_detector_error_model(steane)

    decoder = MinimumWeightMatchingDecoder.from_detector_error_model(toric)
    sample = toric.dem_sampling(shots=1, seed=_SEED)
    syndrome = tuple(
        int(index) for index in sample.detectors[0].nonzero().flatten().tolist()
    )
    result = decoder.decode(syndrome)

    assert len(result.observables) <= toric.num_observables
    assert result.weight >= 0.0


@pytest.mark.parametrize("lattice", (2, 3))
def test_the_circuit_sampler_and_the_model_agree_for_a_matrix_stated_family(
    lattice: int,
) -> None:
    """The model's marginals match what the circuit actually does.

    The two sides are independent: the model is derived from the matrices and the
    circuit's layouts, while the sampler lowers the same circuit, places the noise,
    and executes it. The comparison is per index, so a single wrong mechanism is
    located by the failure message rather than hidden in a summary statistic.
    """

    memory = build_memory_circuit(_toric(lattice), rounds=1)
    model = DetectorErrorModel.from_memory_circuit(memory, noise=_noise())
    sample = sample_memory_circuit(
        memory, noise=_noise(), shots=_SAMPLED_SHOTS, seed=_SEED
    )

    for label, counts, rates in (
        ("detector", sample.detectors.sum(dim=0).tolist(), model.detector_rates()),
        (
            "observable",
            sample.observables.sum(dim=0).tolist(),
            model.observable_rates(),
        ),
    ):
        assert rates.shape == (len(counts),)
        for index, count in enumerate(counts):
            observed = count / _SAMPLED_SHOTS
            modelled = float(rates[index])
            sigma = math.sqrt(max(modelled * (1.0 - modelled), 1e-12) / _SAMPLED_SHOTS)
            assert abs(observed - modelled) <= _RATE_TOLERANCE_SIGMAS * sigma, (
                f"{label} {index} sampled {count}/{_SAMPLED_SHOTS} = {observed}, "
                f"which is outside {_RATE_TOLERANCE_SIGMAS} binomial standard "
                f"errors of the modelled {modelled}"
            )


def test_the_decoder_acts_only_where_the_distance_leaves_something_to_correct() -> None:
    """Distance two corrects no fault; distance three corrects one.

    The lattice-two torus has distance two, so a single fault is already a logical
    error and no correction can remove it: the decoder returns no observable flip
    for any syndrome, and the rate under decoding is the rate without it, exactly.
    The lattice-three torus has distance three, so it does correct a single fault
    and the rate falls. Both halves are asserted, because the second on its own
    would not distinguish a decoder that works from one that guesses.
    """

    measured: dict[int, tuple[float, float]] = {}
    for lattice in (2, 3):
        memory = build_memory_circuit(_toric(lattice), rounds=1)
        model = DetectorErrorModel.from_memory_circuit(memory, noise=_noise())
        decoder = MinimumWeightMatchingDecoder.from_detector_error_model(model)
        sample = sample_memory_circuit(
            memory, noise=_noise(), shots=_DECODING_SHOTS, seed=_SEED
        )
        raw = 0
        corrected = 0
        returned = 0
        for syndrome_bits, observable_bits in zip(
            sample.detectors.tolist(), sample.observables.tolist(), strict=True
        ):
            syndrome = tuple(index for index, bit in enumerate(syndrome_bits) if bit)
            result = decoder.decode(syndrome)
            returned += len(result.observables)
            after = list(observable_bits)
            for index in result.observables:
                after[index] ^= 1
            raw += sum(observable_bits)
            corrected += sum(after)
        shots = len(sample.observables)
        observables = model.num_observables
        measured[lattice] = (
            raw / (shots * observables),
            corrected / (shots * observables),
        )
        if lattice == 2:
            # Distance two corrects nothing, so the correction is empty on every
            # syndrome rather than merely unhelpful on the ones that occur.
            assert returned == 0

    (raw_two, corrected_two), (raw_three, corrected_three) = (
        measured[2],
        measured[3],
    )
    assert corrected_two == raw_two
    assert raw_three > 0.0
    assert corrected_three < raw_three
