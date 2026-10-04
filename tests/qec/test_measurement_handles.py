"""The recorded-bit layer: a measurement handle names a bit, and one is readable.

A detector error model and a detection event both speak about *parities* of
recorded bits. This file pins the layer underneath them, where a bit is
addressed by the handle that names it: the handle vector a configured experiment
declares, the two readings over a chosen sub-vector, and the property that makes
the layer worth having -- a detector's parity recomputed from the handles it
names is the detector's parity, and a handle no layout names is readable anyway.

The tests are written against measurements rather than against a second opinion
about the layout. Where a count is asserted it is the arithmetic the code and the
round count state, and where a rate is asserted it is the rate the noise record
was given.

Most of the file reads a record the sampler produced, and the stabilizer engine
is an optional dependency, so the file skips when `stim` is absent -- the same
convention the other files whose subject is what the sampler returns follow. The
handle arithmetic itself is covered by the model-route tests that skip with it.
"""

from __future__ import annotations

import functools

import pytest
import torch

import flagquantum.qec as q
from flagquantum.qec import (
    MeasurementRef,
    MeasurementSamples,
    PhenomenologicalNoise,
    RepetitionCode,
    RotatedSurfaceCode,
    build_memory_circuit,
    sample_memory_circuit,
    sample_memory_measurements,
)

pytestmark = pytest.mark.integration

pytest.importorskip("stim")

_X_CHECK_MAX_ROUNDS = 1


@functools.lru_cache(maxsize=None)
def _surface(distance: int, rounds: int) -> q.MemoryCircuit:
    return build_memory_circuit(RotatedSurfaceCode(distance=distance), rounds=rounds)


def _noise() -> PhenomenologicalNoise:
    return PhenomenologicalNoise(
        data_flip=0.02,
        phase_flip=0.03,
        both_flip=0.01,
        measurement_flip=0.04,
    )


def _used_handles(memory: q.MemoryCircuit) -> set[MeasurementRef]:
    used = {
        reference
        for detector in memory.detectors.detectors
        for reference in detector.parity
    }
    used |= {
        reference
        for observable in memory.observables.observables
        for reference in observable.measurement_parity
    }
    return used


def test_handle_vector_is_the_code_and_round_arithmetic() -> None:
    """One handle per check per round, then one per data wire, in that order."""

    code = RotatedSurfaceCode(distance=3)
    for rounds in (1, 2, 4):
        memory = build_memory_circuit(code, rounds=rounds)
        refs = memory.measurement_refs
        expected = [
            MeasurementRef(round_index, check.ancilla_wire)
            for round_index in range(rounds)
            for check in code.checks
        ] + [MeasurementRef(None, wire) for wire in code.data_wires]
        assert refs == tuple(expected)
        assert len(refs) == rounds * len(code.checks) + len(code.data_wires)
        # A handle vector is a set of distinct names, so a bit has one address.
        assert len(set(refs)) == len(refs)
        # The handles are code-derived, so a repetition code's vector is shorter
        # by exactly the difference in declared wires.
        assert len(RepetitionCode(distance=3).checks) < len(code.checks)


def test_every_layout_reference_is_a_declared_handle() -> None:
    """A layout cannot name a bit the experiment does not record."""

    for code in (RotatedSurfaceCode(distance=3), RepetitionCode(distance=3)):
        for rounds in (1, 3):
            memory = build_memory_circuit(code, rounds=rounds)
            assert _used_handles(memory) <= set(memory.measurement_refs)


def test_a_layout_need_not_name_every_handle() -> None:
    """Recorded and referenced are different questions, measured not assumed."""

    one_round = _surface(3, 1)
    unused = set(one_round.measurement_refs) - _used_handles(one_round)
    x_ancillas = {
        check.ancilla_wire
        for check in RotatedSurfaceCode(distance=3).checks
        if check.stabilizer.x_wires
    }
    # A one-round patch's X-type checks are deterministic in neither the initial
    # state nor the terminal readout, so nothing can be compared and no detector
    # names them; they are still measured and still recorded.
    assert unused == {MeasurementRef(0, wire) for wire in x_ancillas}
    # A second round compares against the first, which names every handle.
    longer = _surface(3, 3)
    assert set(longer.measurement_refs) - _used_handles(longer) == set()


def test_a_handle_vector_needs_a_code_that_implements_the_protocol() -> None:
    """The handle vector is read off the protocol, not off a concrete class."""

    class NotACode:
        pass

    with pytest.raises(TypeError, match="StabilizerCode protocol"):
        build_memory_circuit(NotACode(), rounds=1)  # type: ignore[arg-type]


def test_sample_memory_measurements_reads_every_handle() -> None:
    memory = _surface(3, 2)
    samples = sample_memory_measurements(memory, noise=_noise(), shots=64, seed=11)
    assert isinstance(samples, MeasurementSamples)
    assert samples.refs == memory.measurement_refs
    assert samples.num_handles == len(memory.measurement_refs)
    assert samples.shots == 64
    assert samples.outcomes.shape == (64, samples.num_handles)
    assert samples.outcomes.dtype == torch.int8
    # Outcomes are bits, one per handle per shot.
    assert set(int(value) for value in torch.unique(samples.outcomes)) <= {0, 1}


def test_a_detector_parity_is_the_parity_of_its_handles() -> None:
    """The two readings of one run cannot disagree about a shot."""

    memory = _surface(3, 2)
    noise = _noise()
    samples = sample_memory_measurements(memory, noise=noise, shots=128, seed=5)
    events = sample_memory_circuit(memory, noise=noise, shots=128, seed=5)
    assert events.shots == samples.shots
    for detector in memory.detectors.detectors:
        parity = torch.zeros(samples.shots, dtype=torch.int8)
        for reference in detector.parity:
            parity ^= samples.outcome(reference)
        assert torch.equal(parity, events.detectors[:, detector.index])
    for observable in memory.observables.observables:
        parity = torch.zeros(samples.shots, dtype=torch.int8)
        for reference in observable.measurement_parity:
            parity ^= samples.outcome(reference)
        assert torch.equal(parity, events.observables[:, observable.index])


def test_a_handle_no_layout_names_is_still_readable() -> None:
    """The layer's reason to exist: the bit is addressable on its own."""

    memory = _surface(3, _X_CHECK_MAX_ROUNDS)
    dropped = next(
        reference
        for reference in memory.measurement_refs
        if reference not in _used_handles(memory)
    )
    samples = sample_memory_measurements(memory, noise=_noise(), shots=256, seed=3)
    bits = samples.outcome(dropped)
    assert bits.shape == (256,)
    # An X-type check is measured in |+>, so its outcome is unbiased rather than
    # pinned to the known state an all-zero preparation would give a Z-type one.
    rate = float(bits.to(torch.float64).mean())
    assert 0.35 < rate < 0.65
    # It is reproducible for a seed, which is what makes it a reading at all.
    again = sample_memory_measurements(memory, noise=_noise(), shots=256, seed=3)
    assert torch.equal(bits, again.outcome(dropped))


def test_vector_is_one_boolean_per_shot_per_handle_in_the_order_asked() -> None:
    memory = _surface(3, 2)
    samples = sample_memory_measurements(memory, noise=_noise(), shots=32, seed=19)
    first, second = samples.refs[0], samples.refs[5]
    forward = samples.vector([first, second])
    assert forward.shape == (32, 2)
    assert forward.dtype == torch.bool
    assert torch.equal(forward[:, 0], samples.outcome(first).to(torch.bool))
    assert torch.equal(forward[:, 1], samples.outcome(second).to(torch.bool))
    backward = samples.vector([second, first])
    assert torch.equal(backward[:, 0], forward[:, 1])
    assert torch.equal(backward[:, 1], forward[:, 0])


def test_integer_packs_the_order_asked_first_handle_least_significant() -> None:
    memory = _surface(3, 2)
    samples = sample_memory_measurements(memory, noise=_noise(), shots=64, seed=23)
    handles = list(samples.refs[:4])
    packed = samples.integer(handles)
    assert packed.shape == (64,)
    assert packed.dtype == torch.int64
    assert int(packed.max()) <= 0b1111
    # The packing is stated here rather than inherited, so it is checked by hand
    # against the bit vector rather than against another call of the same method.
    bits = samples.vector(handles).to(torch.int64)
    manual = sum(bits[:, position] << position for position in range(4))
    assert torch.equal(packed, manual)
    # The order is the caller's, so two orders state two integers.
    reversed_packed = samples.integer(list(reversed(handles)))
    manual_reversed = sum(bits[:, position] << (3 - position) for position in range(4))
    assert torch.equal(reversed_packed, manual_reversed)
    assert not torch.equal(packed, reversed_packed)


def test_a_single_handle_reads_as_a_one_bit_integer() -> None:
    memory = _surface(3, 2)
    samples = sample_memory_measurements(memory, noise=_noise(), shots=32, seed=29)
    handle = samples.refs[3]
    assert torch.equal(
        samples.integer([handle]), samples.outcome(handle).to(torch.int64)
    )
    assert torch.equal(
        samples.vector([handle])[:, 0], samples.outcome(handle).to(torch.bool)
    )


def test_reading_a_handle_the_experiment_does_not_record_is_refused() -> None:
    memory = _surface(3, 2)
    samples = sample_memory_measurements(memory, noise=_noise(), shots=8, seed=31)
    unknown = MeasurementRef(9, 0)
    with pytest.raises(ValueError, match="is not one this experiment records"):
        samples.outcome(unknown)
    with pytest.raises(ValueError, match="is not one this experiment records"):
        samples.vector([samples.refs[0], unknown])
    with pytest.raises(ValueError, match="at least one handle"):
        samples.vector([])
    with pytest.raises(TypeError, match="MeasurementRef"):
        samples.vector([(0, 0)])  # type: ignore[list-item]


def test_the_record_refuses_a_shape_that_does_not_match_its_handles() -> None:
    memory = _surface(3, 2)
    samples = sample_memory_measurements(memory, noise=_noise(), shots=8, seed=37)
    with pytest.raises(ValueError, match="must address at least one handle"):
        MeasurementSamples(refs=(), outcomes=samples.outcomes[:, :0])
    with pytest.raises(ValueError, match="cannot repeat a handle"):
        MeasurementSamples(
            refs=(samples.refs[0], samples.refs[0]),
            outcomes=samples.outcomes[:, :2],
        )
    with pytest.raises(ValueError, match="two-dimensional"):
        MeasurementSamples(refs=samples.refs, outcomes=samples.outcomes[:, 0])
    with pytest.raises(ValueError, match="column"):
        MeasurementSamples(refs=samples.refs, outcomes=samples.outcomes[:, :-1])


def test_the_sampler_refuses_a_circuit_or_noise_record_of_the_wrong_type() -> None:
    memory = _surface(3, 1)
    with pytest.raises(TypeError, match="MemoryCircuit"):
        sample_memory_measurements(
            "not a circuit",  # type: ignore[arg-type]
            noise=_noise(),
            shots=4,
        )
    with pytest.raises(TypeError, match="PhenomenologicalNoise"):
        sample_memory_measurements(
            memory,
            noise="not noise",  # type: ignore[arg-type]
            shots=4,
        )
    with pytest.raises(ValueError, match="positive"):
        sample_memory_measurements(memory, noise=_noise(), shots=0)
    with pytest.raises(TypeError, match="shots must be a positive integer"):
        sample_memory_measurements(
            memory,
            noise=_noise(),
            shots=2.5,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="seed must be an integer or None"):
        sample_memory_measurements(
            memory,
            noise=_noise(),
            shots=4,
            seed=1.5,  # type: ignore[arg-type]
        )


def test_a_noiseless_run_pins_parities_rather_than_bits() -> None:
    """The handle layer is neither stronger nor weaker than the layout layer.

    Under the all-zero preparation a Z-type check's ancilla is deterministic and
    reads 0 in every round, because a Z-type check leaves that state alone. An
    X-type check's ancilla is not deterministic and reads unbiased, which is why
    no detector may name its round-zero bit. An individual terminal data wire is
    not deterministic either -- the X-type gadgets entangle the data with their
    ancillas -- and yet every detector and every observable reads 0, because the
    layouts name parities of handles rather than handles.

    So the two layers answer different questions and neither subsumes the other:
    the layout layer gives the deterministic parities a decoder consumes, and the
    handle layer gives every recorded bit, including the free ones. A test that
    read a rate off a single handle and called it a channel rate would be reading
    the state, which is the mistake this file's split-by-determinism avoids.
    """

    memory = _surface(3, 2)
    code = RotatedSurfaceCode(distance=3)
    x_type = {check.ancilla_wire for check in code.checks if check.stabilizer.x_wires}
    samples = sample_memory_measurements(
        memory, noise=PhenomenologicalNoise(), shots=512, seed=41
    )
    pinned = 0
    free = 0
    for reference in memory.measurement_refs:
        rate = float(samples.outcome(reference).to(torch.float64).mean())
        if reference.round_index is not None and reference.wire in x_type:
            assert 0.4 < rate < 0.6, reference
            free += 1
        elif reference.round_index is not None:
            assert rate == 0.0, reference
            pinned += 1
        else:
            assert 0.4 < rate < 0.6, reference
    assert pinned == 2 * (len(code.checks) - len(x_type))
    assert free == 2 * len(x_type)
    events = sample_memory_circuit(
        memory, noise=PhenomenologicalNoise(), shots=512, seed=41
    )
    assert int(events.detectors.sum()) == 0
    assert int(events.observables.sum()) == 0


def test_a_measurement_flip_fires_a_deterministic_handle_at_its_stated_rate() -> None:
    """A rate the record states is the rate the bit it names fires at.

    The handles are split by what they measure rather than pooled, because only
    a handle whose noiseless outcome is pinned can carry a rate: flipping an
    unbiased bit leaves it unbiased, so an X-type check's handle reads near one
    half at every rate. That is a statement about the state, not about the
    channel, and pooling the two kinds would average it away.
    """

    memory = _surface(3, 2)
    code = RotatedSurfaceCode(distance=3)
    x_type = {check.ancilla_wire for check in code.checks if check.stabilizer.x_wires}
    samples = sample_memory_measurements(
        memory,
        noise=PhenomenologicalNoise(measurement_flip=0.25),
        shots=2000,
        seed=43,
    )
    pinned = [
        reference
        for reference in memory.measurement_refs
        if reference.round_index is not None and reference.wire not in x_type
    ]
    free = [
        reference
        for reference in memory.measurement_refs
        if reference.round_index is not None and reference.wire in x_type
    ]
    assert pinned and free
    observed = sum(int(samples.outcome(reference).sum()) for reference in pinned) / (
        len(pinned) * samples.shots
    )
    assert abs(observed - 0.25) < 0.03
    unbiased = sum(int(samples.outcome(reference).sum()) for reference in free) / (
        len(free) * samples.shots
    )
    assert abs(unbiased - 0.5) < 0.03
    # The same rate reaches the detection events, through the handles they do
    # name: a channel and the bits it fires cannot be read as two stories.
    events = sample_memory_circuit(
        memory,
        noise=PhenomenologicalNoise(measurement_flip=0.25),
        shots=512,
        seed=43,
    )
    noiseless = sample_memory_circuit(
        memory, noise=PhenomenologicalNoise(), shots=512, seed=43
    )
    assert float(noiseless.detectors.any(dim=1).to(torch.float64).mean()) == 0.0
    assert float(events.detectors.any(dim=1).to(torch.float64).mean()) > 0.5
    # A logical flip is not part of the claim: it needs a whole logical string to
    # be wrong, so at this rate and shot count the right assertion is that the
    # observable is a well-formed reading of the handles, which the parity test
    # above pins, and not that it fires.
