"""The gate-bound noise grammar of a memory experiment.

A memory experiment can be described by two noise records, and the record the
caller states is what selects the grammar. `PhenomenologicalNoise` names round
boundaries and check readouts; `flagquantum.noise.NoiseModel` names gates, and
its fault sits immediately after the gate its rule matched, which is upstream
CUDA-Q's placement for a channel bound to a named gate. One function accepts both
records -- `DetectorErrorModel.from_memory_circuit`, `sample_memory_measurements`,
and `sample_memory_circuit` -- and neither record is a second entry point.

The evidence here has three parts. The first is structural: a rule on a gate the
program does not carry places nothing, a rule on a repeated gate places one fault
per round rather than one at a boundary, and the two routes enumerate one
location list from one lowering. The second is exact: at probability one every
shot is compared against the detection events the model built from the same
placement, which fails on a fault placed one instruction away from the right one
instead of averaging the mistake into a rate. The third is a refusal table: a
channel this grammar cannot place, a rule naming a readout or a preparation, a
one-qubit channel bound to two wires, and a fault that would follow the program's
end are each refused by name rather than approximated.
"""

from __future__ import annotations

import pytest
import torch

import flagquantum.noise as fqn
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.errors import CapabilityError
from flagquantum.qec import (
    DetectorErrorModel,
    PhenomenologicalNoise,
    RepetitionCode,
    RotatedSurfaceCode,
    build_memory_circuit,
    sample_memory_circuit,
    sample_memory_measurements,
)
from flagquantum.qec.dem_construction import (
    _gate_bound_mechanisms,
    _lowered_program,
    _spliced_fault,
)
from flagquantum.qec.sampling import _measurement_plan, _noise_locations

# Sampling executes the program through the stabilizer engine, which is an
# optional distribution, so the file skips rather than failing to import when it
# is absent.
pytest.importorskip("stim")

pytestmark = pytest.mark.integration

# A strength high enough that a placed fault is visible well above the sampling
# floor at this shot count, and low enough that two faults on one shot stay rare
# enough for the per-detector marginal to be the model's first-order rate.
_SHOTS = 100_000
_SIGMA = 4.0
_PROBABILITY = 0.02
_SEED = 13

_CODES = (
    pytest.param(RepetitionCode(distance=3), 3, id="repetition-d3-r3"),
    pytest.param(RotatedSurfaceCode(distance=3), 3, id="surface-d3-r3"),
)

_PAULI_CHANNELS = (
    pytest.param(fqn.bit_flip_channel, id="bit-flip"),
    pytest.param(fqn.phase_flip_channel, id="phase-flip"),
)


def _memory(code, rounds):
    return build_memory_circuit(code, rounds=rounds)


def _bit_flip(gate, probability=_PROBABILITY):
    return fqn.NoiseModel().add(gate, fqn.bit_flip_channel(probability))


# --------------------------------------------------------------------------- #
# Placement structure
# --------------------------------------------------------------------------- #


def test_a_rule_on_an_absent_gate_places_nothing() -> None:
    """A record stated over a gate set contributes nothing for a gate outside it.

    The model describes the experiment's noise, so a rule whose gate the program
    does not carry has no location. Refusing would make a code-independent record
    unusable on any code that omits one gate, and reading it as a fallback would
    place a fault at a gate the caller never named.
    """

    memory = _memory(RotatedSurfaceCode(distance=3), 2)
    absent = fqn.NoiseModel().add("swap", fqn.bit_flip_channel(0.5))
    assert _gate_bound_mechanisms(absent, _lowered_program(memory)) == ()
    model = DetectorErrorModel.from_memory_circuit(memory, noise=absent)
    assert model.num_detectors == len(memory.detectors.detectors)
    assert model.errors == ()
    sample = sample_memory_circuit(memory, noise=absent, shots=64, seed=0)
    assert not bool(sample.detectors.any())


def test_a_rule_places_one_fault_per_round_not_one_at_a_boundary() -> None:
    """A named gate is one round-relative offset, so a rule on it acts in every round.

    This is the difference between the two grammars: a `PhenomenologicalNoise`
    field is a round boundary and appears once, and a gate name appears once per
    round. The lowered program's block length is what converts the ruled gate's
    instruction index into the round it is in, and the placement is pinned to the
    matched gate rather than to a round boundary.
    """

    memory = _memory(RotatedSurfaceCode(distance=3), 3)
    plan = _measurement_plan(memory)
    gates = [
        index
        for index, instruction in enumerate(plan.program.instructions)
        if instruction.name == "h"
    ]
    locations = _noise_locations(memory, plan, _bit_flip("h"))
    assert len(locations) == len(gates)
    assert len(locations) % memory.rounds == 0
    for location in locations:
        matched = plan.program.instructions[location.instruction_index - 1]
        assert matched.name == "h"
        assert matched.wires == (location.wire,)
        assert location.round_index == location.instruction_index // plan.block
        assert location.kind == "data"
    for round_index in range(memory.rounds):
        assert (
            sum(location.round_index == round_index for location in locations)
            == len(gates) // memory.rounds
        )


def test_the_two_routes_enumerate_one_location_list() -> None:
    """The model and the sampler place a gate-bound fault at one and the same index.

    The instruction the model splices after and the instruction the sampler
    places before are both read from `_gate_bound_mechanisms`, so the two cannot
    drift: a fault read at one instruction by the model and placed at another by
    the sampler would be a sampled model that describes a different experiment.
    Every channel the record is allowed to place is a single-qubit Pauli fault,
    which is why the sampler's conjugation can express all of them.
    """

    memory = _memory(RotatedSurfaceCode(distance=3), 2)
    plan = _measurement_plan(memory)
    noise = _bit_flip("cx")
    mechanisms = _gate_bound_mechanisms(noise, plan.program)
    locations = _noise_locations(memory, plan, noise)
    assert len(mechanisms) == len(locations) > 0
    for mechanism, location in zip(mechanisms, locations, strict=True):
        assert mechanism.after_instruction + 1 == location.instruction_index
        assert mechanism.wire == location.wire
        assert mechanism.probability == location.probability
        assert mechanism.kind == location.kind


def test_a_spliced_fault_inserts_one_sequence_and_moves_nothing() -> None:
    """The fault is written after its gate and the rest of the program is untouched.

    A splice that lost or reordered an instruction would move every later
    readout's recorded column, which the model reads through the layouts and the
    sampler reads through the same plan. A phase fault is three instructions and a
    bit fault is one, and both are exact identities when they do not fire.
    """

    memory = _memory(RotatedSurfaceCode(distance=3), 2)
    program = _lowered_program(memory)
    bit = _gate_bound_mechanisms(_bit_flip("cx"), program)
    phase = _gate_bound_mechanisms(
        fqn.NoiseModel().add("cx", fqn.phase_flip_channel(_PROBABILITY)), program
    )
    assert bit and phase
    assert bit[0].after_instruction == phase[0].after_instruction
    assert bit[0].wire == phase[0].wire

    for mechanism in bit:
        spliced = _spliced_fault(program, mechanism)
        position = mechanism.after_instruction + 1
        assert spliced.instructions[:position] == program.instructions[:position]
        assert spliced.instructions[position].name == "x"
        assert spliced.instructions[position].wires == (mechanism.wire,)
        assert spliced.instructions[position + 1 :] == program.instructions[position:]
        assert spliced.n_wires == program.n_wires
        assert spliced.metadata == program.metadata

    spliced = _spliced_fault(program, phase[0])
    position = phase[0].after_instruction + 1
    assert [item.name for item in spliced.instructions[position : position + 3]] == [
        "h",
        "x",
        "h",
    ]


@pytest.mark.parametrize(("code", "rounds"), _CODES)
@pytest.mark.parametrize("factory", _PAULI_CHANNELS)
def test_the_sampler_reproduces_the_models_own_arithmetic(
    code, rounds, factory
) -> None:
    """At probability one the circuit simulator is the model's own sampling.

    Every location fires on every shot, so the model's `dem_sampling` is the XOR
    of all its signatures and is the same for every seed, and the sampler's shot
    must be that same detection event. The comparison is exact rather than
    statistical, so a fault placed one instruction away from the right one, or
    read against the wrong wire, changes a detector instead of shifting a rate
    inside the sampling floor. Both directions of the join are covered: the model
    derives its signature from the source and the sampler from the lowered
    program, and a disagreement between the two is what this equality detects.
    """

    memory = _memory(code, rounds)
    noise = fqn.NoiseModel().add("cx", factory(1.0))
    model = DetectorErrorModel.from_memory_circuit(memory, noise=noise)
    if not model.errors:
        pytest.skip("this code carries no cx fault this channel can place")
    reference = model.dem_sampling(shots=4, seed=0)
    assert torch.equal(
        reference.detectors, model.dem_sampling(shots=4, seed=99).detectors
    )
    sample = sample_memory_circuit(memory, noise=noise, shots=4, seed=_SEED)
    assert torch.equal(
        sample.detectors.to(torch.int64), reference.detectors.to(torch.int64)
    )
    assert torch.equal(
        sample.observables.to(torch.int64), reference.observables.to(torch.int64)
    )
    if isinstance(code, RotatedSurfaceCode) and factory is fqn.bit_flip_channel:
        # An X fault after a CX on this code reaches detectors, so the equality
        # above is not the empty signature agreeing with itself.
        assert bool(sample.detectors.any())


def test_a_bit_flip_and_a_phase_flip_are_two_different_faults() -> None:
    """A phase flip is a Z fault, so binding it to a gate is not binding a bit flip.

    Both channels are read off their own Kraus pair, and the family each names
    decides which Pauli the fault is. Bound to a two-qubit gate the two are
    plainly different faults -- an X fault on a Z-type check's CNOT control
    reaches its ancilla and the X fault on the target does not -- so a grammar
    that read the channel's name without its operator would place one for the
    other, and the two models would have the same detectors.
    """

    memory = _memory(RotatedSurfaceCode(distance=3), 3)
    bit = DetectorErrorModel.from_memory_circuit(memory, noise=_bit_flip("cx"))
    phase = DetectorErrorModel.from_memory_circuit(
        memory,
        noise=fqn.NoiseModel().add("cx", fqn.phase_flip_channel(_PROBABILITY)),
    )
    assert bit.errors and phase.errors
    assert bit.num_detectors == phase.num_detectors
    assert not torch.allclose(bit.detector_rates(), phase.detector_rates())


# --------------------------------------------------------------------------- #
# Statistical agreement with the model
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("code", "rounds"), _CODES)
@pytest.mark.parametrize("factory", _PAULI_CHANNELS)
def test_the_sampled_detector_rates_are_the_models(code, rounds, factory) -> None:
    """A physical rate is compared against the rate the same model predicts.

    The comparison is per detector and at `_SIGMA` standard deviations of the
    sampled mean. It catches a fault placed against the wrong gate or in the
    wrong round: both change which detectors a mechanism reaches, so the marginal
    rates of the detectors it moved away from fall below their prediction by far
    more than the sampling floor.
    """

    memory = _memory(code, rounds)
    noise = fqn.NoiseModel().add("h", factory(_PROBABILITY))
    model = DetectorErrorModel.from_memory_circuit(memory, noise=noise)
    sample = sample_memory_circuit(memory, noise=noise, shots=_SHOTS, seed=_SEED)
    predicted = model.detector_rates().to(torch.float64)
    observed = sample.detectors.to(torch.float64).mean(dim=0)
    tolerance = (
        _SIGMA * torch.sqrt(predicted * (1.0 - predicted) / _SHOTS) + 1.0 / _SHOTS
    )
    deviation = (observed - predicted).abs()
    assert bool((deviation <= tolerance).all()), (
        f"detector rate deviation {float(deviation.max()):.6f} exceeds "
        f"{float(tolerance.max()):.6f}; predicted {predicted.tolist()}, "
        f"observed {observed.tolist()}"
    )


def test_the_measurement_route_reads_the_same_run() -> None:
    """Reading a run by handle and reading it as detection events cannot disagree.

    Both readings take one lowering, one location list and one engine record, so
    a detector's parity is the parity of the handles it names. Widening the
    accepted record on one reading and not the other would leave the two routes
    describable by different experiments.
    """

    memory = _memory(RotatedSurfaceCode(distance=3), 2)
    noise = fqn.NoiseModel().add("cx", fqn.bit_flip_channel(0.5))
    samples = sample_memory_measurements(memory, noise=noise, shots=32, seed=_SEED)
    sample = sample_memory_circuit(memory, noise=noise, shots=32, seed=_SEED)
    assert samples.outcomes.shape[0] == sample.detectors.shape[0] == 32
    assert samples.outcomes.shape[1] == len(memory.measurement_refs)


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "channel",
    (
        pytest.param(fqn.depolarizing_channel(0.1), id="depolarizing"),
        pytest.param(fqn.amplitude_damping_channel(0.1), id="amplitude-damping"),
        pytest.param(fqn.phase_damping_channel(0.1), id="phase-damping"),
    ),
)
def test_a_channel_that_is_not_a_pauli_fault_is_refused_by_name(channel) -> None:
    """A mixture is not one Pauli fault, so it has no placement in this grammar.

    Placing one of a depolarizing channel's terms as the whole fault would state
    a fault the record does not: its rate would be a fraction of the channel's and
    its Pauli would be whichever term happened to come first. The refusal names
    the channel and the two it does place, and both routes state it, because a
    record one route can place and the other cannot is a sampler and a model that
    describe different experiments.
    """

    memory = _memory(RotatedSurfaceCode(distance=3), 2)
    noise = fqn.NoiseModel().add("h", channel)
    with pytest.raises(CapabilityError) as raised:
        DetectorErrorModel.from_memory_circuit(memory, noise=noise)
    assert channel.name in str(raised.value)
    assert "bit_flip" in str(raised.value) and "phase_flip" in str(raised.value)
    with pytest.raises(CapabilityError, match=channel.name):
        sample_memory_circuit(memory, noise=noise, shots=4)
    with pytest.raises(CapabilityError, match=channel.name):
        sample_memory_measurements(memory, noise=noise, shots=4)


@pytest.mark.parametrize("gate", ("measure", "reset"))
def test_a_rule_on_a_readout_or_a_preparation_is_refused(gate: str) -> None:
    """This grammar attaches a fault to a gate, and a readout is not one.

    A readout fault's position is the check it corrupts, which is what the
    measurement family of a `PhenomenologicalNoise` states. Accepting a rule on
    `measure` would place a fault after every readout, terminal data readouts
    included, which is not the experiment the record describes; the refusal names
    the record that can state it and does not depend on the program carrying the
    name.
    """

    memory = _memory(RotatedSurfaceCode(distance=3), 2)
    noise = fqn.NoiseModel().add(gate, fqn.bit_flip_channel(0.1))
    with pytest.raises(CapabilityError) as raised:
        DetectorErrorModel.from_memory_circuit(memory, noise=noise)
    assert gate in str(raised.value)
    assert "PhenomenologicalNoise" in str(raised.value)
    with pytest.raises(CapabilityError, match=gate):
        sample_memory_circuit(memory, noise=noise, shots=4)


def test_a_rule_whose_channel_cannot_be_inferred_reaches_the_caller() -> None:
    """The record's own refusal is reported as it states itself.

    A two-qubit channel on a one-qubit gate is the record's inconsistency, not
    this grammar's, so it is the record's error that reaches the caller. The
    grammar adds no second refusal for it, because a second message for one
    failure is a second statement of what is wrong.
    """

    memory = _memory(RotatedSurfaceCode(distance=3), 2)
    noise = fqn.NoiseModel().add("h", fqn.two_qubit_depolarizing_channel(0.1))
    with pytest.raises(ValueError, match="two_qubit_depolarizing"):
        DetectorErrorModel.from_memory_circuit(memory, noise=noise)
    with pytest.raises(ValueError, match="two_qubit_depolarizing"):
        sample_memory_circuit(memory, noise=noise, shots=4)


def test_a_one_qubit_channel_bound_to_two_wires_is_refused() -> None:
    """A one-qubit fault on two explicitly named wires is a fault this grammar does not place.

    The record itself permits the rule, because it requires only that the named
    wires be a subset of the matched instruction's, so the refusal is this
    grammar's and it names the rule's gate. Placing one fault on each wire instead
    would be two rules for one, at twice the rate the record states.
    """

    memory = _memory(RotatedSurfaceCode(distance=3), 2)
    noise = fqn.NoiseModel().add("cx", fqn.bit_flip_channel(0.1), qubits=(3, 9))
    with pytest.raises(CapabilityError) as raised:
        DetectorErrorModel.from_memory_circuit(memory, noise=noise)
    assert "bit_flip" in str(raised.value) and "2 qubits" in str(raised.value)
    with pytest.raises(CapabilityError, match="bit_flip"):
        sample_memory_circuit(memory, noise=noise, shots=4)


def test_a_rule_that_matches_the_programs_last_instruction_is_refused() -> None:
    """A fault that would follow the program's end is stated rather than dropped.

    A dropped location is a silently thinner model, so the refusal is explicit.
    The emitted memory program ends with a data readout rather than a gate, so the
    guard is exercised against a program that does end with one: a hand-built
    program is the only way to reach it, and the guard is what keeps a location at
    one past the end from being discarded by the placement loop.
    """

    memory = _memory(RepetitionCode(distance=3), 1)
    program = _lowered_program(memory)
    assert program.instructions[-1].name == "reset"
    truncated = CircuitIR(
        n_wires=program.n_wires,
        instructions=(*program.instructions[:-1], Instruction(name="x", wires=(0,))),
        metadata=dict(program.metadata),
    )
    with pytest.raises(CapabilityError) as raised:
        _gate_bound_mechanisms(_bit_flip("x"), truncated)
    assert "last instruction" in str(raised.value)


def test_an_unknown_noise_record_is_refused_by_both_routes() -> None:
    """The two records are the union, and a third type is refused rather than guessed.

    Failing closed here is what makes the parameter one union rather than a family
    that grows by `isinstance` fallthrough, and the message names both records so
    the caller can see what the union is.
    """

    memory = _memory(RotatedSurfaceCode(distance=3), 2)
    for noise in (None, 1, "noise", object()):
        with pytest.raises(TypeError) as raised:
            DetectorErrorModel.from_memory_circuit(memory, noise=noise)
        assert "PhenomenologicalNoise" in str(raised.value)
        assert "NoiseModel" in str(raised.value)
        with pytest.raises(TypeError):
            sample_memory_circuit(memory, noise=noise, shots=4)


def test_the_round_boundary_grammar_is_unchanged() -> None:
    """A `PhenomenologicalNoise` places its faults exactly where it always did.

    The union accepts a second record; it does not move the first one's
    placements. A round-boundary record carries no gate name, so widening the
    parameter cannot change where its data faults open a round or where its
    readout faults sit, and the gate-bound route enumerates only the data family
    for a Pauli channel.
    """

    memory = _memory(RotatedSurfaceCode(distance=3), 2)
    plan = _measurement_plan(memory)
    boundaries = _noise_locations(
        memory, plan, PhenomenologicalNoise(data_flip=0.01, measurement_flip=0.01)
    )
    assert boundaries
    for location in boundaries:
        if location.kind == "measurement":
            assert plan.program.instructions[location.instruction_index].name == (
                "measure"
            )
        else:
            assert location.instruction_index == location.round_index * plan.block
    gate_bound = _noise_locations(memory, plan, _bit_flip("cx"))
    assert gate_bound
    assert {location.kind for location in gate_bound} == {"data"}
    # A gate-bound data fault opens no round: it follows a gate, so it cannot sit
    # at the round boundary the round-boundary route places its data faults at.
    assert all(
        location.instruction_index != location.round_index * plan.block
        for location in gate_bound
    )
