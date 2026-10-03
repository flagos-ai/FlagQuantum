"""The three Pauli data families, on the model route and on the sampler route.

The noise record states four families -- an X, a Z and a Y fault on a data wire
and a flip on a check's readout -- and upstream's ``CssNoise`` states the same
four as ``px``, ``pz``, ``py`` and ``pm``. The matrix route always carried all
three Pauli faults, because it reads a parity matrix and a support rather than
executing a program. The circuit route and the sampler used to carry the X fault
alone, so a record stating a Z or Y rate built a model that silently ignored it
and sampled a run in which it never happened.

These tests state what each route now does with the three families and pin the
two to each other. The signatures are forced through the circuit by the
construction route's own injector and read off the layouts, so the sampler is
compared against the model's answer and never against a second opinion about what
the Pauli should be; the sampler's own placement is asserted separately, because
a rate and a signature can agree while the channel placed is the wrong Pauli for
a different reason.

Two of the tests execute the sampler, which needs the optional stabilizer engine,
so those two skip when `stim` is absent. The enumeration and placement tests above
them read the model and the lowered program rather than running them, so they keep
running in a lane that installs no optional backend.
"""

from __future__ import annotations

import functools
import math
from collections import Counter

import pytest
import torch

from flagquantum.qec import (
    DetectorErrorModel,
    MemoryCircuit,
    PhenomenologicalNoise,
    RepetitionCode,
    RotatedSurfaceCode,
    build_memory_circuit,
    sample_memory_circuit,
)
from flagquantum.qec.dem_construction import (
    _forced_signature,
    _inject_data_flip,
    _mechanisms,
)
from flagquantum.qec.sampling import (
    _measurement_plan,
    _noise_locations,
    _noisy_program,
)

pytestmark = pytest.mark.integration

# The codes the two routes are compared on. A repetition code has Z-type checks
# alone, so a Z fault is invisible to it in every round; a rotated surface code
# has both check types, so a Z fault is invisible in the round it is placed in and
# visible in the round after it. Both shapes are needed: the first states that a
# Z fault is not invented where nothing can see it, the second that it is not
# dropped where something can.
_CASES: dict[str, tuple[object, int]] = {
    "repetition-d3": (RepetitionCode(distance=3), 2),
    "surface-d3": (RotatedSurfaceCode(distance=3), 2),
}

_SEED = 11
_SHOTS = 2000

# A sampled per-detector rate is accepted when it sits within this many binomial
# standard errors of the modelled rate. Five covers a miss probability of about
# 6e-7 per detector, which is what keeps a sweep over sixteen detectors from
# failing on its own sample while a wrong rate -- the whole family dropped, or a
# Z fault sampled as an X one -- stays far outside.
_RATE_TOLERANCE_SIGMAS = 5.0


@functools.lru_cache(maxsize=None)
def _memory(case: str) -> MemoryCircuit:
    """Return the case's memory circuit, built once for the whole module."""

    code, rounds = _CASES[case]
    return build_memory_circuit(code, rounds=rounds)  # type: ignore[arg-type]


@functools.lru_cache(maxsize=None)
def _round_signature(
    case: str, family: str, wire: int, round_index: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the flips one family's fault forces in one round on one wire.

    The fault is injected by the construction route's injector and read by its
    signature reader, so this is the model's own answer for that location and not
    a re-derivation of it.
    """

    memory = _memory(case)
    return _forced_signature(
        memory,
        _inject_data_flip(memory, round_index=round_index, wire=wire, kind=family),
    )


def _xor(first: tuple[int, ...], second: tuple[int, ...]) -> tuple[int, ...]:
    """Return the symmetric difference of two flip sets."""

    return tuple(sorted(set(first) ^ set(second)))


def _family_flips(case: str, family: str, wire: int) -> tuple[tuple[int, ...], ...]:
    """Return the flips every round's fault of one family composes to.

    Pauli faults compose by XOR on the flip sets, which is the same rule
    ``_merge_mechanisms`` rests on, so the fault the sampler draws once per round
    flips the XOR of the per-round signatures.
    """

    detectors: tuple[int, ...] = ()
    observables: tuple[int, ...] = ()
    for round_index in range(_memory(case).rounds):
        forced = _round_signature(case, family, wire, round_index)
        detectors = _xor(detectors, forced[0])
        observables = _xor(observables, forced[1])
    return detectors, observables


def _one_wire_vector(size: int, position: int) -> tuple[float, ...]:
    """Return a per-qubit vector that is certain at ``position`` and zero elsewhere."""

    return tuple(1.0 if index == position else 0.0 for index in range(size))


def _flipped(tensor: torch.Tensor) -> tuple[int, ...]:
    """Return the indices a one-shot detection-event or observable row flips."""

    return tuple(int(index) for index in tensor.nonzero().flatten())


def test_each_family_is_a_location_of_its_own() -> None:
    """Three families on three wires over two rounds, plus two checks' readouts.

    One wire can carry three mechanisms in one round, one per family, and each
    carries its own family's rate: the record states the families separately, so
    a wire that is noisy in one Pauli and quiet in the others is enumerated once.
    """

    memory = _memory("repetition-d3")
    noise = PhenomenologicalNoise(
        data_flip=0.01,
        phase_flip=0.02,
        both_flip=0.03,
        measurement_flip=0.04,
    )

    counts = Counter(mechanism.kind for mechanism in _mechanisms(memory, noise))
    assert counts == {"data": 6, "phase": 6, "both": 6, "measurement": 4}

    rates = {
        (mechanism.kind, mechanism.round_index, mechanism.wire): mechanism.probability
        for mechanism in _mechanisms(memory, noise)
    }
    assert rates[("data", 0, 0)] == 0.01
    assert rates[("phase", 0, 0)] == 0.02
    assert rates[("both", 0, 0)] == 0.03
    # A measurement mechanism names the ancilla whose readout it corrupts, so the
    # wire is the check's own rather than a data wire.
    assert rates[("measurement", 0, 3)] == 0.04


def test_a_family_at_zero_rate_is_not_a_location() -> None:
    """A family nobody raised costs no mechanism and no channel.

    The rate is the family's own, so a profile that raises the X family alone is
    not three locations per wire at three rates -- two of which would be zero and
    would have to be dropped later anyway.
    """

    memory = _memory("repetition-d3")
    kinds = Counter(
        mechanism.kind
        for mechanism in _mechanisms(memory, PhenomenologicalNoise(data_flip=0.05))
    )
    assert set(kinds) == {"data"}

    assert _mechanisms(memory, PhenomenologicalNoise()) == ()
    assert (
        _mechanisms(memory, PhenomenologicalNoise(phase_flip=0.0, both_flip=0.0)) == ()
    )


@pytest.mark.parametrize("case", ["repetition-d3", "surface-d3"])
def test_the_three_families_compose_as_the_paulis_do(case: str) -> None:
    """A Y fault flips what the X fault and the Z fault flip, and nothing else.

    X and Z anticommute into Y, and a Y flips exactly the detectors an X or a Z
    at the same place and round flips. The two partial faults therefore flip
    disjoint detector sets -- an X fault is invisible to the X-type checks and a Z
    fault to the Z-type ones -- and the Y fault flips their union. This holds
    location by location, which is what makes the Y family one mechanism at one
    rate rather than two.
    """

    code, _ = _CASES[case]
    for wire in code.data_wires:  # type: ignore[attr-defined]
        for round_index in range(_memory(case).rounds):
            x_detectors = set(_round_signature(case, "data", wire, round_index)[0])
            z_detectors = set(_round_signature(case, "phase", wire, round_index)[0])
            y_detectors = set(_round_signature(case, "both", wire, round_index)[0])
            assert not (x_detectors & z_detectors)
            assert y_detectors == x_detectors | z_detectors

            x_observables = set(_round_signature(case, "data", wire, round_index)[1])
            z_observables = set(_round_signature(case, "phase", wire, round_index)[1])
            y_observables = set(_round_signature(case, "both", wire, round_index)[1])
            assert not (x_observables & z_observables)
            assert y_observables == x_observables | z_observables


@pytest.mark.parametrize("case", ["repetition-d3", "surface-d3"])
def test_a_z_fault_reaches_only_the_checks_that_can_see_it(case: str) -> None:
    """A Z fault is recorded where an X-type check compares two syndromes.

    The circuit route is a Z-memory experiment: its logical observable is read in
    the Z basis, so a Z fault never flips it, and its round-zero detectors belong
    to the Z-type checks, so a Z fault placed at the round-zero boundary changes
    both syndromes the round-zero band is built from and is no mechanism at all.
    On a repetition code, whose checks are all Z-type, no round can see one.
    """

    code, _ = _CASES[case]
    for wire in code.data_wires:  # type: ignore[attr-defined]
        assert _family_flips(case, "phase", wire)[1] == ()
        assert _round_signature(case, "phase", wire, 0) == ((), ())

    later = {
        wire: _family_flips(case, "phase", wire)[0]
        for wire in code.data_wires  # type: ignore[attr-defined]
    }
    if case == "repetition-d3":
        assert all(detectors == () for detectors in later.values())
    else:
        assert any(detectors for detectors in later.values())


def test_the_sampler_places_each_family_as_one_conjugated_channel() -> None:
    """The engine has one channel, so the family is stated by what wraps it.

    The bit flip is placed at the round boundary for every family; the Z and Y
    families add the conjugation that turns that one draw into the Pauli they
    name. Nothing else changes, which is asserted by the instruction count: the
    noiseless program plus exactly the wrappers and the channels. One round is
    used so that one raised location is one location, and the family's rate is
    stated through its per-element vector so the location raised is on the wire
    the assertion names.
    """

    memory = build_memory_circuit(RepetitionCode(distance=3), rounds=1)
    plan = _measurement_plan(memory)
    wrappers = {
        "data": ("", ""),
        "phase": ("h", "h"),
        "both": ("sdg", "s"),
        "measurement": ("", ""),
    }
    for family, (before, after) in wrappers.items():
        if family == "measurement":
            # A measurement mechanism names an ancilla, and one round's first
            # check owns ancilla 3 on this code.
            profile = {
                "measurement_flip_per_check": _one_wire_vector(
                    len(memory.code.checks), 0
                )
            }
            wire = 3
        else:
            profile = {
                f"{family}_flip_per_qubit": _one_wire_vector(
                    len(memory.code.data_wires), 0
                )
            }
            wire = 0
        locations = _noise_locations(memory, plan, PhenomenologicalNoise(**profile))
        assert [location.kind for location in locations] == [family]
        assert locations[0].wire == wire

        program = _noisy_program(plan, locations)
        placed = [name for name in (before, "bit_flip", after) if name]
        channels = [
            index
            for index, instruction in enumerate(program.instructions)
            if instruction.name == "bit_flip"
        ]
        assert len(channels) == 1
        channel = channels[0]
        assert len(program.instructions) == len(plan.program.instructions) + len(placed)

        # The channel sits at the location's own instruction, with the family's
        # wrappers around it and nothing else inserted, and it is placed *before*
        # that instruction rather than after it.
        first = channel - placed.index("bit_flip")
        assert [
            instruction.name
            for instruction in program.instructions[first : first + len(placed)]
        ] == placed
        assert all(
            instruction.wires == (wire,)
            for instruction in program.instructions[first : first + len(placed)]
        )
        assert (
            program.instructions[first + len(placed)]
            is plan.program.instructions[locations[0].instruction_index]
        )


@pytest.mark.parametrize("case", ["repetition-d3", "surface-d3"])
def test_a_certain_family_fault_is_exactly_what_the_sampler_draws(case: str) -> None:
    """A family fault at probability one is the model's signature, shot for shot.

    The record is raised for one wire alone through the per-qubit vector, so every
    location the sampler places is that wire's fault in one round at probability
    one: the run is deterministic, and its detection events and observable flips
    are compared against the XOR of the signatures the construction route forces
    for the same family, wire and rounds. This is the join the row is about, and
    it holds for all three families on a code with both check types.
    """

    pytest.importorskip("stim")

    memory = _memory(case)
    code, _ = _CASES[case]
    data_wires = code.data_wires  # type: ignore[attr-defined]
    for family in ("data", "phase", "both"):
        for position, wire in enumerate(data_wires):
            expected = _family_flips(case, family, wire)
            noise = PhenomenologicalNoise(
                **{
                    f"{family}_flip_per_qubit": _one_wire_vector(
                        len(data_wires), position
                    )
                }
            )
            sample = sample_memory_circuit(memory, noise=noise, shots=2, seed=_SEED)
            assert _flipped(sample.detectors[0]) == expected[0]
            assert _flipped(sample.observables[0]) == expected[1]
            assert torch.equal(sample.detectors[0], sample.detectors[1])
            assert torch.equal(sample.observables[0], sample.observables[1])


def test_a_sampled_family_profile_agrees_with_the_model() -> None:
    """The rate a family is sampled at is the rate the model states for it.

    Only the two families the sampler could not place before are raised, so a run
    that still dropped them would read all zeros against a model that does not.
    The comparison is per detector and per observable at five binomial standard
    errors, which is narrow enough to catch a dropped family and wide enough not
    to fail on the sample itself.
    """

    memory = _memory("surface-d3")
    noise = PhenomenologicalNoise(phase_flip=0.05, both_flip=0.05)
    model = DetectorErrorModel.from_memory_circuit(memory, noise=noise)
    assert model.num_errors > 0

    pytest.importorskip("stim")

    sample = sample_memory_circuit(memory, noise=noise, shots=_SHOTS, seed=_SEED)
    measured = (
        (
            sample.detectors.float().mean(dim=0),
            model.detector_rates(),
            len(memory.detectors),
        ),
        (
            sample.observables.float().mean(dim=0),
            model.observable_rates(),
            len(memory.observables),
        ),
    )
    for sampled, predicted, count in measured:
        assert sampled.shape == (count,)
        for index in range(count):
            rate = float(sampled[index])
            modelled = float(predicted[index])
            tolerance = _RATE_TOLERANCE_SIGMAS * math.sqrt(
                modelled * (1.0 - modelled) / _SHOTS
            )
            assert abs(rate - modelled) <= tolerance, (
                f"index {index} sampled {rate}, which is outside {tolerance} of "
                f"the modelled {modelled}"
            )
