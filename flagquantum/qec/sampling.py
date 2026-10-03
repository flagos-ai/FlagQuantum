"""Sampled detection events from a configured memory experiment.

A detector error model states what the noise does; a sampler states what a run
of the experiment reads out. This module supplies the second: it lowers the
memory circuit once, places each noise location the configured
`~flagquantum.qec.PhenomenologicalNoise` names at the instruction it belongs to,
executes the resulting circuit on the stabilizer engine, and reads the
detection-event and observable layouts off the recorded bits.

Where a location belongs is the whole content of the join. A data error is a
round-boundary error and is placed *before* the round's first gate, so it opens
the frame that round's detectors compare against; a measurement error is placed
immediately *before* the readout of the check it corrupts. Neither position
exists in the source program, whose bounded hybrid capture refuses a channel
call outright, so the placement is derived here from the lowered program rather
than written into the experiment.

Which Pauli a data error is is the other half of the join, and it is the part
the engine constrains. The engine executes one channel -- a single-wire bit
flip -- so the record's Z and Y families are placed as that channel conjugated
by the Clifford that turns the flip into the Pauli they name, which is one draw
at that family's rate rather than a pair of independent bit flips. A family the
conjugation cannot express has no route here and is refused rather than sampled
as a nearby Pauli.

The derived placement rests on one property of the emitted program, checked
rather than assumed: the syndrome loop lowers to `rounds` identical round
blocks, each measuring every check once in the code's declared check order. A
program that lowers to anything else is refused, because a placement derived
from it would attribute a mechanism to the wrong round.

The model and this sampler share the locations and nothing else. The model
derives each mechanism's signature from the source, this module derives each
mechanism's instruction position from the lowered program, and the tests pin the
two to each other; the sampled rates are then compared against the model's
predicted rates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType

import torch

from ..compiler._hybrid import INDEX, capture_source, lower_dynamic_program
from ..core.ir import CircuitIR, Instruction
from ..noise import bit_flip_channel
from ..simulation.stabilizer import sample_noisy_measurements
from .circuit import MeasurementRef, MemoryCircuit
from .dem import DemSample
from .dem_construction import (
    _check_rate_order,
    _data_family_rates,
    _MechanismKind,
)
from .noise import PhenomenologicalNoise

_MEASURE_OPCODE = "measure"
_NOISE_OPCODE = "bit_flip"

# The Clifford conjugation the engine's one channel needs around it to become the
# record's other data-fault families, as (before, after) opcodes.
# ``H X H = Z`` and ``S_DAG X S = iY``, so a single bit-flip draw conjugated this
# way *is* one Z or one Y fault rather than a pair of independent draws, which is
# what a fault with one rate has to be. The construction route reaches the same
# two Paulis by composing forced gates out of ``h`` and ``x``, because the source
# it injects into carries no other parameter-free single-qubit gate; here the
# conjugation is emitted directly and is the narrower statement of the same
# identity. A measurement fault is the bit flip itself -- a readout is flipped in
# the basis it is read in -- so its entry is empty on both sides.
_FAMILY_CONJUGATION: Mapping[_MechanismKind, tuple[str, str]] = MappingProxyType(
    {
        "data": ("", ""),
        "phase": ("h", "h"),
        "both": ("sdg", "s"),
        "measurement": ("", ""),
    }
)


@dataclass(frozen=True)
class _MeasurementPlan:
    """Where every detection-event reference sits in the engine's record.

    ``block`` is the number of instructions one syndrome round lowers to, so
    round ``r`` owns ``program.instructions[r * block : (r + 1) * block]``.
    ``measure_offsets`` is the template position of each check's readout, in the
    code's declared check order. ``syndrome_columns`` and ``terminal_columns``
    are engine record columns: the first indexes ``(round_index, ancilla_wire)``,
    the second a data wire.
    """

    program: CircuitIR
    block: int
    measure_offsets: tuple[int, ...]
    syndrome_columns: Mapping[tuple[int, int], int]
    terminal_columns: Mapping[int, int]


@dataclass(frozen=True)
class _NoiseLocation:
    """One noise mechanism, the instruction it is placed before, and its rate.

    The rate is carried rather than looked up at placement because a per-element
    vector gives every location its own: a location is built only where its own
    effective rate is nonzero, and the channel placed there is that rate's
    channel rather than the family's scalar.

    ``kind`` names the noise record's own field, as
    :class:`~flagquantum.qec.dem_construction._Mechanism` does, so the two routes
    enumerate the same families under the same names.
    """

    kind: _MechanismKind
    round_index: int
    wire: int
    instruction_index: int
    probability: float


def _checked_shots(shots: int) -> int:
    if isinstance(shots, bool) or not isinstance(shots, Integral):
        raise TypeError("shots must be a positive integer")
    if shots <= 0:
        raise ValueError("shots must be a positive integer")
    return int(shots)


def _checked_seed(seed: int | None) -> int | None:
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral)):
        raise TypeError("seed must be an integer or None")
    return None if seed is None else int(seed)


def _lowered_program(memory: MemoryCircuit) -> CircuitIR:
    """Return the memory program lowered the way the model lowers it."""

    program = capture_source(memory.source, (INDEX,))
    return lower_dynamic_program(
        program,
        (memory.rounds,),
        max_dynamic_measurements=memory.rounds * len(memory.code.checks),
    ).circuit


def _measurement_plan(memory: MemoryCircuit) -> _MeasurementPlan:
    """Return the record columns of every syndrome and terminal readout.

    The plan is read off the lowered program, not asserted about it: the loop
    must have lowered to identical round blocks, and each block must measure
    every check exactly once, in the code's declared check order. Both
    properties are what makes a round index and a check index name one recorded
    bit, so a program without them is refused.
    """

    program = _lowered_program(memory)
    instructions = program.instructions
    rounds = memory.rounds
    checks = tuple(memory.code.checks)
    if len(instructions) % rounds:
        raise ValueError(
            f"the memory program lowers to {len(instructions)} instruction(s), "
            f"which {rounds} rounds cannot divide into identical syndrome rounds"
        )
    block = len(instructions) // rounds
    # Two rounds are the same round when they execute the same operations on the
    # same wires. They are not the same records: a lowered measurement carries
    # the absolute classical bit it writes, which necessarily differs per round,
    # so equality of the whole instruction would refuse every program.
    template = tuple(
        (instruction.name, instruction.wires) for instruction in instructions[:block]
    )
    for round_index in range(1, rounds):
        start = round_index * block
        if (
            tuple(
                (instruction.name, instruction.wires)
                for instruction in instructions[start : start + block]
            )
            != template
        ):
            raise ValueError(
                f"the memory program's round {round_index} is not identical to "
                "round zero, so a noise location cannot be attributed to a round"
            )
    measures = tuple(
        index for index, (name, _) in enumerate(template) if name == _MEASURE_OPCODE
    )
    expected = tuple((check.ancilla_wire,) for check in checks)
    if tuple(template[index][1] for index in measures) != expected:
        raise ValueError(
            "the memory program's syndrome round does not measure the code's "
            "checks once each in the declared order, so a recorded bit cannot be "
            "attributed to a check"
        )
    columns = len(checks)
    syndrome_columns = {
        (round_index, check.ancilla_wire): round_index * columns + position
        for round_index in range(rounds)
        for position, check in enumerate(checks)
    }
    data_wires = tuple(memory.code.data_wires)
    terminal_columns = {
        wire: columns * rounds + position for position, wire in enumerate(data_wires)
    }
    return _MeasurementPlan(
        program=program,
        block=block,
        measure_offsets=measures,
        syndrome_columns=syndrome_columns,
        terminal_columns=terminal_columns,
    )


def _noise_locations(
    memory: MemoryCircuit, plan: _MeasurementPlan, noise: PhenomenologicalNoise
) -> tuple[_NoiseLocation, ...]:
    """Return every location the noise record configures, in placement order.

    The three data families come first, one pass per family and each pass one
    round per data wire, with a location before the round's first instruction;
    measurement locations come second, one per round per check, sitting before
    the readout of the check it corrupts. Each location carries the rate that
    named it -- the element of the vector, or the scalar when no vector is
    stated -- and a location whose own rate is zero is dropped here rather than at
    placement, so the caller never pays for a channel that cannot fire and a
    per-element vector is how one quiet location between two noisy ones is stated.

    A wire can therefore carry up to three locations in one round, one per
    family, and the record states each family's rate separately: a profile that
    is noisy in one Pauli and quiet in the others costs one channel per wire, not
    three. The families and their order are the ones
    :func:`~flagquantum.qec.dem_construction._mechanisms` enumerates, so the two
    routes place and record the same locations.

    The per-check vector is indexed in the matrix row order, Z-type checks first,
    which is not the declaration order the plan's offsets are in; the two are
    related by :func:`~flagquantum.qec.dem_construction._check_rate_order`, the
    same translation the construction routes use, so a rate cannot name one check
    here and another there.

    Raises:
        ValueError: If a per-element rate vector the noise states does not name
            every data wire or every check the code declares.
    """

    checks = tuple(memory.code.checks)
    data_wires = memory.code.data_wires
    measurement_flip_rates = noise.measurement_flip_rates(num_checks=len(checks))
    rate_order = _check_rate_order(checks)
    locations: list[_NoiseLocation] = []
    for kind, rates in _data_family_rates(noise, num_qubits=len(data_wires)):
        for round_index in range(memory.rounds):
            start = round_index * plan.block
            locations.extend(
                _NoiseLocation(kind, round_index, int(wire), start, rates[position])
                for position, wire in enumerate(data_wires)
                if rates[position]
            )
    for round_index in range(memory.rounds):
        start = round_index * plan.block
        locations.extend(
            _NoiseLocation(
                "measurement",
                round_index,
                int(check.ancilla_wire),
                start + plan.measure_offsets[position],
                measurement_flip_rates[rate_order[position]],
            )
            for position, check in enumerate(checks)
            if measurement_flip_rates[rate_order[position]]
        )
    return tuple(locations)


def _noisy_program(
    plan: _MeasurementPlan,
    locations: Sequence[_NoiseLocation],
) -> CircuitIR:
    """Return the lowered program with one Pauli channel per location.

    Locations are placed before the instruction they were derived for, which is
    what makes a data error a round-boundary error and a measurement error a
    readout error. Every location handed in gets a channel at the rate that
    location carries: a location whose effective rate is zero never produced one,
    because :func:`_noise_locations` drops it, and the record is not consulted
    here at all. Re-deriving the rate from the record would give two places that
    decide what a location's rate is, and a per-element vector is exactly the
    case where the two could answer differently.

    The channel itself is the engine's single bit flip in every case, wrapped in
    the family's conjugation from :data:`_FAMILY_CONJUGATION`. The wrapper is
    what makes the draw a Z or a Y with one rate rather than a pair of
    independent bit flips, and it leaves the noiseless circuit alone because
    ``H H`` and ``S_DAG S`` are the identity when the channel does not fire.
    """

    by_index: dict[int, list[tuple[int, _MechanismKind, float]]] = {}
    for location in locations:
        by_index.setdefault(location.instruction_index, []).append(
            (location.wire, location.kind, float(location.probability))
        )
    instructions: list[Instruction] = []
    for index, instruction in enumerate(plan.program.instructions):
        for wire, kind, probability in by_index.get(index, ()):
            before, after = _FAMILY_CONJUGATION[kind]
            if before:
                instructions.append(Instruction(name=before, wires=(wire,)))
            instructions.append(
                Instruction(
                    name=_NOISE_OPCODE,
                    wires=(wire,),
                    params={"probability": probability},
                    matrix=bit_flip_channel(probability).kraus,
                    metadata={"is_channel": True},
                )
            )
            if after:
                instructions.append(Instruction(name=after, wires=(wire,)))
        instructions.append(instruction)
    return CircuitIR(
        n_wires=plan.program.n_wires,
        instructions=tuple(instructions),
        metadata=dict(plan.program.metadata),
    )


def _recorded_bits(
    memory: MemoryCircuit, plan: _MeasurementPlan, record: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the detection events and observable flips a record holds."""

    shots = int(record.shape[0])
    dtype = torch.int8
    detectors = torch.zeros((shots, len(memory.detectors)), dtype=dtype)
    observables = torch.zeros((shots, len(memory.observables)), dtype=dtype)

    def recorded(reference: MeasurementRef) -> torch.Tensor:
        if reference.round_index is None:
            return record[:, plan.terminal_columns[reference.wire]]
        column = plan.syndrome_columns.get((reference.round_index, reference.wire))
        if column is None:
            raise ValueError(
                f"detector syndrome measurement names ancilla wire "
                f"{reference.wire}, which no check owns"
            )
        return record[:, column]

    for detector in memory.detectors.detectors:
        parity = torch.zeros(shots, dtype=dtype)
        for reference in detector.parity:
            parity ^= recorded(reference)
        detectors[:, detector.index] = parity
    for observable in memory.observables.observables:
        parity = torch.zeros(shots, dtype=dtype)
        for reference in observable.measurement_parity:
            parity ^= record[:, plan.terminal_columns[reference.wire]]
        observables[:, observable.index] = parity
    return detectors, observables


def sample_memory_circuit(
    circuit: MemoryCircuit,
    *,
    noise: PhenomenologicalNoise,
    shots: int,
    seed: int | None = None,
) -> DemSample:
    """Sample detection events and observable flips from a memory experiment.

    The circuit is lowered once, each configured noise location is placed at the
    instruction it belongs to, and the resulting circuit is executed on the
    stabilizer engine. A shot's detection events are the parities the detector
    layout declares, read off the recorded syndrome bits and terminal data
    readouts; an observable flip is the parity of its support's terminal
    readouts.

    The result is a sample of the experiment, not of
    `~flagquantum.qec.DetectorErrorModel`: the model is this sampler's
    statistical reference in the tests, and no mechanism signature is reused
    from it here.

    Args:
        circuit: The configured memory experiment to sample.
        noise: The phenomenological noise the experiment runs under.
        shots: Positive number of sampled shots.
        seed: Seed for the sampling stream. The stabilizer engine documents
            identical samples for an identical seed only on one engine version
            and one machine's instruction set.

    Returns:
        A `~flagquantum.qec.DemSample` whose ``detectors`` and ``observables``
        are `int8` tensors of shape ``(shots, num_detectors)`` and
        ``(shots, num_observables)``.

    Raises:
        TypeError: ``circuit`` is not a memory circuit, ``noise`` is not a
            phenomenological noise record, or ``shots`` or ``seed`` is not an
            integer.
        ValueError: ``shots`` is not positive, or the program does not lower to
            identical syndrome rounds that measure each check once.
        StabilizerDependencyError: The optional stabilizer engine is not
            installed.
    """

    if not isinstance(circuit, MemoryCircuit):
        raise TypeError("circuit must be a MemoryCircuit")
    if not isinstance(noise, PhenomenologicalNoise):
        raise TypeError("noise must be a PhenomenologicalNoise")
    shots = _checked_shots(shots)
    seed = _checked_seed(seed)
    plan = _measurement_plan(circuit)
    locations = _noise_locations(circuit, plan, noise)
    program = _noisy_program(plan, locations)
    data_wires = tuple(circuit.code.data_wires)
    record = sample_noisy_measurements(
        program, shots=shots, terminal_wires=data_wires, seed=seed
    )
    detectors, observables = _recorded_bits(circuit, plan, record)
    return DemSample(detectors=detectors, observables=observables)


__all__ = ("sample_memory_circuit",)
