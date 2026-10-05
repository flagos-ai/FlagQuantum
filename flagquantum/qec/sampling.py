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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Literal

import torch

from ..compiler._hybrid import INDEX, capture_source, lower_dynamic_program
from ..core.ir import CircuitIR, Instruction
from ..noise import KrausChannel, bit_flip_channel, phase_flip_channel, y_flip_channel
from ..simulation.stabilizer import sample_noisy_measurements
from .circuit import MeasurementRef, MemoryCircuit
from .dem import DemSample
from .dem_construction import _check_rate_order
from .noise import PhenomenologicalNoise

_MEASURE_OPCODE = "measure"
# The single-qubit fault a data location may carry, as the channel that places
# it. The three names are the three Pauli faults the noise record states: an X
# fault is `bit_flip`, a Z fault is `phase_flip`, and a Y fault is `y_flip`,
# which is one branch carrying the square root of the rate rather than an X fault
# and a Z fault that fire independently. A measurement fault is not a Pauli fault
# at all; it is placed as `bit_flip` because a misreported syndrome bit is a flip
# of that bit.
_ChannelName = Literal["bit_flip", "phase_flip", "y_flip"]
_DATA_FAULT_CHANNELS: Mapping[_ChannelName, Callable[[float], KrausChannel]] = (
    MappingProxyType(
        {
            "bit_flip": bit_flip_channel,
            "phase_flip": phase_flip_channel,
            "y_flip": y_flip_channel,
        }
    )
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

    ``channel`` is the opcode of the channel this location is placed as, which is
    what separates two locations that share a kind, a wire, and a round: the
    three data faults are three locations at one instruction, and a location that
    named only its wire would place X faults where Z faults were configured.
    """

    kind: Literal["data", "measurement"]
    channel: Literal["bit_flip", "phase_flip", "y_flip"]
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

    A data location sits before the round's first instruction for every data wire
    and every one of the three data faults; a measurement location sits before the
    readout of the check it corrupts. Each location carries the rate that named it
    -- the element of the vector, or the scalar when no vector is stated -- and a
    location whose own rate is zero is dropped here rather than at placement, so
    the caller never pays for a channel that cannot fire and a per-element vector
    is how one quiet location between two noisy ones is stated.

    All three data faults are placed at the same instruction, and that is the
    location model's own statement rather than a simplification: the record
    applies each of them "to every data wire at the start of every syndrome round,
    before that round's parity-check CNOTs", so the three differ in which channel
    is placed and not in where it goes. The three channels are the three Pauli
    faults, and the Z fault propagates to the X-type checks the way the X fault
    propagates to the Z-type ones, which is a property of the circuit this module
    executes rather than anything derived here. A Y fault is placed as one channel
    with one branch rather than as an X fault and a Z fault beside each other,
    because the two would fire independently and that is a different channel.

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
    data_families: tuple[tuple[_ChannelName, Sequence[float]], ...] = (
        ("bit_flip", noise.data_flip_rates(num_qubits=len(memory.code.data_wires))),
        ("phase_flip", noise.phase_flip_rates(num_qubits=len(memory.code.data_wires))),
        ("y_flip", noise.both_flip_rates(num_qubits=len(memory.code.data_wires))),
    )
    measurement_flip_rates = noise.measurement_flip_rates(num_checks=len(checks))
    rate_order = _check_rate_order(checks)
    locations: list[_NoiseLocation] = []
    for round_index in range(memory.rounds):
        start = round_index * plan.block
        for channel, rates in data_families:
            locations.extend(
                _NoiseLocation(
                    "data", channel, round_index, int(wire), start, rates[position]
                )
                for position, wire in enumerate(memory.code.data_wires)
                if rates[position]
            )
    for round_index in range(memory.rounds):
        start = round_index * plan.block
        locations.extend(
            _NoiseLocation(
                "measurement",
                "bit_flip",
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
    """Return the lowered program with one channel per location.

    Locations are placed before the instruction they were derived for, which is
    what makes a data error a round-boundary error and a measurement error a
    readout error. Every location handed in gets a channel at the rate that
    location carries: a location whose effective rate is zero never produced one,
    because :func:`_noise_locations` drops it, and the record is not consulted
    here at all. Re-deriving the rate from the record would give two places that
    decide what a location's rate is, and a per-element vector is exactly the
    case where the two could answer differently.

    The channel is read off the location the same way its rate is, and for the
    same reason: the record states three data faults at one instruction, so a
    placement that decided the channel from the wire alone would have to pick one
    of the three and would silently place X faults where Z faults were configured.

    Two locations may share an instruction and a wire -- the three data faults do
    -- and each gets its own instruction, because two channels at one rate are not
    the same channel as one channel at twice the rate. Each channel is placed at
    the location's own rate, which is what makes the composition exact: applying a
    Z fault at ``p_z`` and a Y fault at ``p_y`` are independent branches of
    different channels, and the engine composes them over the Pauli group.
    """

    by_index: dict[int, list[tuple[_ChannelName, int, float]]] = {}
    for location in locations:
        by_index.setdefault(location.instruction_index, []).append(
            (location.channel, location.wire, float(location.probability))
        )
    instructions: list[Instruction] = []
    for index, instruction in enumerate(plan.program.instructions):
        for channel, wire, probability in by_index.get(index, ()):
            factory = _DATA_FAULT_CHANNELS[channel]
            instructions.append(
                Instruction(
                    name=channel,
                    wires=(wire,),
                    params={"probability": float(probability)},
                    matrix=factory(probability).kraus,
                    metadata={"is_channel": True},
                )
            )
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
