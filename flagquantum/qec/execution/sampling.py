"""Sampled detection events from a configured memory experiment.

A detector error model states what the noise does; a sampler states what a run
of the experiment reads out. This module supplies the second: it lowers the
memory circuit once, places each noise location the caller's record configures
at the instruction it belongs to, executes the resulting circuit on the
stabilizer engine, and reads the detection-event and observable layouts off the
recorded bits.

Where a location belongs is the whole content of the join, and the record the
caller states is what decides it. A
`~flagquantum.qec.PhenomenologicalNoise` names round boundaries and check
readouts: a data error is placed *before* the round's first gate, so it opens
the frame that round's detectors compare against, and a measurement error is
placed immediately *before* the readout of the check it corrupts. A
`~flagquantum.noise.NoiseModel` names gates: an error is placed immediately
*after* the gate its rule matched, which is upstream's placement and one error
per round per matching gate. Neither position exists in the source program, whose
bounded hybrid capture refuses a channel call outright, so the placement is
derived here from the lowered program rather than written into the experiment.

The stabilizer engine reads the actual Pauli Kraus operators. A gate-bound
one-/two-qubit mixture is one complete channel at its location, with mutually
exclusive branches; separate channel locations remain independent. The existing
round-boundary X/Z/Y factories and old gate-bound X/Z reconstruction are
preserved. Unsupported channels are refused without substituting another Pauli.

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

import os
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType, TracebackType
from typing import Literal, NoReturn, SupportsIndex

import torch

from flagquantum.qec.dem.construction import (
    _check_rate_order,
    _data_family_rates,
    _gate_bound_mechanisms,
)
from flagquantum.qec.execution.signatures import _MechanismKind

from ...core.ir import CircuitIR, Instruction
from ...noise import (
    KrausChannel,
    NoiseModel,
    bit_flip_channel,
    phase_flip_channel,
    y_flip_channel,
)
from ...simulation.stabilizer import sample_noisy_measurements
from ...simulation.stabilizer.engine import (
    _prepare_noisy_measurements,
    _PreparedMeasurements,
)
from ..circuits.memory import MeasurementRef, MemoryCircuit
from ..dem import DemSample
from ..noise import PhenomenologicalNoise
from .signatures import _lowered_program

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

# The one lowered opcode a memory experiment's readout-basis rotation is a
# sequence of. The rotation itself is built once, in
# :func:`~flagquantum.qec.circuits.memory.build_memory_circuit`, as one gate per data
# qubit; naming the opcode here is what lets the plan recognise it in the lowered
# program without re-deriving the experiment's basis semantics.
_ROTATION_OPCODE = "h"

# Which channel each data-fault family is placed as. A location carries both
# names because they have two readers: the family is the noise record's own field,
# which is what a location is planned and recorded under, and the channel is the
# Pauli that field faults with, which is what :func:`_noisy_program` emits there.
# The two tables are the same three Paulis under the two names those readers use.
_DATA_FAULT_CHANNEL_BY_FAMILY: Mapping[_MechanismKind, _ChannelName] = MappingProxyType(
    {
        "data": "bit_flip",
        "phase": "phase_flip",
        "both": "y_flip",
        "measurement": "bit_flip",
    }
)


@dataclass(frozen=True)
class _MeasurementPlan:
    """Where every detection-event reference sits in the engine's record.

    ``offset`` is the number of instructions the experiment's readout-basis
    rotation lowers to, and ``block`` the number one syndrome round lowers to, so
    round ``r`` owns
    ``program.instructions[offset + r * block : offset + (r + 1) * block]``.
    The two are stated separately rather than folded into one arithmetic because
    the rotation belongs to the experiment and not to a round: it is what makes
    the instruction at ``offset`` a round-boundary location rather than part of
    the preparation. ``measure_offsets`` is the template position of each check's
    readout, in the code's declared check order. ``syndrome_columns`` and
    ``terminal_columns`` are engine record columns: the first indexes
    ``(round_index, ancilla_qubit)``, the second a data qubit.
    """

    program: CircuitIR
    offset: int
    block: int
    measure_offsets: tuple[int, ...]
    syndrome_columns: Mapping[tuple[int, int], int]
    terminal_columns: Mapping[int, int]


@dataclass(frozen=True, eq=False)
class MeasurementSamples:
    """Every recorded bit of a sampled run, addressed by its measurement handle.

    A detection event says what a *parity* of recorded bits came out as; this
    record says what each bit itself came out as, per handle, which is what host
    logic over a mid-circuit measurement needs and what the layout objects cannot
    supply on their own: a handle the layouts drop is readable here, and a
    detector's parity can be recomputed from the handles it names.

    ``refs`` is the handle vector the experiment declares, in the order
    :attr:`~flagquantum.qec.MemoryCircuit.measurement_refs` states it, and
    ``outcomes`` is an ``int8`` tensor of shape ``(shots, len(refs))`` parallel
    to it. The two readings over a chosen sub-vector are the point of the record:
    :meth:`vector` gives one boolean per shot per handle, in the order asked for,
    and :meth:`integer` packs that same order into one integer per shot with the
    first handle in the least significant bit.

    Equality compares the tensors by content, and instances are deliberately
    unhashable, for the reason `~flagquantum.qec.DemSample` is.
    """

    refs: tuple[MeasurementRef, ...]
    outcomes: torch.Tensor

    def __post_init__(self) -> None:
        if not self.refs:
            raise ValueError("a measurement record must address at least one handle")
        if len(set(self.refs)) != len(self.refs):
            raise ValueError("a measurement handle vector cannot repeat a handle")
        if self.outcomes.dim() != 2:
            raise ValueError(
                "measurement outcomes must be a two-dimensional (shots, handles) tensor"
            )
        if self.outcomes.shape[1] != len(self.refs):
            raise ValueError(
                f"measurement outcomes hold {self.outcomes.shape[1]} column(s) for "
                f"{len(self.refs)} handle(s)"
            )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MeasurementSamples):
            return NotImplemented
        return self.refs == other.refs and torch.equal(self.outcomes, other.outcomes)

    @property
    def shots(self) -> int:
        """Number of sampled shots."""

        return int(self.outcomes.shape[0])

    @property
    def num_handles(self) -> int:
        """Number of measurement handles this record addresses."""

        return len(self.refs)

    def _columns(self, references: Sequence[MeasurementRef]) -> tuple[int, ...]:
        if not references:
            raise ValueError("a measurement reading must name at least one handle")
        if any(not isinstance(reference, MeasurementRef) for reference in references):
            raise TypeError("measurement readings are addressed by MeasurementRef")
        known = {reference: column for column, reference in enumerate(self.refs)}
        columns: list[int] = []
        for reference in references:
            column = known.get(reference)
            if column is None:
                raise ValueError(
                    f"measurement handle (round={reference.round_index}, "
                    f"qubit={reference.qubit}) is not one this experiment records"
                )
            columns.append(column)
        return tuple(columns)

    def outcome(self, reference: MeasurementRef) -> torch.Tensor:
        """One handle's recorded bit, per shot, as an ``int8`` tensor."""

        (column,) = self._columns((reference,))
        return self.outcomes[:, column]

    def vector(self, references: Sequence[MeasurementRef]) -> torch.Tensor:
        """The handles' recorded bits as a boolean ``(shots, len(references))``.

        This is the reading a caller uses where logic depends on a value rather
        than on a parity: the column order is the order asked for, so the same
        handles read in two orders give two different vectors rather than one.
        """

        columns = self._columns(references)
        return self.outcomes[:, list(columns)].to(torch.bool)

    def integer(self, references: Sequence[MeasurementRef]) -> torch.Tensor:
        """The same handles packed into one integer per shot, as an ``int64``.

        The first handle named is the least significant bit. The packing is
        stated here rather than inherited, because the order a caller asks in is
        the only order this record knows: two calls naming the same handles in
        two orders state two different integers, and both are the reading of the
        order given.
        """

        columns = self._columns(references)
        bits = self.outcomes[:, list(columns)].to(torch.int64)
        weights = torch.tensor(
            [1 << position for position in range(len(columns))],
            dtype=torch.int64,
            device=bits.device,
        )
        return (bits * weights).sum(dim=1)


@dataclass(frozen=True)
class _NoiseLocation:
    """One noise mechanism, the instruction it is placed before, and its rate.

    The rate is carried rather than looked up at placement because a per-element
    vector gives every location its own: a location is built only where its own
    effective rate is nonzero, and the channel placed there is that rate's
    channel rather than the family's scalar.

    ``kind`` names the noise record's own field, as
    :class:`~flagquantum.qec.dem.construction._Mechanism` does, so the two routes
    enumerate the same families under the same names, and ``channel`` names the
    channel that family is placed as. A family's fault *is* one Pauli, so a
    location states which of the three channels carries it rather than leaving
    placement to re-derive a Pauli from a family's name.
    """

    kind: _MechanismKind
    channel: _ChannelName
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


def _rotation_template(
    memory: MemoryCircuit,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Return the lowered instruction template of the experiment's rotation.

    One rotation instruction per data qubit, in the code's declared data-qubit
    order. A Z-basis experiment is the all-zero preparation the runtime already
    starts in and reads out through the runtime's final sample, so its rotation
    is the empty sequence and its program begins at round zero.
    """

    if memory.readout_basis == "z":
        return ()
    return tuple((_ROTATION_OPCODE, (qubit,)) for qubit in memory.code.data_qubits)


def _measurement_plan(memory: MemoryCircuit) -> _MeasurementPlan:
    """Return the record columns of every syndrome and terminal readout.

    The plan is read off the lowered program, not asserted about it. Before the
    rounds the program must carry the experiment's basis rotation, and after them
    the same rotation again, because that is what makes the first instruction of
    round zero a round boundary rather than part of the preparation and the
    terminal sample the readout the observable is declared in; a program whose
    ends are anything else is refused rather than attributed to a round. What is
    left between them must be identical round blocks, and each block must measure
    every check exactly once, in the code's declared check order. Those three
    properties are what make a round index and a check index name one recorded
    bit.
    """

    program = _lowered_program(memory)
    instructions = program.instructions
    rounds = memory.rounds
    checks = tuple(memory.code.checks)
    rotation = _rotation_template(memory)
    offset = len(rotation)

    def template_of(
        source: Sequence[Instruction],
    ) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return tuple((item.name, item.wires) for item in source)

    if len(instructions) < 2 * offset:
        raise ValueError(
            f"the memory program lowers to {len(instructions)} instruction(s), "
            f"which cannot carry the {memory.readout_basis.upper()}-basis "
            f"rotation of its {offset} data qubit(s) on both ends"
        )
    for end, source in (
        ("prepare", instructions[:offset]),
        ("read out", instructions[len(instructions) - offset :]),
    ):
        if template_of(source) != rotation:
            raise ValueError(
                f"the memory program does not {end} its data qubits in the "
                f"experiment's {memory.readout_basis.upper()} basis, so a noise "
                "location would be placed in a program the layouts do not describe"
            )
    body = instructions[offset : len(instructions) - offset]
    if len(body) % rounds:
        raise ValueError(
            f"the memory program lowers to {len(body)} instruction(s) between its "
            f"rotations, which {rounds} rounds cannot divide into identical "
            "syndrome rounds"
        )
    block = len(body) // rounds
    # Two rounds are the same round when they execute the same operations on the
    # same qubits. They are not the same records: a lowered measurement carries
    # the absolute classical bit it writes, which necessarily differs per round,
    # so equality of the whole instruction would refuse every program.
    template = template_of(body[:block])
    for round_index in range(1, rounds):
        start = round_index * block
        if template_of(body[start : start + block]) != template:
            raise ValueError(
                f"the memory program's round {round_index} is not identical to "
                "round zero, so a noise location cannot be attributed to a round"
            )
    measures = tuple(
        index for index, (name, _) in enumerate(template) if name == _MEASURE_OPCODE
    )
    expected = tuple((check.ancilla_qubit,) for check in checks)
    if tuple(template[index][1] for index in measures) != expected:
        raise ValueError(
            "the memory program's syndrome round does not measure the code's "
            "checks once each in the declared order, so a recorded bit cannot be "
            "attributed to a check"
        )
    columns = len(checks)
    syndrome_columns = {
        (round_index, check.ancilla_qubit): round_index * columns + position
        for round_index in range(rounds)
        for position, check in enumerate(checks)
    }
    data_wires = tuple(memory.code.data_qubits)
    terminal_columns = {
        wire: columns * rounds + position for position, wire in enumerate(data_wires)
    }
    return _MeasurementPlan(
        program=program,
        offset=offset,
        block=block,
        measure_offsets=measures,
        syndrome_columns=syndrome_columns,
        terminal_columns=terminal_columns,
    )


def _noise_locations(
    memory: MemoryCircuit,
    plan: _MeasurementPlan,
    noise: PhenomenologicalNoise | NoiseModel,
) -> tuple[_NoiseLocation, ...]:
    """Return every location the stated record configures, in placement order.

    Which record the caller states is what selects the placement grammar, and the
    two are one union rather than two entry points: a caller changing the record
    changes the experiment's description and nothing else. A round-boundary
    record is placed by :func:`_round_boundary_locations` and a gate-bound one by
    :func:`_gate_bound_locations`, which read the same lowered program this plan
    was built from.

    Raises:
        TypeError: ``noise`` is neither record.
    """

    if isinstance(noise, NoiseModel):
        return _gate_bound_locations(plan, noise)
    if not isinstance(noise, PhenomenologicalNoise):
        raise TypeError(
            "noise must be a PhenomenologicalNoise or a NoiseModel, got "
            f"{type(noise).__name__}"
        )
    return _round_boundary_locations(memory, plan, noise)


def _gate_bound_locations(
    plan: _MeasurementPlan, noise: NoiseModel
) -> tuple[_NoiseLocation, ...]:
    """Return every location a gate-bound record places, in placement order.

    The faults and the instruction each one follows come from
    :func:`~flagquantum.qec.dem.construction._gate_bound_mechanisms`, the walk the
    model's own route splices with, so a fault cannot be placed at one position
    here and read at another there. The position a location is placed at is the
    instruction *after* the gate the rule matched, because a channel bound to a
    named gate acts on the state that gate leaves behind; the round the fault is
    reported in is the round that gate is in, which is what ``plan``'s block
    length converts an instruction index into.

    Every channel :func:`~flagquantum.qec.dem.construction._gate_bound_mechanisms`
    accepts is a single-qubit Pauli fault, so every one of them is one of the
    three channels :data:`_DATA_FAULT_CHANNELS` holds and no location reaches the
    engine as a channel it would refuse.
    """

    return tuple(
        _NoiseLocation(
            kind=mechanism.kind,
            channel=_DATA_FAULT_CHANNEL_BY_FAMILY[mechanism.kind],
            round_index=mechanism.after_instruction // plan.block,
            wire=mechanism.wire,
            instruction_index=mechanism.after_instruction + 1,
            probability=mechanism.probability,
        )
        for mechanism in _gate_bound_mechanisms(noise, plan.program)
    )


def _round_boundary_locations(
    memory: MemoryCircuit, plan: _MeasurementPlan, noise: PhenomenologicalNoise
) -> tuple[_NoiseLocation, ...]:
    """Return every location the round-boundary record configures, in order.

    A data location sits at the round's first instruction for every data qubit
    and every one of the three data faults; a measurement location sits before the
    readout of the check it corrupts. Each location carries the rate that named it
    -- the element of the vector, or the scalar when no vector is stated -- and a
    location whose own rate is zero is dropped here rather than at placement, so
    the caller never pays for a channel that cannot fire and a per-element vector
    is how one quiet location between two noisy ones is stated.

    All three data faults are placed at the same instruction, and that is the
    location model's own statement rather than a simplification: the record
    applies each of them "to every data qubit at the start of every syndrome round,
    before that round's parity-check CNOTs", so the three differ in which channel
    is placed and not in where it goes. A round's three families are one group
    before that round's first gate, one pass per family in the order
    :func:`~flagquantum.qec.dem.construction._data_family_rates` states them, so a
    family cannot be placed under one name here and recorded under another there.
    The three channels are the three Pauli faults, and the Z fault propagates to
    the X-type checks the way the X fault propagates to the Z-type ones, which is
    a property of the circuit this module executes rather than anything derived
    here. A Y fault is placed as one channel with one branch rather than as an X
    fault and a Z fault beside each other, because the two would fire
    independently and that is a different channel.

    The per-check vector is indexed in the matrix row order, Z-type checks first,
    which is not the declaration order the plan's offsets are in; the two are
    related by :func:`~flagquantum.qec.dem.construction._check_rate_order`, the
    same translation the construction routes use, so a rate cannot name one check
    here and another there.

    Raises:
        ValueError: If a per-element rate vector the noise states does not name
            every data qubit or every check the code declares.
    """

    checks = tuple(memory.code.checks)
    data_wires = memory.code.data_qubits
    data_families = _data_family_rates(noise, num_qubits=len(data_wires))
    measurement_flip_rates = noise.measurement_flip_rates(num_checks=len(checks))
    # The matrix row order is what a *per-check vector* is stated in, and it is
    # the same translation the construction route applies, so a rate cannot name
    # one check here and another there. A uniform rate is order-free, which is
    # what lets a code carrying a mixed check -- with no family order to be read
    # in -- be sampled at all.
    rate_order = (
        _check_rate_order(checks)
        if noise.measurement_flip_per_check
        else tuple(range(len(checks)))
    )
    locations: list[_NoiseLocation] = []
    for round_index in range(memory.rounds):
        start = plan.offset + round_index * plan.block
        for kind, rates in data_families:
            channel = _DATA_FAULT_CHANNEL_BY_FAMILY[kind]
            locations.extend(
                _NoiseLocation(
                    kind, channel, round_index, int(qubit), start, rates[position]
                )
                for position, qubit in enumerate(data_wires)
                if rates[position]
            )
    for round_index in range(memory.rounds):
        start = plan.offset + round_index * plan.block
        locations.extend(
            _NoiseLocation(
                "measurement",
                "bit_flip",
                round_index,
                int(check.ancilla_qubit),
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

    The channel is read off the location the same way its rate is, and for the
    same reason: a location states the channel its family is placed as, so a
    placement that decided the Pauli from the qubit alone would have to pick one of
    the three and would silently place X faults where Z faults were configured.

    Two locations may share an instruction and a qubit -- one per data-fault
    family -- and each gets its own instruction, because two channels at one rate
    are not the same channel as one channel at twice the rate. Each channel is
    placed at the location's own rate and carries the Kraus pair its own factory
    builds, which is what makes the composition exact: a Z fault at ``p_z`` and a
    Y fault at ``p_y`` are independent channels, and the engine reads each one's
    Pauli off the pair rather than off its name.
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


def _positioned_noise_program(
    circuit: MemoryCircuit,
    plan: _MeasurementPlan,
    noise: PhenomenologicalNoise | NoiseModel,
) -> CircuitIR:
    if not isinstance(noise, NoiseModel):
        return _noisy_program(plan, _noise_locations(circuit, plan, noise))
    from ..dem.circuit_noise import _circuit_noise_locations

    by_index: dict[int, list[Instruction]] = {}
    for location in _circuit_noise_locations(noise, plan.program):
        by_index.setdefault(location.position + 1, []).append(
            Instruction(
                name=location.channel.name,
                wires=location.qubits,
                params=(
                    {"probability": location.terms[0][1]}
                    if location.independent
                    else {}
                ),
                matrix=location.channel.kraus,
                metadata={"is_channel": True},
            )
        )
    instructions: list[Instruction] = []
    for index, instruction in enumerate(plan.program.instructions):
        instructions.extend(by_index.get(index, ()))
        instructions.append(instruction)
    return CircuitIR(
        n_wires=plan.program.n_wires,
        instructions=tuple(instructions),
        metadata=dict(plan.program.metadata),
    )


def _parity_columns(
    memory: MemoryCircuit, plan: _MeasurementPlan
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Snapshot parity columns; later draws retain no caller-owned layouts."""

    def column(reference: MeasurementRef) -> int:
        if reference.round_index is None:
            return plan.terminal_columns[reference.qubit]
        result = plan.syndrome_columns.get((reference.round_index, reference.qubit))
        if result is None:
            raise ValueError(
                f"detector syndrome measurement names ancilla qubit "
                f"{reference.qubit}, which no check owns"
            )
        return result

    detectors: list[tuple[int, ...]] = [()] * len(memory.detectors)
    observables: list[tuple[int, ...]] = [()] * len(memory.observables)
    for detector in memory.detectors.detectors:
        detectors[detector.index] = tuple(column(ref) for ref in detector.parity)
    for observable in memory.observables.observables:
        observables[observable.index] = tuple(
            plan.terminal_columns[ref.qubit] for ref in observable.measurement_parity
        )
    return tuple(detectors), tuple(observables)


def _parity_bits(
    columns: tuple[tuple[int, ...], ...], record: torch.Tensor
) -> torch.Tensor:
    result = torch.zeros((record.shape[0], len(columns)), dtype=torch.int8)
    groups: dict[int, list[tuple[int, tuple[int, ...]]]] = {}
    for index, parity in enumerate(columns):
        if parity:
            groups.setdefault(len(parity), []).append((index, parity))
    if not groups:
        return result
    # Equal-width byte gathers avoid padding long observables or materializing
    # int64 records per reference. Byte sums wrap modulo 256, preserving parity
    # even for products wider than 255; empty parities remain zero.
    binary_record = record.to(torch.uint8)
    for group in groups.values():
        outputs = torch.tensor([index for index, _ in group], dtype=torch.long)
        sources = torch.tensor([parity for _, parity in group], dtype=torch.long)
        bits = binary_record[:, sources].sum(dim=2, dtype=torch.uint8).bitwise_and_(1)
        result[:, outputs] = bits.to(torch.int8)
    return result


def _recorded_bits(
    memory: MemoryCircuit, plan: _MeasurementPlan, record: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the detection events and observable flips a record holds."""
    detectors, observables = _parity_columns(memory, plan)
    return _parity_bits(detectors, record), _parity_bits(observables, record)


def _sample_record(
    circuit: MemoryCircuit,
    *,
    noise: PhenomenologicalNoise | NoiseModel,
    shots: int,
    seed: int | None,
) -> tuple[_MeasurementPlan, torch.Tensor]:
    """Lower, place and execute one experiment, returning its engine record."""

    plan = _measurement_plan(circuit)
    program = _positioned_noise_program(circuit, plan, noise)
    data_qubits = tuple(circuit.code.data_qubits)
    record = sample_noisy_measurements(
        program, shots=shots, terminal_qubits=data_qubits, seed=seed
    )
    return plan, record


def _measurement_samples(
    circuit: MemoryCircuit, plan: _MeasurementPlan, record: torch.Tensor
) -> MeasurementSamples:
    """Read the record by handle, in the order the circuit declares its handles.

    The handle vector and the record's column order are two statements of the
    same layout, so the columns are looked up in the plan rather than assumed to
    be the identity: a lowering whose record disagrees with the handles it was
    derived from is refused here rather than read one column out.
    """

    refs = circuit.measurement_refs
    columns: list[int] = []
    for reference in refs:
        if reference.round_index is None:
            column = plan.terminal_columns.get(reference.qubit)
        else:
            column = plan.syndrome_columns.get((reference.round_index, reference.qubit))
        if column is None:
            raise ValueError(
                f"measurement handle (round={reference.round_index}, "
                f"qubit={reference.qubit}) has no recorded column in the lowered "
                "program"
            )
        columns.append(column)
    return MeasurementSamples(refs=refs, outcomes=record[:, columns].to(torch.int8))


def sample_memory_measurements(
    circuit: MemoryCircuit,
    *,
    noise: PhenomenologicalNoise | NoiseModel,
    shots: int,
    seed: int | None = None,
) -> MeasurementSamples:
    """Sample every recorded bit of a memory experiment, addressed by handle.

    The run is the one :func:`sample_memory_circuit` performs, read one level
    lower: the same lowering, the same noise locations and the same engine
    record, with the record returned whole rather than folded into detection
    events and observable flips. The two readings therefore cannot disagree about
    a shot -- a detector's parity is the parity of the handles it names, and a
    handle the layouts drop is still readable.

    Args:
        circuit: The configured memory experiment to sample.
        noise: The noise the experiment runs under, stated either as the
            round-boundary phenomenological record or as a gate-bound
            `~flagquantum.noise.NoiseModel`.
        shots: Positive number of sampled shots.
        seed: Seed for the sampling stream, under the engine's own reproducibility
            caveat.

    Returns:
        A `~flagquantum.qec.MeasurementSamples` whose ``outcomes`` is an ``int8``
        tensor of shape ``(shots, len(circuit.measurement_refs))``.

    Raises:
        TypeError: ``circuit`` is not a memory circuit, ``noise`` is neither noise
            record, or ``shots`` or ``seed`` is not an integer.
        ValueError: ``shots`` is not positive, or the program does not lower to
            identical syndrome rounds that measure each check once.
        CapabilityError: The noise record names a fault this grammar cannot
            place, or states a channel outside the supported one-/two-qubit Pauli mixture grammar.
        StabilizerDependencyError: The optional stabilizer engine is not
            installed.
    """

    if not isinstance(circuit, MemoryCircuit):
        raise TypeError("circuit must be a MemoryCircuit")
    if not isinstance(noise, (PhenomenologicalNoise, NoiseModel)):
        raise TypeError(
            "noise must be a PhenomenologicalNoise or a NoiseModel, got "
            f"{type(noise).__name__}"
        )
    shots = _checked_shots(shots)
    seed = _checked_seed(seed)
    plan, record = _sample_record(circuit, noise=noise, shots=shots, seed=seed)
    return _measurement_samples(circuit, plan, record)


def sample_memory_circuit(
    circuit: MemoryCircuit,
    *,
    noise: PhenomenologicalNoise | NoiseModel,
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

    Where a location belongs is the record's own statement. A
    `~flagquantum.qec.PhenomenologicalNoise` names round boundaries and check
    readouts, so a data fault opens the frame its round's detectors compare
    against and a readout fault sits before the readout it corrupts. A
    `~flagquantum.noise.NoiseModel` names gates, so a fault sits immediately after
    the gate its rule matched, which is upstream's placement and one fault per
    round per matching gate.

    The result is a sample of the experiment, not of
    `~flagquantum.qec.DetectorErrorModel`: the model is this sampler's
    statistical reference in the tests, and no mechanism signature is reused
    from it here.

    Args:
        circuit: The configured memory experiment to sample.
        noise: The noise the experiment runs under, stated either as the
            round-boundary phenomenological record or as a gate-bound
            `~flagquantum.noise.NoiseModel`.
        shots: Positive number of sampled shots.
        seed: Seed for the sampling stream. The stabilizer engine documents
            identical samples for an identical seed only on one engine version
            and one machine's instruction set.

    Returns:
        A `~flagquantum.qec.DemSample` whose ``detectors`` and ``observables``
        are `int8` tensors of shape ``(shots, num_detectors)`` and
        ``(shots, num_observables)``.

    Raises:
        TypeError: ``circuit`` is not a memory circuit, ``noise`` is neither noise
            record, or ``shots`` or ``seed`` is not an integer.
        ValueError: ``shots`` is not positive, or the program does not lower to
            identical syndrome rounds that measure each check once.
        CapabilityError: The noise record names a fault this grammar cannot
            place, or states a channel outside the supported one-/two-qubit Pauli mixture grammar.
        StabilizerDependencyError: The optional stabilizer engine is not
            installed.
    """

    if not isinstance(circuit, MemoryCircuit):
        raise TypeError("circuit must be a MemoryCircuit")
    if not isinstance(noise, (PhenomenologicalNoise, NoiseModel)):
        raise TypeError(
            "noise must be a PhenomenologicalNoise or a NoiseModel, got "
            f"{type(noise).__name__}"
        )
    shots = _checked_shots(shots)
    seed = _checked_seed(seed)
    plan, record = _sample_record(circuit, noise=noise, shots=shots, seed=seed)
    detectors, observables = _recorded_bits(circuit, plan, record)
    return DemSample(detectors=detectors, observables=observables)


class MemorySampler:
    """A compiled memory experiment owned by its creator process and thread.

    Obtain a handle with :func:`compile_memory_sampler` or the opt-in
    ``flagquantum.qec.adapters.stim.compile_stim_sampler``. Sampling advances one
    continuous engine stream; a fixed seed reproduces an identical sequence of
    sample calls on the same engine version and instruction set. Configuration
    is fixed at construction. Use a context manager or call :meth:`close` to
    release prepared state. Live handles cannot be copied or serialized.
    """

    __slots__ = (
        "_sampler",
        "_detector_columns",
        "_observable_columns",
        "_owner_pid",
        "_owner_thread",
    )

    _sampler: _PreparedMeasurements | None
    _detector_columns: tuple[tuple[int, ...], ...]
    _observable_columns: tuple[tuple[int, ...], ...]
    _owner_pid: int
    _owner_thread: threading.Thread

    def __init__(self) -> None:
        raise TypeError("use compile_memory_sampler to construct a MemorySampler")

    def _check_owner(self) -> None:
        if (
            os.getpid() != self._owner_pid
            or threading.current_thread() is not self._owner_thread
        ):
            raise RuntimeError(
                "memory sampler belongs to its creator process and thread"
            )

    def _open_sampler(self) -> _PreparedMeasurements:
        self._check_owner()
        if self._sampler is None:
            raise RuntimeError("memory sampler is closed")
        return self._sampler

    def sample(self, *, shots: int) -> DemSample:
        """Draw a positive number of independent experiment repetitions.

        Returns caller-owned CPU int8 detector and observable bit tensors.
        Invalid shot counts raise TypeError or ValueError without advancing the
        stream. Lifecycle misuse raises RuntimeError. Execution failures close
        the handle before propagating the original error.
        """
        prepared = self._open_sampler()
        shots = _checked_shots(shots)
        try:
            record = prepared.sample(shots=shots)
            return DemSample(
                detectors=_parity_bits(self._detector_columns, record),
                observables=_parity_bits(self._observable_columns, record),
            )
        except BaseException:
            # RNG advancement cannot be rolled back, including interrupted draws.
            self.close()
            raise

    def close(self) -> None:
        """Release prepared state; repeated calls by the owner are harmless."""
        self._check_owner()
        if self._sampler is not None:
            self._sampler.close()
            self._sampler = None
            self._detector_columns = ()
            self._observable_columns = ()

    def __enter__(self) -> MemorySampler:
        self._open_sampler()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        raise TypeError("live memory samplers cannot be copied or serialized")


def compile_memory_sampler(
    circuit: MemoryCircuit,
    *,
    noise: PhenomenologicalNoise | NoiseModel,
    seed: int | None = None,
) -> MemorySampler:
    """Compile one memory experiment for repeated, bounded-size sample calls.

    Lowering, noise placement, parity mapping and engine compilation happen once.
    The handle snapshots configuration and retains no sampled batches or mutable
    caller layouts. Only its creator process and thread may sample or close it.

    Args:
        circuit: The configured memory experiment.
        noise: Existing phenomenological or gate-bound noise configuration.
        seed: Initial stream seed, with the engine's reproducibility scope.

    Returns:
        A handle supporting ``sample(shots=...)``, ``close()`` and context use.

    Raises:
        TypeError: Circuit, noise or seed has an invalid type.
        ValueError: The configured memory or seed is invalid.
        CapabilityError: The existing noise or Clifford grammar is unsupported.
        StabilizerDependencyError: The optional engine is not installed.

    Examples:
        >>> from importlib.util import find_spec
        >>> from flagquantum.qec import (
        ...     PhenomenologicalNoise, RepetitionCode, build_memory_circuit,
        ...     compile_memory_sampler,
        ... )
        >>> if find_spec("stim") is not None:
        ...     memory = build_memory_circuit(RepetitionCode(distance=3), rounds=3)
        ...     with compile_memory_sampler(memory, noise=PhenomenologicalNoise(), seed=53) as sampler:
        ...         batch = sampler.sample(shots=4)
        ...         assert batch.detectors.shape == (4, 8)
        ...         assert not batch.detectors.any() and not batch.observables.any()
    """
    if not isinstance(circuit, MemoryCircuit):
        raise TypeError("circuit must be a MemoryCircuit")
    if not isinstance(noise, (PhenomenologicalNoise, NoiseModel)):
        raise TypeError(
            "noise must be a PhenomenologicalNoise or a NoiseModel, got "
            f"{type(noise).__name__}"
        )
    seed = _checked_seed(seed)
    plan = _measurement_plan(circuit)
    program = _positioned_noise_program(circuit, plan, noise)
    detectors, observables = _parity_columns(circuit, plan)
    prepared = _prepare_noisy_measurements(
        program, terminal_qubits=tuple(circuit.code.data_qubits), seed=seed
    )
    return _memory_sampler(prepared, detectors, observables)


def _memory_sampler(
    prepared: _PreparedMeasurements,
    detectors: tuple[tuple[int, ...], ...],
    observables: tuple[tuple[int, ...], ...],
) -> MemorySampler:
    """Attach the same owned lifecycle to native records or normalized flips."""
    handle = object.__new__(MemorySampler)
    handle._sampler = prepared
    handle._detector_columns = detectors
    handle._observable_columns = observables
    handle._owner_pid = os.getpid()
    handle._owner_thread = threading.current_thread()
    return handle


__all__ = (
    "MeasurementSamples",
    "sample_memory_circuit",
    "sample_memory_measurements",
    "MemorySampler",
    "compile_memory_sampler",
)
