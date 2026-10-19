"""Sources of the physical mechanisms a detector error model describes.

A :class:`~flagquantum.qec.DetectorErrorModel` states which detectors and which
observables each mechanism flips. This module owns the mechanisms an experiment's
**circuit** describes, in the two grammars the record the caller states selects
between. The other description this package reads -- a code's parity-check and
logical-operator matrices -- belongs to
:mod:`flagquantum.qec.dem_matrices`, whose names are re-exported here so that an
importer of this module keeps reaching them.

A **circuit** description is derived. :func:`_memory_circuit_entries` enumerates
the locations a :class:`~flagquantum.qec.PhenomenologicalNoise` record
configures and reads each one's flip set off the circuit's own detector and
observable layouts, from the code's checks and the frame the experiment is
prepared and read out in. Nothing is lowered and nothing is executed, so a
memory circuit of any width reaches a model: the derivation is a property of the
declarative layout. The route this replaced forced each location through an
execution and read the same flip set off the same layouts; that route survives
as :func:`_forced_signature`, which the suite holds the derivation against
location by location.

A circuit description places its faults in one of two grammars, and the record
the caller states is what selects it. A
:class:`~flagquantum.qec.PhenomenologicalNoise` names round boundaries and check
readouts, so its faults are forced into the experiment's own source -- at the
loop for a data fault, before a measurement for a readout fault.
A :class:`~flagquantum.noise.NoiseModel` names gates, so its faults are spliced
into the lowered program immediately after the instruction each rule matched,
and its rate is read off the channel the rule carries. The second is upstream's
placement, where a channel bound to a named gate acts on the state that gate
leaves behind. Because a gate appears once per round, one gate-bound rule states
one fault per round rather than one fault at one round boundary, and the two
grammars are the two ways the same experiment can be described rather than two
views of one description.

A **matrix** description is read, and it is read next door: this module's
:func:`_memory_circuit_entries` derives a circuit's mechanisms, and
:mod:`flagquantum.qec.dem_matrices` derives a code's from its parity-check and
logical-operator matrices. Its detector geometry is consequently the
code-capacity one -- ``num_rounds * num_checks`` detectors, each round compared
against the round before it, with no terminal data readout -- which is
deliberately not the geometry :func:`_memory_circuit_entries` produces, because a
memory circuit measures the data qubits and so ends in terminal detectors. The
difference belongs to the description, not to the model, and a test pins each
geometry to its own route.

The circuit route enumerates four families -- the X, Z and Y data faults
and the measurement fault -- and forces each one through the program instead of
reading it off a matrix. The injection writes its guard into the experiment's own
source, whose language carries ``h`` and ``x`` as its only single-qubit gates that
take no parameter, so a Z fault is forced as the X fault conjugated by ``h`` and a
Y fault as that Z fault followed by the X fault; both are exact gate identities
rather than approximations. Which detectors and which observables a fault reaches
is decided by the basis the experiment is prepared and read out in, which
:attr:`~flagquantum.qec.MemoryCircuit.readout_basis` states: an X fault flips the
Z-type checks' syndrome and the Z-basis readout, a Z fault flips the X-type
checks' syndrome and the X-basis readout, and a Y fault flips both. A fault
injected at the top of round ``r`` stays on the data qubit, so it opens the
detector band of round ``r`` and cancels in every band after it; a syndrome
measurement flip opens its own check's band at round ``r`` and at ``r + 1``. The
terminal detectors exist only for the checks the preparation state pins, which is
the same set the layout records. What a fault then flips is the program's answer
rather than a plan's, and the Z-basis layout makes it a narrower answer than the
matrix route's: a Z fault never flips the Z-type logical observable, and it reaches
only the X-type checks, which have a detector for every round after the first. A Z
fault at the round-zero boundary is therefore no mechanism at all on a Z-memory
circuit -- the band it would flip has no reference to be compared against -- while
the same fault one round later is one.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from numbers import Integral
from types import MappingProxyType
from typing import Literal

import torch

from ..compiler._hybrid import INDEX, capture_source, lower_dynamic_program
from ..core.ir import CircuitIR, Instruction
from ..errors import CapabilityError, ValidationError
from ..noise import NoiseModel
from ..noise.channels import KrausChannel
from ..runtime.dynamic.hybrid_session import execute_hybrid_dynamic_session
from .circuit import MeasurementRef, MemoryCircuit
from .codes import CodeCheck

# The code record a matrix description is read from stays reachable here, because
# that name reached callers of this module before the matrix route moved next door.
from .codes import StabilizerCode as StabilizerCode
from .dem_matrices import CssCodeMatrices as CssCodeMatrices
from .dem_matrices import Entry
from .dem_matrices import _binary_matrix as _binary_matrix
from .dem_matrices import _code_matrix_entries as _code_matrix_entries
from .dem_matrices import _column_supports as _column_supports
from .dem_matrices import css_code_matrices as css_code_matrices
from .noise import PhenomenologicalNoise

__all__ = ("CssCodeMatrices", "css_code_matrices")

# One data fault's family, named by the field the noise record states it in. The
# sampler needs the same three names, so this is a type rather than a comment.
_DataFaultFamily = Literal["data", "phase", "both"]
_MechanismKind = Literal["data", "phase", "both", "measurement"]


def _data_family_rates(
    noise: PhenomenologicalNoise, *, num_qubits: int
) -> tuple[tuple[_DataFaultFamily, tuple[float, ...]], ...]:
    """Return the three data families' rate vectors in the order both routes use.

    The names are the noise record's own fields -- ``data_flip`` for an X fault,
    ``phase_flip`` for a Z fault and ``both_flip`` for a Y fault -- and this is
    the one place the order they are enumerated in is stated. The sampler
    enumerates data locations from this same sequence, so a family cannot be
    placed under one name there and recorded under another here, and a family
    added to the record reaches both routes by being added here.
    """

    return (
        ("data", noise.data_flip_rates(num_qubits=num_qubits)),
        ("phase", noise.phase_flip_rates(num_qubits=num_qubits)),
        ("both", noise.both_flip_rates(num_qubits=num_qubits)),
    )


def _require_source_anchors(
    circuit: MemoryCircuit, mechanisms: tuple[_Mechanism, ...]
) -> None:
    """Refuse a source that does not carry the lines the mechanisms read.

    A data mechanism is injected at the top of a round and a measurement mechanism
    at a check's own measurement, so the derivation reads the same two lines the
    injectors used to anchor on. Only the lines the enumerated mechanisms actually
    need are required, and each anchor must occur exactly once: a source missing it
    describes a program the model would misdescribe, and a source repeating it
    cannot say which occurrence a mechanism belongs to.
    """

    if any(mechanism.kind in _FAULT_GATES for mechanism in mechanisms):
        _single_anchor(
            circuit.source, _ROUND_LOOP_ANCHOR, label="the round loop anchor"
        )
    if any(mechanism.kind == "measurement" for mechanism in mechanisms):
        for check in circuit.code.checks:
            _single_anchor(
                circuit.source,
                f"{_FLIP_INDENT}last = qp.measure(wires={check.ancilla_qubit})\n",
                label="the check measurement anchor",
            )


def _require_owned_syndrome_references(
    circuit: MemoryCircuit, *, owned: set[int]
) -> None:
    """Refuse a layout whose syndrome reference no check records a bit for.

    A syndrome reference is read at the position of the check that owns its
    ancilla, so an ancilla no check owns has no recorded bit: every detector
    reading it would compare against a bit that does not exist. A layout may name
    one — ``MemoryCircuit`` ties the reference to the code's declared ancilla
    qubits, not to its checks — so the failure is stated rather than left to a
    lookup that finds nothing and reports an even parity.
    """

    for detector in circuit.detectors.detectors:
        for reference in detector.parity:
            if reference.round_index is not None and reference.qubit not in owned:
                raise ValueError(
                    f"detector syndrome measurement names ancilla qubit "
                    f"{reference.qubit}, which no check owns"
                )


def _flipped_measurements(
    circuit: MemoryCircuit, mechanism: _Mechanism
) -> frozenset[tuple[int | None, int]]:
    """Return the layout references one noise location flips.

    A reference is named the way the layouts name it: a syndrome measurement by
    its round and its check's ancilla qubit, and a terminal data sample by ``None``
    and the data qubit. A mechanism is a classical bit-flip set over those
    references, and a detector or an observable fires exactly when its parity meets
    that set in an odd number of places -- so this is the whole of the derivation,
    and everything after it is a parity count.

    A **measurement** mechanism flips the one syndrome bit its check records that
    round. A **data** mechanism flips a syndrome bit in every round from the round
    it is injected in to the last, for every check the fault anticommutes with on
    that qubit, and it flips the terminal sample of the qubit when the fault
    anticommutes with the basis the experiment is read out in. Which checks and
    which readouts those are is read from the code's checks and the circuit's
    ``readout_basis``: a Z fault anticommutes with the X factor a check carries on
    that qubit, an X fault with the Z factor, and a Y fault with either -- so the
    question is asked of **the one factor the check carries on the faulted qubit**,
    which is the whole of the rule. A pure Z-type check carries a Z factor on
    every qubit of its support, so every Z fault on its support flips it and no X
    fault does, which is the rule this derivation started as; a mixed check
    answers the same question per qubit, and a qubit its stabilizer carries no
    factor on is not in its support at all.

    The fault stays on the data qubit, so the run of flipped syndrome bits starts
    at the injected round and does not stop; that is why a data fault opens one
    detector band and cancels in the rest.
    """

    if mechanism.kind == "measurement":
        return frozenset({(mechanism.round_index, mechanism.wire)})
    read_in_x = circuit.readout_basis == "x"
    if mechanism.fault == "y":
        flips_readout = True
    elif mechanism.fault == "z":
        flips_readout = read_in_x
    else:
        flips_readout = not read_in_x
    flips: set[tuple[int | None, int]] = set()
    if flips_readout:
        flips.add((None, mechanism.wire))
    for check in circuit.code.checks:
        stabilizer = check.stabilizer
        if mechanism.wire not in stabilizer.support:
            continue
        if mechanism.fault == "y":
            coupled = True
        elif mechanism.fault == "z":
            coupled = mechanism.wire in stabilizer.x_qubits
        else:
            coupled = mechanism.wire in stabilizer.z_qubits
        if not coupled:
            continue
        flips.update(
            (round_index, check.ancilla_qubit)
            for round_index in range(mechanism.round_index, circuit.rounds)
        )
    return frozenset(flips)


def _layout_signature(
    circuit: MemoryCircuit, mechanism: _Mechanism
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the detectors and observables one mechanism flips.

    Every detector's parity and every observable's parity is counted against the
    references :func:`_flipped_measurements` returns, so the signature is read off
    the layouts rather than derived from the code a second time. The reference
    list is the one the circuit states: a terminal detector reads the data
    qubits
    its own layout names, which is not always the support of the check it belongs
    to, and the count follows the layout either way.
    """

    flips = _flipped_measurements(circuit, mechanism)
    detectors = tuple(
        detector.index
        for detector in circuit.detectors.detectors
        if sum(
            1
            for reference in detector.parity
            if (reference.round_index, reference.qubit) in flips
        )
        % 2
        == 1
    )
    observables = tuple(
        observable.index
        for observable in circuit.observables.observables
        if sum(
            1
            for reference in observable.measurement_parity
            if (reference.round_index, reference.qubit) in flips
        )
        % 2
        == 1
    )
    return detectors, observables


def _memory_circuit_entries(
    circuit: MemoryCircuit,
    noise: PhenomenologicalNoise | NoiseModel,
    *,
    decompose_composite_faults: bool = False,
) -> tuple[int, int, tuple[Entry, ...]]:
    """Return the model shape and every mechanism a memory circuit contains.

    Each location :func:`_mechanisms` enumerates has its signature derived from
    the circuit's own layouts by :func:`_flipped_measurements`, so the model is a
    reading of the record the circuit states rather than a claim about it. The
    derivation needs no execution, so the width of the code does not bound it. A
    gate-bound record is the exception: a rule names a gate rather than a round, so
    its placement is read off the lowered program and its signature off the program
    that placement produces.

    The source is still held against the layouts. A round-boundary record's
    mechanisms are read rather than injected, so a source that does not carry the
    round loop or a check's measurement, or a layout that reads a syndrome bit no
    check records, would otherwise build a model that misdescribes the program it
    was handed. Both are refused with the reason the injection engine used to
    state, and only the parts of the source the enumerated mechanisms actually
    read are required: a circuit modelled from data faults alone needs no
    measurement anchor. A gate-bound record's own refusals reach the caller
    unchanged: a circuit whose source does not lower to the rounds its record
    states fails closed with a stated reason instead of building a model that
    misdescribes it.

    ``decompose_composite_faults`` decides how a composite data fault -- the Y
    family, one fault that is an X flip and a Z flip at one location -- is
    enumerated. Read the default way, it is one mechanism whose signature is the
    XOR of the two parts, which is what the fault does and what
    :mod:`tests.qec.test_data_fault_families` pins. Read the other way, it is two
    faults, the X part and the Z part, each enumerated at the parent's own rate
    and forced at the parent's own round and qubit. The two readings describe
    different distributions and not two encodings of one: read together the parts
    always fire together, read apart they fire independently, so a detector's
    marginal flip rate is the same under both to floating-point rounding while the
    joint law is not. What the second reading buys is a mechanism of at most two
    detectors where the first states one of more, which is the shape a
    minimum-weight matcher weights -- the parts of a composite fault are read off
    the layout exactly as the X and Z faults of their own families are, so they
    are as graphlike as those families are and no more.

    A part that flips nothing is an entry with an empty signature, which
    ``DetectorErrorModel._merge_mechanisms`` drops for every construction route
    alike; no separate rule is stated here, so a part cannot be dropped under one
    reading and kept under the other.

    The detector and observable counts come from the circuit's own layouts rather
    than from a count re-derived from the code, so the returned shape and the
    returned signatures are read from one authority.
    """

    if not isinstance(circuit, MemoryCircuit):
        raise TypeError("circuit must be a MemoryCircuit")
    if not isinstance(decompose_composite_faults, bool):
        raise TypeError("decompose_composite_faults must be a bool")
    entries: list[Entry] = []
    if isinstance(noise, NoiseModel):
        if decompose_composite_faults:
            raise CapabilityError(
                "decompose_composite_faults names the Y family of a "
                "round-boundary record, and this record is gate-bound: a "
                "gate-bound rule states one Pauli per placement, either "
                f"{sorted(_GATE_BOUND_FAMILIES)}, so it states no composite "
                "fault for the option to read apart"
            )
        program = _lowered_program(circuit)
        for bound in _gate_bound_mechanisms(noise, program):
            detectors, observables = _executed_flip_set(
                circuit, _spliced_fault(program, bound)
            )
            entries.append((bound.probability, detectors, observables))
    elif isinstance(noise, PhenomenologicalNoise):
        mechanisms = _mechanisms(circuit, noise)
        if not mechanisms:
            return len(circuit.detectors), len(circuit.observables), ()
        _require_source_anchors(circuit, mechanisms)
        owned = {check.ancilla_qubit for check in circuit.code.checks}
        _require_owned_syndrome_references(circuit, owned=owned)
        for mechanism in mechanisms:
            parts: tuple[_Mechanism, ...] = (mechanism,)
            if decompose_composite_faults and mechanism.kind == _COMPOSITE_KIND:
                parts = tuple(
                    replace(mechanism, kind=kind, fault=_FAULT_LETTER_BY_KIND[kind])
                    for kind in _COMPOSITE_PARTS
                )
            for part in parts:
                if part.kind not in _FAULT_GATES and part.kind != "measurement":
                    # Unreachable through ``_mechanisms``, which sets the field
                    # from the annotation; stated for a caller that builds
                    # records itself, where a bare ``else`` would read an unknown
                    # kind as a measurement flip whenever the qubit is a declared
                    # ancilla.
                    raise ValueError(f"unknown mechanism kind {part.kind!r}")
                entries.append((part.probability, *_layout_signature(circuit, part)))
    else:
        raise TypeError(
            "noise must be a PhenomenologicalNoise or a NoiseModel, got "
            f"{type(noise).__name__}"
        )
    return len(circuit.detectors), len(circuit.observables), tuple(entries)


_ROUND_LOOP_ANCHOR = "    for round_index in range(rounds):\n"
_FLIP_INDENT = "        "
# The single-qubit gates whose product is each Pauli fault, in the hybrid
# language's own gate set, one entry per family the noise record states. The
# injection writes into a source whose language carries `h` and `x` as its only
# parameter-free single-qubit gates, so the two families that are not a bit flip
# are built out of those: `h` conjugates the `x` into a `z`, and a `y` is the bit
# flip followed by that conjugation, since Z X = iY. Both are exact identities, so
# a forced fault is the Pauli it is named for and not an approximation of it; the
# global phase `i` cannot reach a flip set, which is what the signature is read
# from.
_FAULT_GATES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "data": ("X",),
        "phase": ("H", "X", "H"),
        "both": ("X", "H", "X", "H"),
    }
)

# The letter each family's fault is written with, and the inverse a caller that
# names the letter uses. The letters are the three Pauli faults the noise record
# states as three separate rates, and they are what a :class:`_Mechanism` carries
# as ``fault`` so that two data mechanisms can be compared without knowing which
# family an enumeration happened to visit. A measurement mechanism is not a Pauli
# fault at all and is written as the bit flip's letter, because the field is part
# of every record.
_FAULT_LETTER_BY_KIND: Mapping[_MechanismKind, Literal["x", "z", "y"]] = (
    MappingProxyType({"data": "x", "phase": "z", "both": "y", "measurement": "x"})
)
_FAULT_KIND_BY_LETTER: Mapping[str, _MechanismKind] = MappingProxyType(
    {"x": "data", "z": "phase", "y": "both"}
)

# The composite family and the two single-Pauli parts it is the XOR of. The
# sequence is stated once so a decomposition cannot name the parts in one order
# here and another wherever the same fault is described, and so the claim that the
# parts cover the composite exactly is a property of one tuple rather than of two
# literals that happen to agree.
_COMPOSITE_KIND: _MechanismKind = "both"
_COMPOSITE_PARTS: tuple[_MechanismKind, ...] = ("data", "phase")


@dataclass(frozen=True)
class _Mechanism:
    """One physical noise location, by kind, round, and qubit.

    The record carries the coordinates the matching injector needs rather than a
    rendered source. A caller that fires a set of mechanisms at once has to
    re-inject them into the one program it executes, and it cannot rebuild a
    source from a string alone; carrying coordinates also keeps injection in
    exactly one place. ``kind`` names the noise record's own field: ``"data"``
    for an X fault on a data qubit, ``"phase"`` for a Z fault and ``"both"`` for a
    Y fault, where ``wire`` is that data qubit, and ``"measurement"`` for a fault
    on a check's syndrome measurement, where ``wire`` is that check's ancilla. The
    kind is a literal rather than a free string, so a caller that builds a record
    by hand is told at the type checker which flips exist.

    ``fault`` is the same family read as the letter of the Pauli it is, in the
    letters the :class:`~flagquantum.qec.PhenomenologicalNoise` fields are
    documented with: ``"x"`` for a ``"data"`` mechanism, ``"z"`` for a ``"phase"``
    one and ``"y"`` for a ``"both"`` one. It is ``"x"`` for a measurement
    mechanism, which is not a Pauli fault at all, because the field is part of
    every record and inventing a fourth value for the one kind that has no choice
    would make the records harder to compare. The two fields are one statement
    read two ways rather than two claims: :data:`_FAULT_LETTER_BY_KIND` is the only
    place the correspondence is stated, and it is what ``_mechanisms`` sets the
    field from.
    """

    kind: _MechanismKind
    round_index: int
    wire: int
    probability: float
    fault: Literal["x", "z", "y"] = "x"


# The two channels a gate-bound rule may carry, each with the data-fault family
# it is. A bit flip is the X fault and a phase flip is the Z fault -- the same
# pairing :data:`_FAULT_GATES` states -- so a channel and a family cannot be
# paired two ways, and a channel that is neither names a capability this grammar
# does not have rather than a fault it places.
_GATE_BOUND_FAMILIES: Mapping[str, _DataFaultFamily] = MappingProxyType(
    {"bit_flip": "data", "phase_flip": "phase"}
)

# The Pauli each family's fault is, as the matrix the flipped Kraus operator has
# to equal times the rate's square root. A channel's name is a claim about its
# operators, and this is what the claim is checked against.
_PAULI_MATRICES: Mapping[_DataFaultFamily, tuple[tuple[float, float], ...]] = (
    MappingProxyType(
        {
            "data": ((0.0, 1.0), (1.0, 0.0)),
            "phase": ((1.0, 0.0), (0.0, -1.0)),
        }
    )
)

# The same fault identities as :data:`_FAULT_GATES`, written in the lowered
# language: the source route writes the fault into the source, this one writes it
# into the program that source lowered to, and the lowered language refuses `z`
# as outside its dynamic profile, so a phase fault is the bit flip conjugated by
# `h` on both routes. Lowercasing is the correspondence between the two
# languages' gate names, which is what makes the two spellings one statement.
_GATE_BOUND_GATES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        family: tuple(gate.lower() for gate in gates)
        for family, gates in _FAULT_GATES.items()
    }
)

# A gate-bound rule attaches its fault to the gate the rule matched, so the two
# instructions that are not gates cannot be one. A readout fault's own position
# is the measurement family of a round-boundary record, which states which check
# it corrupts, and a preparation fault is not a Pauli fault the forced signature
# can read off the layouts at all.
_UNPLACEABLE_RULE_GATES = frozenset({"measure", "reset"})


@dataclass(frozen=True)
class _GateBoundMechanism:
    """One noise location a rule binds to a named gate, by kind, qubit and rate.

    ``after_instruction`` is the index of the instruction the fault immediately
    follows and is what makes this placement gate-bound: a rule names a gate, the
    gate is matched in the lowered program, and the fault sits after it. A
    round-boundary mechanism has no such index and is a :class:`_Mechanism`
    instead, so which of the two records a caller states is what decides which
    grammar places the fault.
    """

    kind: _DataFaultFamily
    after_instruction: int
    wire: int
    probability: float


def _lowered_program(memory: MemoryCircuit) -> CircuitIR:
    """Return the memory program lowered once, the way every route lowers it.

    Both grammars read this one lowering: the round-boundary route lowers an
    injected source with it, the gate-bound route indexes the program it returns,
    and the sampler places its faults in the same program. A fault therefore
    cannot be read at one instruction index by the model and placed at another by
    the sampler.
    """

    return _lowered_source(
        memory.source, rounds=memory.rounds, checks=len(memory.code.checks)
    )


def _pauli_fault(channel: KrausChannel) -> tuple[_DataFaultFamily, float]:
    """Return the family a channel names and the rate its Kraus pair carries.

    The rate and the Pauli are both read off the Kraus pair rather than trusted
    from the channel's parameter map, because the pair is what a draw does: a
    channel named ``bit_flip`` whose flipped operator is a different Pauli would
    otherwise be placed as an X fault, and the model would describe a fault the
    circuit does not run. The stabilizer engine makes the same check on the
    sampler's side, so the rate the model states is the rate the sample flips at.
    """

    family = _GATE_BOUND_FAMILIES.get(channel.name)
    if family is None:
        raise CapabilityError(
            "a channel bound to a named gate is placed only as "
            f"{sorted(_GATE_BOUND_FAMILIES)}, and {channel.name!r} is neither: a "
            "depolarizing or damping channel states a mixture rather than one "
            "Pauli fault, and this grammar fails closed rather than placing one "
            "of its terms as the whole fault"
        )
    operators = tuple(channel.kraus)
    if len(operators) != 2:
        raise CapabilityError(
            f"the {channel.name!r} channel carries {len(operators)} Kraus "
            "operator(s); a single-qubit Pauli fault is an identity and one flip"
        )
    flipped = torch.as_tensor(operators[1])
    probability = float(torch.real(torch.trace(flipped.mH @ flipped) / 2).item())
    if not 0.0 <= probability <= 1.0:
        raise ValidationError(
            f"the {channel.name!r} channel flips with probability {probability}, "
            "which is not a probability"
        )
    pauli = torch.tensor(
        _PAULI_MATRICES[family], dtype=flipped.dtype, device=flipped.device
    )
    if not torch.allclose(flipped, math.sqrt(probability) * pauli, atol=1e-6):
        raise CapabilityError(
            f"the {channel.name!r} channel's flipped Kraus operator is not the "
            f"square root of its rate times the {family} family's Pauli, so the "
            "channel does not state the fault its name does"
        )
    return family, probability


def _gate_bound_mechanisms(
    noise: NoiseModel, program: CircuitIR
) -> tuple[_GateBoundMechanism, ...]:
    """Return every fault a gate-bound record places in ``program``, in order.

    The gates a rule attaches to are the record's own business: ``channels_for``
    decides which instruction a rule matches and which qubits of it the channel
    acts on, so the matching rule is stated once and this route reads it rather
    than restating it. What this function decides is the placement -- the fault
    follows the instruction -- and which channels a placement exists for.

    A rule naming a gate the program does not carry places nothing. That is what
    a noise model over a gate set means rather than a fallback, and it is not
    silent about a capability: the channel's name is checked when a rule *does*
    match, so a record stating a channel this grammar cannot place is refused as
    soon as any of its rules matches.

    Raises:
        CapabilityError: A rule names a readout or a preparation, a rule's
            channel is not a single-qubit Pauli fault, a matched instruction is
            the program's last so the fault would sit outside it, or a rule binds
            its one-qubit channel to more than one qubit.
        ValueError: The record itself cannot resolve a rule's channel on a
            matched instruction, which is the record's own refusal and is
            reported as it states itself.
    """

    for rule in noise.rules:
        unplaceable = sorted(set(rule.gate_names) & _UNPLACEABLE_RULE_GATES)
        if unplaceable:
            raise CapabilityError(
                f"a gate-bound noise rule cannot name {unplaceable[0]!r}: this "
                "grammar attaches a fault to the gate a rule names, and a readout "
                "or a preparation is not one. A readout fault is the measurement "
                "family of a PhenomenologicalNoise record, which states which "
                "check it corrupts"
            )
    mechanisms: list[_GateBoundMechanism] = []
    last = len(program.instructions) - 1
    for index, instruction in enumerate(program.instructions):
        for channel, wires in noise.channels_for(instruction):
            family, probability = _pauli_fault(channel)
            if len(wires) != 1:
                raise CapabilityError(
                    f"the {channel.name!r} rule on {instruction.name!r} acts on "
                    f"{len(wires)} qubits; a gate-bound fault is one Pauli on one "
                    "wire, and a two-qubit correlated fault has no placement here"
                )
            if index == last:
                raise CapabilityError(
                    f"the {channel.name!r} rule on {instruction.name!r} matches "
                    "the program's last instruction, so the fault would follow "
                    "the program's end rather than a gate in it"
                )
            if probability:
                mechanisms.append(
                    _GateBoundMechanism(
                        kind=family,
                        after_instruction=index,
                        wire=int(wires[0]),
                        probability=probability,
                    )
                )
    return tuple(mechanisms)


def _spliced_fault(program: CircuitIR, mechanism: _GateBoundMechanism) -> CircuitIR:
    """Return ``program`` with one gate-bound fault spliced in after its gate.

    The fault is the family's gate sequence from :data:`_GATE_BOUND_GATES`,
    written straight after the instruction the rule matched, which is upstream's
    placement: a channel bound to a named gate acts on the state that gate leaves
    behind. The sequence is one instruction for a bit flip and three for a phase
    flip, and the three are an exact identity rather than an approximation.
    """

    position = mechanism.after_instruction + 1
    fault = tuple(
        Instruction(name=name, wires=(mechanism.wire,))
        for name in _GATE_BOUND_GATES[mechanism.kind]
    )
    return CircuitIR(
        n_wires=program.n_wires,
        instructions=(
            *program.instructions[:position],
            *fault,
            *program.instructions[position:],
        ),
        metadata=dict(program.metadata),
    )


def _check_rate_order(checks: tuple[CodeCheck, ...]) -> tuple[int, ...]:
    """Return each declared check's index in the matrix row order.

    ``measurement_flip_per_check`` is indexed the way upstream's ``pm_per_check``
    is and the way :func:`css_code_matrices` builds its rows: every Z-type check
    first, in declaration order, and then every X-type check. A code is free to
    declare its checks in any order -- the rotated surface code declares them in
    lattice order, which interleaves the two types -- so the position a check has
    in that declaration is not the index its rate is stated at, and this is the
    translation between the two.

    **A check that is neither pure X-type nor pure Z-type has no index here.** The
    row order exists because the matrix route splits a code into two families, and
    a mixed check belongs to neither: it carries both factors, so it has no row in
    either family's matrix. Refusing it is the only answer that keeps one vector
    from naming two different checks in the two routes; a caller that wants
    per-check measurement noise on such a code states the uniform rate, whose
    value does not depend on the order at all, which is why the two callers of
    this function consult it only when a per-check vector is actually stated.
    """

    z_positions = [
        position for position, check in enumerate(checks) if check.stabilizer.z_qubits
    ]
    x_positions = [
        position for position, check in enumerate(checks) if check.stabilizer.x_qubits
    ]
    if len(z_positions) + len(x_positions) != len(checks):
        mixed = next(
            check
            for check in checks
            if check.stabilizer.x_qubits and check.stabilizer.z_qubits
        )
        raise ValueError(
            f"check {mixed.index} measures a mixed X-and-Z stabilizer, so it "
            "belongs to neither of the two families a per-check measurement-flip "
            "vector is ordered by and there is no row index it could be stated "
            "at: state the uniform measurement_flip rate for a code like this, "
            "whose value does not depend on the order"
        )
    order = [0] * len(checks)
    for row, position in enumerate((*z_positions, *x_positions)):
        order[position] = row
    return tuple(order)


def _mechanisms(
    circuit: MemoryCircuit, noise: PhenomenologicalNoise
) -> tuple[_Mechanism, ...]:
    """Return every noise location a probability makes possible, in order.

    The three data faults come first -- one per round per data qubit per family,
    walking the code's own qubits in order and each qubit's families in the order
    the record states them -- and measurement faults second, one per round per
    check. Every loop walks the code's own tuples in order, which is the order the
    emitted program measures in. A location whose **own** probability is zero
    cannot flip anything, so it is not enumerated at all: a caller that counts
    mechanisms then counts exactly what can happen, and a per-element vector is how
    a caller states that one location is quiet while its neighbours are not.

    The three fault families are enumerated separately rather than as one
    depolarizing location, because a code-capacity model detects them differently
    and a caller states them as three separate rates. This matches the matrix
    route, which has always enumerated all three; a data location is the same
    location whether the model is built from the program or from the matrices, and
    the two routes must not disagree about which faults exist.

    One qubit can therefore carry up to three mechanisms in a round, one per
    family, and the record states each family's rate separately: a qubit that is
    noisy in one Pauli and quiet in the others is enumerated once, which is what
    makes a bit-flip-only profile cost one mechanism rather than three.

    The probability a location is enumerated with is the element of the vector
    that names it, or the scalar when no vector is stated, and the element is
    resolved through the same accessors the matrix route uses, so the two routes
    cannot read one vector two ways. The per-check vector is indexed in the matrix
    row order rather than in the code's declaration order, which is what
    :func:`_check_rate_order` translates; a uniform per-check rate is order-free,
    so a code that carries a mixed check -- and therefore has no matrix row order
    -- is enumerated by the same loop with the declaration order standing in.

    A code that declares a data qubit twice is refused rather than enumerated
    twice. Its second copy would be the same physical location as the first
    with the same signature, so merging them states one location's rate as two
    independent flips, ``p * (1 - p) + p * (1 - p)``, instead of ``p`` — a wrong
    model with no signal. A code whose checks share an ancilla qubit is refused
    for the same reason: both checks record their syndrome bit at one position
    in the classical register, so the later check's position overwrites the
    earlier one's and each round's detectors read that one bit for both. The
    injection engine refuses that shape too, but only where it injects — its
    anchor is the measurement line, so a model built from data flips alone never
    reaches it — which is why the refusal is stated here, before any mechanism
    is enumerated.

    Raises:
        ValueError: If a per-element rate vector the noise states does not name
            every data qubit or every check the code declares.
    """

    data_wires = circuit.code.data_qubits
    if len(set(data_wires)) != len(data_wires):
        raise ValueError(
            "the code declares a repeated data qubit, so a mechanism at that "
            "location would be enumerated twice and merged with itself"
        )
    checks = tuple(circuit.code.checks)
    ancilla_qubits = [check.ancilla_qubit for check in checks]
    if len(set(ancilla_qubits)) != len(ancilla_qubits):
        repeated = sorted(
            wire for wire in set(ancilla_qubits) if ancilla_qubits.count(wire) > 1
        )
        raise ValueError(
            f"the code declares a repeated check ancilla qubit ({repeated[0]}): "
            "two checks record one syndrome bit, so the model would read the "
            "same bit for both"
        )
    measurement_flip_rates = noise.measurement_flip_rates(num_checks=len(checks))
    # The matrix row order is what a *per-check vector* is stated in. A uniform
    # rate carries one value to every check, so no order can change what it reads,
    # and a code carrying a mixed check has no family order to be read in at all.
    rate_order = (
        _check_rate_order(checks)
        if noise.measurement_flip_per_check
        else tuple(range(len(checks)))
    )
    mechanisms: list[_Mechanism] = []
    families = _data_family_rates(noise, num_qubits=len(data_wires))
    for round_index in range(circuit.rounds):
        for position, wire in enumerate(data_wires):
            for kind, rates in families:
                rate = rates[position]
                if rate:
                    mechanisms.append(
                        _Mechanism(
                            kind=kind,
                            round_index=round_index,
                            wire=wire,
                            probability=rate,
                            fault=_FAULT_LETTER_BY_KIND[kind],
                        )
                    )
    for round_index in range(circuit.rounds):
        for position, check in enumerate(checks):
            rate = measurement_flip_rates[rate_order[position]]
            if rate:
                mechanisms.append(
                    _Mechanism(
                        kind="measurement",
                        round_index=round_index,
                        wire=check.ancilla_qubit,
                        probability=rate,
                    )
                )
    return tuple(mechanisms)


def _checked_round(circuit: MemoryCircuit, round_index: int, *, label: str) -> int:
    """Return ``round_index`` once it names a round this circuit configures."""

    if isinstance(round_index, bool) or not isinstance(round_index, Integral):
        raise TypeError(f"{label} must be an integer")
    if round_index not in range(circuit.rounds):
        raise ValueError(
            f"{label} {round_index} is outside the configured rounds "
            f"0..{circuit.rounds - 1}"
        )
    return int(round_index)


def _checked_wire(wires: tuple[int, ...], wire: int, *, label: str) -> int:
    """Return ``wire`` once the code declares it."""

    if isinstance(wire, bool) or not isinstance(wire, Integral):
        raise TypeError(f"{label} must be an integer")
    if wire not in wires:
        raise ValueError(f"{label} {wire} is not declared by the code")
    return int(wire)


def _single_anchor(source: str, anchor: str, *, label: str) -> int:
    """Return the offset of ``anchor``, refusing a source that repeats or lacks it.

    An anchor that occurs twice cannot say which occurrence the mechanism
    belongs to, so it is refused rather than resolved to the first one.
    """

    occurrences = source.count(anchor)
    if occurrences != 1:
        raise ValueError(
            f"{label} must occur exactly once in the source, found {occurrences}"
        )
    return source.index(anchor)


def _forced_flip(round_index: int, wire: int, *, kind: str = "data") -> str:
    """Return the guarded gates that force one error of ``kind`` on ``wire``.

    The family's gates are :data:`_FAULT_GATES`, emitted in order inside the
    round guard, so a Z or Y fault is the bit flip conjugated by ``h`` rather
    than a second kind of injection. ``kind`` is checked here because the two
    injectors and the callers that hand-build a :class:`_Mechanism` all reach
    this one function, and a misspelled family has to be refused rather than
    silently forced as the bit flip the default names.
    """

    gates = _FAULT_GATES.get(kind)
    if gates is None:
        raise ValueError(
            f"unknown data-fault kind {kind!r}; the families are "
            f"{sorted(_FAULT_GATES)}"
        )
    return "".join(
        [
            f"{_FLIP_INDENT}if round_index == {round_index}:\n",
            *(f"{_FLIP_INDENT}    qp.{gate}(wires={wire})\n" for gate in gates),
        ]
    )


def _inject_data_flip(
    circuit: MemoryCircuit,
    *,
    round_index: int,
    wire: int,
    kind: str | None = None,
    fault: str | None = None,
) -> str:
    """Return ``circuit``'s source with one data fault forced on a data qubit.

    The family is named one of two ways and both are the same statement: ``kind``
    is the noise record's own field -- ``"data"``, ``"phase"`` or ``"both"`` --
    and ``fault`` is the letter of the Pauli it is, ``"x"``, ``"z"`` or ``"y"``.
    Naming neither is the X fault, which is the fault this function was written
    for, and naming both is refused rather than silently preferred. A letter is
    translated through :data:`_FAULT_KIND_BY_LETTER` and the gates come from
    :data:`_FAULT_GATES` either way, so the two spellings cannot disagree about
    which fault is forced.

    The fault is the one named and not the one the experiment's readout basis makes
    visible. Those are the same question only in the Z basis: the model states an X
    fault, a Z fault and a Y fault as three independent locations, and each is
    injected as itself whichever basis the qubit is read out in. Substituting one
    for another here would make ``data_flip`` open an X-basis experiment's
    observable, which is a claim the noise record does not make and the caller
    cannot see.

    The flip is guarded by ``if round_index == <round_index>:`` and inserted at
    the start of that round, before any of the round's CNOTs, so the error opens
    the frame the round's detectors compare against.
    """

    checked_round = _checked_round(circuit, round_index, label="data-flip round")
    checked_wire = _checked_wire(circuit.code.data_qubits, wire, label="data-flip wire")
    if fault is not None:
        if kind is not None:
            raise ValueError(
                "a data fault is named either by its family or by its letter, "
                f"not by both ({kind!r} and {fault!r})"
            )
        kind = _FAULT_KIND_BY_LETTER.get(fault)
        if kind is None:
            raise ValueError(
                f"unknown data fault {fault!r}: the model states an X, a Z and a "
                "Y fault, spelled 'x', 'z' and 'y'"
            )
    elif kind is None:
        kind = "data"
    offset = _single_anchor(
        circuit.source, _ROUND_LOOP_ANCHOR, label="the round loop anchor"
    )
    end = offset + len(_ROUND_LOOP_ANCHOR)
    return (
        f"{circuit.source[:end]}"
        f"{_forced_flip(checked_round, checked_wire, kind=kind)}"
        f"{circuit.source[end:]}"
    )


def _inject_measurement_flip(
    circuit: MemoryCircuit, *, round_index: int, ancilla_qubit: int
) -> str:
    """Return ``circuit``'s source with one check measurement forced to flip.

    The flip is an ``X`` on the ancilla immediately before the check measures
    it, guarded by ``if round_index == <round_index>:``. A check's ancilla qubit
    is unique to it, so the anchor names exactly one check; a source where that
    line is missing or repeated is refused rather than injected into the wrong
    check.
    """

    checked_round = _checked_round(circuit, round_index, label="measurement-flip round")
    checked_wire = _checked_wire(
        circuit.code.ancilla_qubits, ancilla_qubit, label="measurement-flip ancilla"
    )
    offset = _single_anchor(
        circuit.source,
        f"{_FLIP_INDENT}last = qp.measure(wires={checked_wire})\n",
        label="the check measurement anchor",
    )
    return (
        f"{circuit.source[:offset]}"
        f"{_forced_flip(checked_round, checked_wire)}"
        f"{circuit.source[offset:]}"
    )


def _forced_signature(
    circuit: MemoryCircuit, source: str
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the detectors and observables that ``source``'s forced error flips.

    The source is lowered the way the experiment's own source is lowered and read
    by :func:`_executed_flip_set`, so a fault injected into the source is read off
    the program that source lowers to rather than off a program of its own.
    """

    lowered = _lowered_source(
        source, rounds=circuit.rounds, checks=len(circuit.code.checks)
    )
    return _executed_flip_set(circuit, lowered)


def _lowered_source(source: str, *, rounds: int, checks: int) -> CircuitIR:
    """Return ``source`` lowered with the round count and measurement budget given.

    One lowering is stated here rather than at each caller because a fault's
    signature is read off the *lowered* program: an injected source and the
    experiment's own source have to lower by the same rule for the two to be
    comparable, and the measurement budget is what the round loop's readouts are
    counted against.
    """

    program = capture_source(source, (INDEX,))
    return lower_dynamic_program(
        program,
        (rounds,),
        max_dynamic_measurements=rounds * checks,
    ).circuit


def _executed_flip_set(
    circuit: MemoryCircuit, program: CircuitIR
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the detectors and observables one program's forced error flips.

    The program is executed twice and the two shots must agree on the *flip set*:
    a mechanism whose signature depends on the trajectory is not a Pauli
    mechanism in the reference gate set, so it is refused instead of contributing
    a signature. The shot is read through the layouts, never off the raw register:
    a detector XORs the measurements its parity names, and an observable XORs the
    terminal samples over the support of its Pauli.

    The register bits themselves need not agree, and for a code with X-type
    checks they do not: an X-type ancilla is prepared in ``|+>``, so its
    round-zero outcome is random. What the model records is the flip a mechanism
    causes relative to the noiseless outcome, and that is the parity the layouts
    define — deterministic for the steady-state X-type detectors precisely
    because the two rounds it compares were projected into the same eigenstate.
    Requiring the raw bits to agree would refuse every code with an X-type check,
    including the rotated surface code, for a randomness the model does not see.

    A syndrome measurement is read at ``round_index * len(checks) +
    position_in_checks``, where the position is the check's index in the code's
    ``checks`` tuple. That is the order the emitted loop measures in, which is
    what the classical register records; ``CodeCheck.index`` is a label the code
    chooses and need not be that order.
    """

    execution = execute_hybrid_dynamic_session(
        program, shots=2, seed=0, strategy="trajectory"
    )
    classical: list[list[int]] = execution.classical_bits.tolist()
    samples: list[list[int]] = execution.samples.tolist()
    positions = {
        check.ancilla_qubit: position
        for position, check in enumerate(circuit.code.checks)
    }
    checks = len(circuit.code.checks)

    def measurement_bit(
        classical_row: list[int], sample_row: list[int], reference: MeasurementRef
    ) -> int:
        """Return the recorded bit one layout reference names.

        A syndrome reference is read at the position of the check that owns its
        ancilla, so an ancilla no check owns has no recorded bit to read. A
        layout may name one — ``MemoryCircuit`` ties the reference to the code's
        declared ancilla qubits, not to its checks — so the lookup states the
        failure instead of raising a bare ``KeyError``.
        """

        if reference.round_index is None:
            return sample_row[reference.qubit]
        position = positions.get(reference.qubit)
        if position is None:
            raise ValueError(
                f"detector syndrome measurement names ancilla qubit "
                f"{reference.qubit}, which no check owns"
            )
        return classical_row[reference.round_index * checks + position]

    def flip_set(
        classical_row: list[int], sample_row: list[int]
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        detectors = tuple(
            detector.index
            for detector in circuit.detectors.detectors
            if sum(
                measurement_bit(classical_row, sample_row, reference)
                for reference in detector.parity
            )
            % 2
            == 1
        )
        observables = tuple(
            observable.index
            for observable in circuit.observables.observables
            if sum(sample_row[wire] for wire in observable.pauli.support) % 2 == 1
        )
        return detectors, observables

    signature = flip_set(classical[0], samples[0])
    if signature != flip_set(classical[1], samples[1]):
        raise ValueError(
            "the forced error's detector and observable flips are not "
            "deterministic across trajectories, so it is not a Pauli mechanism "
            "in the reference gate set"
        )
    return signature
