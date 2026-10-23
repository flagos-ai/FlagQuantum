"""Forced memory-circuit fault injection, lowering and reference execution.

Source gates retain their original order: X is a bit flip, H-X-H is Z, and
X-H-X-H is Y up to an unobservable global phase. DEM construction and sampling
consume the same fault vocabulary and lowering entry points.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from types import MappingProxyType
from typing import Literal

from ...compiler._hybrid import INDEX, capture_source, lower_dynamic_program
from ...core.ir import CircuitIR
from ...runtime.dynamic.hybrid_session import execute_hybrid_dynamic_session
from ..circuits.memory import MeasurementRef, MemoryCircuit
from ..codes._compat import _retain_module

# One shared vocabulary for the physical fault families enumerated by both routes.
_DataFaultFamily = Literal["data", "phase", "both"]

_MechanismKind = Literal["data", "phase", "both", "measurement"]

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


_retain_module(
    "flagquantum.qec.dem_construction",
    _lowered_program,
    _checked_round,
    _checked_wire,
    _single_anchor,
    _forced_flip,
    _inject_data_flip,
    _inject_measurement_flip,
    _forced_signature,
    _lowered_source,
    _executed_flip_set,
)
