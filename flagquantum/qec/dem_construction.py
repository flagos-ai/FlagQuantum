"""Sources of the physical mechanisms a detector error model describes.

A :class:`~flagquantum.qec.DetectorErrorModel` states which detectors and which
observables each mechanism flips. This module owns where those mechanisms come
from, and it knows two descriptions of an experiment. Each one reaches the same
model shape by a different route, and neither is the authority for the other.

A **circuit** description is forced. :func:`_memory_circuit_entries` enumerates
the locations a :class:`~flagquantum.qec.PhenomenologicalNoise` record
configures, injects one at a time into the source program the circuit carries,
and reads the resulting flip set off the circuit's own detector and observable
layouts. The signature is therefore derived from the program rather than
asserted about it, and a mechanism whose flip set is not deterministic across
trajectories is refused.

A **matrix** description is read. :func:`_code_matrix_entries` takes the
parity-check matrix a code is defined by and the matrix of its logical
operators, and derives the same signatures combinatorially; nothing is lowered
or executed. Its detector geometry is consequently the code-capacity one:
``num_rounds * num_checks`` detectors, each round compared against the round
before it, with no terminal data readout. That is deliberately not the geometry
:func:`_memory_circuit_entries` produces, because a memory circuit measures the
data qubits and so ends in terminal detectors. The difference belongs to the
description, not to the model, and a test pins each geometry to its own route.

Both routes carry the same family of faults: a data wire's flip at a round
boundary is the bit flip :class:`~flagquantum.qec.PhenomenologicalNoise` names,
so it shows up in the Z-type detectors and in the Z-type logical operators. The
phase-flip family upstream's ``hx`` and ``lx`` matrices describe, and upstream's
independent ``px``, ``py``, ``pz`` and ``pm`` rates, are not expressible through
one data-flip scalar and one measurement-flip scalar. That limit is recorded in
the alignment contract rather than approximated here.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import torch

from ..compiler._hybrid import INDEX, capture_source, lower_dynamic_program
from ..runtime.dynamic.hybrid_session import execute_hybrid_dynamic_session
from .circuit import MeasurementRef, MemoryCircuit
from .codes import StabilizerCode
from .noise import PhenomenologicalNoise

__all__ = ("code_matrices",)

Entry = tuple[float, tuple[int, ...], tuple[int, ...]]
# One mechanism: its rate, the detectors it flips, and the observables it flips.


def _memory_circuit_entries(
    circuit: MemoryCircuit, noise: PhenomenologicalNoise
) -> tuple[int, int, tuple[Entry, ...]]:
    """Return the model shape and every mechanism a memory circuit contains.

    Each location :func:`_mechanisms` enumerates is forced through the circuit on
    its own and its signature is read off the layouts, so the model is derived
    from the program rather than asserted about it. The refusals of the injection
    engine reach the caller unchanged: a hand-built circuit whose source does not
    match its layouts fails closed with a stated reason instead of building a
    model that misdescribes it.

    The detector and observable counts come from the circuit's own layouts rather
    than from a count re-derived from the code, so the returned shape and the
    returned signatures are read from one authority.
    """

    if not isinstance(circuit, MemoryCircuit):
        raise TypeError("circuit must be a MemoryCircuit")
    if not isinstance(noise, PhenomenologicalNoise):
        raise TypeError("noise must be a PhenomenologicalNoise")
    entries: list[Entry] = []
    for mechanism in _mechanisms(circuit, noise):
        if mechanism.kind == "data":
            source = _inject_data_flip(
                circuit,
                round_index=mechanism.round_index,
                wire=mechanism.wire,
            )
        elif mechanism.kind == "measurement":
            source = _inject_measurement_flip(
                circuit,
                round_index=mechanism.round_index,
                ancilla_wire=mechanism.wire,
            )
        else:
            # Unreachable through ``_mechanisms``, which sets the field from the
            # annotation; stated for a caller that builds records itself, where a
            # bare ``else`` would read an unknown kind as a measurement flip
            # whenever the wire is a declared ancilla.
            raise ValueError(f"unknown mechanism kind {mechanism.kind!r}")
        detectors, observables = _forced_signature(circuit, source)
        entries.append((mechanism.probability, detectors, observables))
    return len(circuit.detectors), len(circuit.observables), tuple(entries)


def _binary_matrix(matrix: object, *, name: str) -> torch.Tensor:
    """Return ``matrix`` as a two-dimensional integer tensor of zeros and ones.

    A floating-point matrix is refused rather than compared against a tolerance:
    a caller who reached for one is describing rates, not a support, and rounding
    it here would decide which checks a fault triggers.
    """

    if not isinstance(matrix, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if matrix.dim() != 2:
        raise ValueError(
            f"{name} must be two-dimensional -- one row per check and one column "
            f"per data qubit -- but has {matrix.dim()} dimension(s)"
        )
    if matrix.is_floating_point() or matrix.is_complex():
        raise TypeError(
            f"{name} must be a binary matrix, so it cannot carry a "
            f"{matrix.dtype} element"
        )
    values = matrix.to(torch.int64)
    if values.numel() and bool(((values != 0) & (values != 1)).any()):
        raise ValueError(f"{name} must contain only zeros and ones")
    return values


def _column_supports(
    matrix: torch.Tensor, num_qubits: int
) -> tuple[tuple[int, ...], ...]:
    """Return the rows each column of ``matrix`` selects, in ascending order."""

    selected = matrix.to(torch.bool)
    return tuple(
        tuple(int(row) for row in selected[:, qubit].nonzero().flatten())
        for qubit in range(num_qubits)
    )


def _code_matrix_entries(
    *,
    hz: torch.Tensor,
    noise: PhenomenologicalNoise,
    lz: torch.Tensor | None = None,
    num_rounds: int = 1,
) -> tuple[int, int, tuple[Entry, ...]]:
    """Return the model shape and every mechanism a code's matrices imply.

    The matrices are read, not simulated. ``hz`` has one row per Z-type check and
    one column per data qubit, and ``hz[k, q]`` is one when a bit flip on qubit
    ``q`` flips check ``k``. ``lz`` uses the same convention for the logical
    operators, with ``lz[r, q]`` one when a bit flip on qubit ``q`` flips logical
    observable ``r``; it may be omitted, and then no mechanism flips an
    observable.

    Every data qubit ``noise.data_flip`` reaches contributes one mechanism per
    round, and every check ``noise.measurement_flip`` reaches contributes one
    measurement mechanism per round. The model has no terminal data readout, so
    a round's mechanisms are read against that round's detector band and the band
    after it, and the final round has no band after it. The detector geometry is
    therefore the code-capacity one, which is deliberately not the geometry
    :func:`_memory_circuit_entries` derives from a memory circuit; the module
    docstring states the difference.

    An entry that flips nothing at all -- a qubit outside every check and outside
    every logical operator -- is not a mechanism a model can store, and an entry
    whose signature another entry already has is one fault to a decoder. Both are
    resolved where every construction route resolves them, in
    ``DetectorErrorModel._merge_mechanisms``, by the independent-parity rule.

    Raises:
        TypeError: If a matrix is not a binary integer tensor or ``noise`` is not
            a :class:`~flagquantum.qec.PhenomenologicalNoise`.
        ValueError: If ``num_rounds`` is below one, if a matrix is not
            two-dimensional, if ``lz`` disagrees with ``hz`` about how many data
            qubits the code has, or if ``hz`` declares no check -- a model needs
            at least one detector.
    """

    if isinstance(num_rounds, bool) or not isinstance(num_rounds, Integral):
        raise TypeError("num_rounds must be an integer")
    if num_rounds < 1:
        raise ValueError("num_rounds must be at least one")
    if not isinstance(noise, PhenomenologicalNoise):
        raise TypeError("noise must be a PhenomenologicalNoise")
    rounds = int(num_rounds)

    checks = _binary_matrix(hz, name="hz")
    num_checks = int(checks.shape[0])
    num_qubits = int(checks.shape[1])
    if num_qubits == 0:
        raise ValueError(
            "hz declares no data qubit, so it declares no check a fault could "
            "flip and the model would have no detector to describe"
        )
    if num_checks == 0:
        raise ValueError(
            f"hz declares {num_qubits} data qubits but no check, so the model "
            "would have no detector to describe"
        )
    logicals = (
        torch.zeros((0, num_qubits), dtype=torch.int64)
        if lz is None
        else _binary_matrix(lz, name="lz")
    )
    if int(logicals.shape[1]) != num_qubits:
        raise ValueError(
            f"lz has {int(logicals.shape[1])} columns but hz has {num_qubits}; a "
            "logical operator matrix must be indexed by the same data qubits"
        )

    hz_columns = _column_supports(checks, num_qubits)
    lz_columns = _column_supports(logicals, num_qubits)
    entries: list[Entry] = []
    for round_index in range(rounds):
        band = round_index * num_checks
        following = band + num_checks
        has_next = round_index + 1 < rounds
        if noise.data_flip:
            for qubit in range(num_qubits):
                detectors = [band + row for row in hz_columns[qubit]]
                if has_next:
                    detectors.extend(following + row for row in hz_columns[qubit])
                entries.append((noise.data_flip, tuple(detectors), lz_columns[qubit]))
        if noise.measurement_flip:
            for check in range(num_checks):
                detectors = [band + check]
                if has_next:
                    detectors.append(following + check)
                entries.append((noise.measurement_flip, tuple(detectors), ()))
    return rounds * num_checks, int(logicals.shape[0]), tuple(entries)


def code_matrices(code: StabilizerCode) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a code record's Z-type check matrix and Z-type logical matrix.

    Column ``q`` of both matrices is the ``q``-th wire :attr:`data_wires` names,
    so a code is free to declare its data qubits on any wires. The rows follow
    the code's own declaration order, and an X-type check or an X-type logical
    operator contributes no row at all rather than an empty one: a fault this
    route can express never flips it, so an all-zero row would declare an
    observable no mechanism ever reports. That is the pair a bit-flip fault
    probes -- an X fault on a data wire flips exactly the Z-type checks whose
    support holds that wire, and exactly the logical operators whose Z support
    holds it -- and it is the input the matrix route of construction takes, so a
    code described by its stabilizers reaches a detector error model without a
    circuit being written for it.

    Raises:
        TypeError: If ``code`` is not a :class:`~flagquantum.qec.StabilizerCode`.
        ValueError: If the code declares no Z-type check -- read as Z memory it
            would have no detector -- or if a check's support names a wire the
            code does not declare as a data wire.
    """

    if not isinstance(code, StabilizerCode):
        raise TypeError("code must be a StabilizerCode")
    data_wires = tuple(code.data_wires)
    columns = {wire: index for index, wire in enumerate(data_wires)}
    checks = tuple(check for check in code.checks if check.stabilizer.z_wires)
    if not checks:
        raise ValueError(
            f"{type(code).__name__} declares no Z-type check, so read as Z "
            "memory it has no detector for a fault to flip"
        )
    hz = torch.zeros((len(checks), len(data_wires)), dtype=torch.int64)
    for row, check in enumerate(checks):
        for wire in check.stabilizer.support:
            if wire not in columns:
                raise ValueError(
                    f"check {check.index} acts on wire {wire}, which "
                    f"{type(code).__name__} does not declare as a data wire"
                )
            hz[row, columns[wire]] = 1
    observables = tuple(
        observable for observable in code.logical_observables if observable.z_wires
    )
    lz = torch.zeros((len(observables), len(data_wires)), dtype=torch.int64)
    for row, observable in enumerate(observables):
        for wire in observable.z_wires:
            if wire not in columns:
                raise ValueError(
                    f"logical observable {row} acts on wire {wire}, which "
                    f"{type(code).__name__} does not declare as a data wire"
                )
            lz[row, columns[wire]] = 1
    return hz, lz


_ROUND_LOOP_ANCHOR = "    for round_index in range(rounds):\n"
_FLIP_INDENT = "        "


@dataclass(frozen=True)
class _Mechanism:
    """One physical noise location, by kind, round, and wire.

    The record carries the coordinates the matching injector needs rather than a
    rendered source. A caller that fires a set of mechanisms at once has to
    re-inject them into the one program it executes, and it cannot rebuild a
    source from a string alone; carrying coordinates also keeps injection in
    exactly one place. ``kind`` is ``"data"`` for a flip on a data wire, where
    ``wire`` is that data wire, and ``"measurement"`` for a flip on a check's
    syndrome measurement, where ``wire`` is that check's ancilla. The kind is a
    literal rather than a free string, so a caller that builds a record by hand
    is told at the type checker which two flips exist.
    """

    kind: Literal["data", "measurement"]
    round_index: int
    wire: int
    probability: float


def _mechanisms(
    circuit: MemoryCircuit, noise: PhenomenologicalNoise
) -> tuple[_Mechanism, ...]:
    """Return every noise location a probability makes possible, in order.

    Data flips come first — one per round per data wire — and measurement flips
    second, one per round per check. Both loops walk the code's own tuples in
    order, which is the order the emitted program measures in. A location whose
    probability is zero cannot flip anything, so it is not enumerated at all: a
    caller that counts mechanisms then counts exactly what can happen.

    A code that declares a data wire twice is refused rather than enumerated
    twice. Its second copy would be the same physical location as the first
    with the same signature, so merging them states one location's rate as two
    independent flips, ``p * (1 - p) + p * (1 - p)``, instead of ``p`` — a wrong
    model with no signal. A code whose checks share an ancilla wire is refused
    for the same reason: both checks record their syndrome bit at one position
    in the classical register, so the later check's position overwrites the
    earlier one's and each round's detectors read that one bit for both. The
    injection engine refuses that shape too, but only where it injects — its
    anchor is the measurement line, so a model built from data flips alone never
    reaches it — which is why the refusal is stated here, before any mechanism
    is enumerated.
    """

    data_wires = circuit.code.data_wires
    if len(set(data_wires)) != len(data_wires):
        raise ValueError(
            "the code declares a repeated data wire, so a mechanism at that "
            "location would be enumerated twice and merged with itself"
        )
    ancilla_wires = [check.ancilla_wire for check in circuit.code.checks]
    if len(set(ancilla_wires)) != len(ancilla_wires):
        repeated = sorted(
            wire for wire in set(ancilla_wires) if ancilla_wires.count(wire) > 1
        )
        raise ValueError(
            f"the code declares a repeated check ancilla wire ({repeated[0]}): "
            "two checks record one syndrome bit, so the model would read the "
            "same bit for both"
        )
    mechanisms: list[_Mechanism] = []
    if noise.data_flip:
        for round_index in range(circuit.rounds):
            for wire in data_wires:
                mechanisms.append(
                    _Mechanism(
                        kind="data",
                        round_index=round_index,
                        wire=wire,
                        probability=noise.data_flip,
                    )
                )
    if noise.measurement_flip:
        for round_index in range(circuit.rounds):
            for check in circuit.code.checks:
                mechanisms.append(
                    _Mechanism(
                        kind="measurement",
                        round_index=round_index,
                        wire=check.ancilla_wire,
                        probability=noise.measurement_flip,
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


def _forced_flip(round_index: int, wire: int) -> str:
    """Return the guarded single-qubit ``X`` that forces one error."""

    return (
        f"{_FLIP_INDENT}if round_index == {round_index}:\n"
        f"{_FLIP_INDENT}    qp.X(wires={wire})\n"
    )


def _inject_data_flip(circuit: MemoryCircuit, *, round_index: int, wire: int) -> str:
    """Return ``circuit``'s source with one ``X`` forced on a data wire.

    The flip is guarded by ``if round_index == <round_index>:`` and inserted at
    the start of that round, before any of the round's CNOTs, so the error opens
    the frame the round's detectors compare against.
    """

    checked_round = _checked_round(circuit, round_index, label="data-flip round")
    checked_wire = _checked_wire(circuit.code.data_wires, wire, label="data-flip wire")
    offset = _single_anchor(
        circuit.source, _ROUND_LOOP_ANCHOR, label="the round loop anchor"
    )
    end = offset + len(_ROUND_LOOP_ANCHOR)
    return (
        f"{circuit.source[:end]}"
        f"{_forced_flip(checked_round, checked_wire)}"
        f"{circuit.source[end:]}"
    )


def _inject_measurement_flip(
    circuit: MemoryCircuit, *, round_index: int, ancilla_wire: int
) -> str:
    """Return ``circuit``'s source with one check measurement forced to flip.

    The flip is an ``X`` on the ancilla immediately before the check measures
    it, guarded by ``if round_index == <round_index>:``. A check's ancilla wire
    is unique to it, so the anchor names exactly one check; a source where that
    line is missing or repeated is refused rather than injected into the wrong
    check.
    """

    checked_round = _checked_round(circuit, round_index, label="measurement-flip round")
    checked_wire = _checked_wire(
        circuit.code.ancilla_wires, ancilla_wire, label="measurement-flip ancilla"
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

    The source is lowered and executed twice, and the two shots must agree on
    the *flip set*: a mechanism whose signature depends on the trajectory is not
    a Pauli mechanism in the reference gate set, so it is refused instead of
    contributing a signature. The shot is read through the layouts, never off the
    raw register: a detector XORs the measurements its parity names, and an
    observable XORs the terminal samples over the support of its Pauli.

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

    program = capture_source(source, (INDEX,))
    lowered = lower_dynamic_program(
        program,
        (circuit.rounds,),
        max_dynamic_measurements=circuit.rounds * len(circuit.code.checks),
    )
    execution = execute_hybrid_dynamic_session(
        lowered.circuit, shots=2, seed=0, strategy="trajectory"
    )
    classical: list[list[int]] = execution.classical_bits.tolist()
    samples: list[list[int]] = execution.samples.tolist()
    positions = {
        check.ancilla_wire: position
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
        declared ancilla wires, not to its checks — so the lookup states the
        failure instead of raising a bare ``KeyError``.
        """

        if reference.round_index is None:
            return sample_row[reference.wire]
        position = positions.get(reference.wire)
        if position is None:
            raise ValueError(
                f"detector syndrome measurement names ancilla wire "
                f"{reference.wire}, which no check owns"
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
