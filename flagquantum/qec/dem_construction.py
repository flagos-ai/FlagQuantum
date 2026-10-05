"""Sources of the physical mechanisms a detector error model describes.

A :class:`~flagquantum.qec.DetectorErrorModel` states which detectors and which
observables each mechanism flips. This module owns where those mechanisms come
from, and it knows two descriptions of an experiment. Each one reaches the same
model shape by a different route, and neither is the authority for the other.

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

A **matrix** description is read. :func:`_code_matrix_entries` takes the
parity-check matrix a code is defined by and the matrix of its logical
operators, and derives the same signatures combinatorially; nothing is lowered
or executed. Its detector geometry is consequently the code-capacity one:
``num_rounds * num_checks`` detectors, each round compared against the round
before it, with no terminal data readout. That is deliberately not the geometry
:func:`_memory_circuit_entries` produces, because a memory circuit measures the
data qubits and so ends in terminal detectors. The difference belongs to the
description, not to the model, and a test pins each geometry to its own route.

Both routes carry the same family of faults. On the matrix route the three data
faults are the three Pauli errors: an X fault reaches the Z-type detectors and
the Z-type logical operators, a Z fault reaches the X-type detectors and the
X-type logical operators, and a Y fault reaches both. Each data fault in round
``r`` shows up in the detector band of round ``r`` and in the band after it, and
a measurement fault shows up in its own check's detector and in the next round's,
which is the code-capacity experiment stated as detector differences.

The circuit route carries the same three data faults. Which detectors and which
observables a fault reaches is decided by the frame the experiment is prepared
and read out in, and the frame is stated by
:attr:`~flagquantum.qec.MemoryCircuit.x_readout_wires`: an X fault flips the
Z-type checks' syndrome and the Z-basis readout, a Z fault flips the X-type
checks' syndrome and the X-basis readout, and a Y fault flips both. A fault
injected at the top of round ``r`` stays on the data wire, so it opens the
detector band of round ``r`` and cancels in every band after it; a syndrome
measurement flip opens its own check's band at round ``r`` and at ``r + 1``. The
terminal detectors exist only for the checks the preparation state pins, which
is the same set the layout records.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import torch

from ..compiler._hybrid import INDEX, capture_source, lower_dynamic_program
from ..runtime.dynamic.hybrid_session import execute_hybrid_dynamic_session
from .circuit import MeasurementRef, MemoryCircuit
from .codes import CodeCheck, StabilizerCode
from .noise import PhenomenologicalNoise

__all__ = ("CssCodeMatrices", "css_code_matrices")

Entry = tuple[float, tuple[int, ...], tuple[int, ...]]
# One mechanism: its rate, the detectors it flips, and the observables it flips.


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

    if any(mechanism.kind == "data" for mechanism in mechanisms):
        _single_anchor(
            circuit.source, _ROUND_LOOP_ANCHOR, label="the round loop anchor"
        )
    if any(mechanism.kind == "measurement" for mechanism in mechanisms):
        for check in circuit.code.checks:
            _single_anchor(
                circuit.source,
                f"{_FLIP_INDENT}last = qp.measure(wires={check.ancilla_wire})\n",
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
    wires, not to its checks — so the failure is stated rather than left to a
    lookup that finds nothing and reports an even parity.
    """

    for detector in circuit.detectors.detectors:
        for reference in detector.parity:
            if reference.round_index is not None and reference.wire not in owned:
                raise ValueError(
                    f"detector syndrome measurement names ancilla wire "
                    f"{reference.wire}, which no check owns"
                )


def _flipped_measurements(
    circuit: MemoryCircuit, mechanism: _Mechanism
) -> frozenset[tuple[int | None, int]]:
    """Return the layout references one noise location flips.

    A reference is named the way the layouts name it: a syndrome measurement by
    its round and its check's ancilla, and a terminal data sample by ``None`` and
    the data wire. A mechanism is a classical bit-flip set over those references,
    and a detector or an observable fires exactly when its parity meets that set
    in an odd number of places — so this is the whole of the derivation, and
    everything after it is a parity count.

    A **measurement** mechanism flips the one syndrome bit its check records that
    round. A **data** mechanism flips a syndrome bit in every round from the round
    it is injected in to the last, for every check the fault anticommutes with on
    that wire, and it flips the terminal sample of the wire when the fault
    anticommutes with the basis the wire is read out in. Which checks and which
    readouts those are is read from the code's check types and the circuit's
    ``x_readout_wires``; a Z-type check measures a Z stabilizer and so sees the
    faults that anticommute with Z, and a wire read out in X sees the same.

    The fault stays on the data wire, so the run of flipped syndrome bits starts at
    the injected round and does not stop; that is why a data fault opens one
    detector band and cancels in the rest.
    """

    if mechanism.kind == "measurement":
        return frozenset({(mechanism.round_index, mechanism.wire)})
    rotated = frozenset(circuit.x_readout_wires)
    if mechanism.fault == "y":
        flips_readout = True
    elif mechanism.fault == "z":
        flips_readout = mechanism.wire in rotated
    else:
        flips_readout = mechanism.wire not in rotated
    flips: set[tuple[int | None, int]] = set()
    if flips_readout:
        flips.add((None, mechanism.wire))
    for check in circuit.code.checks:
        if mechanism.wire not in check.stabilizer.support:
            continue
        x_type = bool(check.stabilizer.x_wires)
        if mechanism.fault == "y":
            coupled = True
        elif mechanism.fault == "z":
            coupled = x_type
        else:
            coupled = not x_type
        if not coupled:
            continue
        flips.update(
            (round_index, check.ancilla_wire)
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
    list is the one the circuit states: a terminal detector reads the data wires
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
            if (reference.round_index, reference.wire) in flips
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
            if (reference.round_index, reference.wire) in flips
        )
        % 2
        == 1
    )
    return detectors, observables


def _memory_circuit_entries(
    circuit: MemoryCircuit, noise: PhenomenologicalNoise
) -> tuple[int, int, tuple[Entry, ...]]:
    """Return the model shape and every mechanism a memory circuit contains.

    Each location :func:`_mechanisms` enumerates has its signature derived from
    the circuit's own layouts by :func:`_flipped_measurements`, so the model is a
    reading of the record the circuit states rather than a claim about it. The
    derivation needs no execution, so the width of the code does not bound it.

    The source is still held against the layouts. Nothing is injected into it any
    more, so a source that does not carry the round loop or a check's measurement,
    or a layout that reads a syndrome bit no check records, would otherwise build
    a model that misdescribes the program it was handed. Both are refused with the
    reason the injection engine used to state, and only the parts of the source
    the enumerated mechanisms actually read are required: a circuit modelled from
    data faults alone needs no measurement anchor.

    The detector and observable counts come from the circuit's own layouts rather
    than from a count re-derived from the code, so the returned shape and the
    returned signatures are read from one authority.
    """

    if not isinstance(circuit, MemoryCircuit):
        raise TypeError("circuit must be a MemoryCircuit")
    if not isinstance(noise, PhenomenologicalNoise):
        raise TypeError("noise must be a PhenomenologicalNoise")
    mechanisms = _mechanisms(circuit, noise)
    if not mechanisms:
        return len(circuit.detectors), len(circuit.observables), ()
    _require_source_anchors(circuit, mechanisms)
    owned = {check.ancilla_wire for check in circuit.code.checks}
    _require_owned_syndrome_references(circuit, owned=owned)
    entries: list[Entry] = []
    for mechanism in mechanisms:
        if mechanism.kind not in ("data", "measurement"):
            # Unreachable through ``_mechanisms``, which sets the field from the
            # annotation; stated for a caller that builds records itself, where a
            # bare ``else`` would read an unknown kind as a measurement flip
            # whenever the wire is a declared ancilla.
            raise ValueError(f"unknown mechanism kind {mechanism.kind!r}")
        entries.append(
            (
                mechanism.probability,
                *_layout_signature(circuit, mechanism),
            )
        )
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
    matrices: CssCodeMatrices,
    noise: PhenomenologicalNoise,
    num_rounds: int = 1,
) -> tuple[int, int, tuple[Entry, ...]]:
    """Return the model shape and every mechanism a code's matrices imply.

    The matrices are read, not simulated. Each has one column per data qubit, and
    a fault is read as the columns it selects: ``hz[k, q]`` is one when an X fault
    on qubit ``q`` flips Z-type check ``k``, ``hx[k, q]`` is one when a Z fault
    flips X-type check ``k``, and the two logical matrices state the same relation
    for the observable rows.

    A round's detector band holds one row per check, Z-type checks first, and the
    band is compared against the band before it, so a data fault lands in its own
    round's band and in the band after it and the final round has no band after
    it. A measurement fault lands in its own check's row and in the next round's,
    and flips no observable, because the logical measurement is taken once at the
    end of the experiment.

    The model describes the code-capacity experiment: the data begins in the
    code's ``+1`` eigenspace of every check, so every round-zero check is
    deterministic and there is no preparation-fault family. A code whose
    preparation is not a code state is the circuit route's business, not this
    one's, and it is there that a check whose round-zero outcome is a coin toss
    has to be handled.

    Every data qubit the noise reaches contributes one mechanism per round per
    fault family, and every check the noise reaches contributes one measurement
    mechanism per round, where "reaches" is the element's own effective rate: a
    per-qubit or per-check vector states one rate per location, and a location
    whose rate is zero is not enumerated at all. An entry that flips nothing at
    all -- a qubit outside every check and outside every logical operator -- is
    not a mechanism a model can store, and an entry whose signature another entry
    already has is one fault to a decoder. Both are resolved where every
    construction route resolves them, in ``DetectorErrorModel._merge_mechanisms``,
    by the independent-parity rule.

    Raises:
        TypeError: If ``noise`` is not a
            :class:`~flagquantum.qec.PhenomenologicalNoise`.
        ValueError: If ``num_rounds`` is below one, if the matrices declare no
            check -- a model needs at least one detector -- or if a per-element
            rate vector the noise states does not name every data qubit or every
            check the matrices declare.
    """

    if isinstance(num_rounds, bool) or not isinstance(num_rounds, Integral):
        raise TypeError("num_rounds must be an integer")
    if num_rounds < 1:
        raise ValueError("num_rounds must be at least one")
    if not isinstance(noise, PhenomenologicalNoise):
        raise TypeError("noise must be a PhenomenologicalNoise")
    if not isinstance(matrices, CssCodeMatrices):
        raise TypeError("matrices must be a CssCodeMatrices")
    rounds = int(num_rounds)

    # ``CssCodeMatrices`` has already refused a disagreement about the number of
    # columns, so one column count describes all four matrices here.
    num_qubits = matrices.num_qubits
    num_z_checks = matrices.num_z_checks
    num_x_checks = matrices.num_x_checks
    num_checks = num_z_checks + num_x_checks
    if num_checks == 0:
        raise ValueError(
            f"the matrices declare {num_qubits} data qubits but no check, so the "
            "model would have no detector to describe"
        )
    num_z_logicals = matrices.num_z_logicals

    # A vector's length is stated against the code, and the matrices are the
    # code here: one column per data qubit and one row per check, Z-type checks
    # first, which is the order `measurement_flip_per_check` is indexed in.
    data_flip_rates = noise.data_flip_rates(num_qubits=num_qubits)
    phase_flip_rates = noise.phase_flip_rates(num_qubits=num_qubits)
    both_flip_rates = noise.both_flip_rates(num_qubits=num_qubits)
    measurement_flip_rates = noise.measurement_flip_rates(num_checks=num_checks)

    hz_columns = _column_supports(matrices.hz, num_qubits)
    hx_columns = _column_supports(matrices.hx, num_qubits)
    lz_columns = _column_supports(matrices.lz, num_qubits)
    lx_columns = _column_supports(matrices.lx, num_qubits)
    entries: list[Entry] = []
    for round_index in range(rounds):
        band = round_index * num_checks
        following = band + num_checks
        has_next = round_index + 1 < rounds
        for qubit in range(num_qubits):
            data_rate = data_flip_rates[qubit]
            phase_rate = phase_flip_rates[qubit]
            both_rate = both_flip_rates[qubit]
            if not (data_rate or phase_rate or both_rate):
                # A location that cannot fire is not a mechanism, so it is not
                # enumerated and does not become a zero-rate column.
                continue
            # An X fault reaches the Z-type checks, an X-type check being blind to
            # it, and the same for the Z fault and the X-type checks. A Y fault is
            # the two of them at once and therefore reaches both blocks.
            z_block = [band + row for row in hz_columns[qubit]]
            x_block = [band + num_z_checks + row for row in hx_columns[qubit]]
            if has_next:
                z_block.extend(following + row for row in hz_columns[qubit])
                x_block.extend(
                    following + num_z_checks + row for row in hx_columns[qubit]
                )
            z_logicals = lz_columns[qubit]
            x_logicals = tuple(num_z_logicals + row for row in lx_columns[qubit])
            if data_rate:
                entries.append((data_rate, tuple(z_block), tuple(z_logicals)))
            if phase_rate:
                entries.append((phase_rate, tuple(x_block), tuple(x_logicals)))
            if both_rate:
                entries.append(
                    (
                        both_rate,
                        tuple(z_block) + tuple(x_block),
                        tuple(z_logicals) + tuple(x_logicals),
                    )
                )
        for check in range(num_checks):
            measurement_rate = measurement_flip_rates[check]
            if not measurement_rate:
                continue
            detectors = [band + check]
            if has_next:
                detectors.append(following + check)
            entries.append((measurement_rate, tuple(detectors), ()))
    return rounds * num_checks, matrices.num_observables, tuple(entries)


@dataclass(frozen=True)
class CssCodeMatrices:
    """The four sparse binary matrices a Calderbank-Shor-Steane code is read by.

    Column ``q`` of every matrix is the ``q``-th data qubit, so the four are
    indexed alike and the record refuses a set that disagrees about how many data
    qubits the code has. ``hz`` holds the Z-type checks and ``lz`` the Z-type
    logical operators, which are the two matrices an X fault reaches -- an X fault
    on qubit ``q`` flips exactly the Z-type checks whose support holds ``q`` and
    exactly the Z-type logical operators whose support holds it. ``hx`` and ``lx``
    are the same pair for a Z fault, which the X-type checks are sensitive to and
    the Z-type checks are blind to. All four are carried because a code's two
    fault families are two different experiments over one set of data qubits, and
    a record that carried only one pair would be silently modelling half a CSS
    code.

    ``hx``, ``lz`` and ``lx`` may be omitted, and a block with no row is how a code
    states that it has no check or no logical operator of that type -- an X-type
    check measured on a code that has none is an empty block rather than a
    missing record. A block that has rows but no column is malformed rather than
    empty, because it states a check over no qubit.

    Raises:
        TypeError: If a block is not an integer tensor.
        ValueError: If a block is not two-dimensional, carries an element that is
            neither zero nor one, has rows but no column, or disagrees with
            another block about how many data qubits the code has.

    Examples:
        >>> import torch
        >>> from flagquantum.qec import CssCodeMatrices
        >>> hz = torch.tensor([[1, 1, 0], [0, 1, 1]])
        >>> matrices = CssCodeMatrices(hz=hz, lz=torch.tensor([[1, 1, 1]]))
        >>> matrices.num_checks, matrices.num_observables
        (2, 1)
    """

    hz: torch.Tensor
    hx: torch.Tensor | None = None
    lz: torch.Tensor | None = None
    lx: torch.Tensor | None = None

    def __post_init__(self) -> None:
        blocks: dict[str, torch.Tensor | None] = {}
        for name in ("hz", "hx", "lz", "lx"):
            block = getattr(self, name)
            blocks[name] = None if block is None else _binary_matrix(block, name=name)
        for name, block in blocks.items():
            if block is not None and int(block.shape[0]) and not int(block.shape[1]):
                raise ValueError(
                    f"{name} has {int(block.shape[0])} row(s) but no column, so it "
                    "states a check or a logical operator over no data qubit"
                )

        widths = {
            name: int(block.shape[1])
            for name, block in blocks.items()
            if block is not None and int(block.shape[1])
        }
        if len(set(widths.values())) > 1:
            stated = ", ".join(f"{name} has {width}" for name, width in widths.items())
            raise ValueError(
                f"the matrices must be indexed by the same data qubits but "
                f"{stated}; every non-empty block has one column per data qubit"
            )
        columns = next(iter(widths.values()), 0)

        for name, block in blocks.items():
            if block is None or not int(block.shape[0]):
                blocks[name] = torch.zeros((0, columns), dtype=torch.int64)
            else:
                blocks[name] = block
        for name, block in blocks.items():
            object.__setattr__(self, name, block)

    @property
    def num_qubits(self) -> int:
        """The number of data qubits every matrix is indexed by."""

        return int(self.hz.shape[1])

    @property
    def num_z_checks(self) -> int:
        """The number of Z-type checks, which are the Z-type detector rows."""

        return int(self.hz.shape[0])

    @property
    def num_x_checks(self) -> int:
        """The number of X-type checks, which are the X-type detector rows."""

        return int(self.hx.shape[0])  # type: ignore[union-attr]

    @property
    def num_checks(self) -> int:
        """The number of detectors in one round, every check counted once."""

        return self.num_z_checks + self.num_x_checks

    @property
    def num_z_logicals(self) -> int:
        """The number of Z-type logical operators, read by an X fault."""

        return int(self.lz.shape[0])  # type: ignore[union-attr]

    @property
    def num_x_logicals(self) -> int:
        """The number of X-type logical operators, read by a Z fault."""

        return int(self.lx.shape[0])  # type: ignore[union-attr]

    @property
    def num_observables(self) -> int:
        """The number of observable rows, the Z-type logicals first."""

        return self.num_z_logicals + self.num_x_logicals


def css_code_matrices(code: StabilizerCode) -> CssCodeMatrices:
    """Return a code record's CSS generator matrices.

    Column ``q`` of every matrix is the ``q``-th wire :attr:`data_wires` names, so
    a code is free to declare its data qubits on any wires, and the rows follow the
    code's own declaration order. Each of the code's checks contributes one row to
    the matrix of its own type and none to the other, and each logical observable
    does the same, so a code that declares an X-type logical operator is modelled
    with one and a code that declares none states an empty block.

    A logical observable that is neither pure X-type nor pure Z-type is refused
    rather than read as one of the two. It anticommutes with some check of each
    type, so it is not a CSS logical operator, and reading only its X support or
    only its Z support would report a different observable than the code declared.

    Raises:
        TypeError: If ``code`` is not a :class:`~flagquantum.qec.StabilizerCode`.
        ValueError: If a logical observable is of mixed type, or if a check or
            observable names a wire the code does not declare as a data wire.
    """

    if not isinstance(code, StabilizerCode):
        raise TypeError("code must be a StabilizerCode")
    data_wires = tuple(code.data_wires)
    columns = {wire: index for index, wire in enumerate(data_wires)}

    def _rows(entries: tuple[tuple[str, tuple[int, ...]], ...]) -> torch.Tensor:
        matrix = torch.zeros((len(entries), len(data_wires)), dtype=torch.int64)
        for row, (label, support) in enumerate(entries):
            for wire in support:
                if wire not in columns:
                    raise ValueError(
                        f"{label} acts on wire {wire}, which "
                        f"{type(code).__name__} does not declare as a data wire"
                    )
                matrix[row, columns[wire]] = 1
        return matrix

    checks = tuple(code.checks)
    hz = _rows(
        tuple(
            (f"check {check.index}", check.stabilizer.z_wires)
            for check in checks
            if check.stabilizer.z_wires
        )
    )
    hx = _rows(
        tuple(
            (f"check {check.index}", check.stabilizer.x_wires)
            for check in checks
            if check.stabilizer.x_wires
        )
    )
    observables = tuple(code.logical_observables)
    for index, observable in enumerate(observables):
        if observable.x_wires and observable.z_wires:
            raise ValueError(
                f"logical observable {index} is neither an X-type nor a Z-type "
                "operator, so it is not a CSS logical operator: reading it as one "
                "of the two would report an observable the code does not declare"
            )
    lz = _rows(
        tuple(
            (f"logical observable {index}", observable.z_wires)
            for index, observable in enumerate(observables)
            if observable.z_wires
        )
    )
    lx = _rows(
        tuple(
            (f"logical observable {index}", observable.x_wires)
            for index, observable in enumerate(observables)
            if observable.x_wires
        )
    )
    return CssCodeMatrices(hz=hz, hx=hx, lz=lz, lx=lx)


_ROUND_LOOP_ANCHOR = "    for round_index in range(rounds):\n"
_FLIP_INDENT = "        "
# The single-qubit gates whose product is each Pauli fault, in the hybrid
# language's own gate set. ``H`` is its own inverse and conjugating ``X`` by it is
# ``Z``, which is how a Z fault is written without a ``Z`` gate.
_FAULT_GATES: dict[str, tuple[str, ...]] = {
    "x": ("X",),
    "z": ("H", "X", "H"),
    "y": ("X", "H", "X", "H"),
}
_MEASUREMENT_FLIP_GATES = _FAULT_GATES["x"]


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

    ``fault`` names which of the three single-qubit Pauli faults a data
    mechanism is: ``"x"``, ``"z"`` or ``"y"``, in the letters the
    :class:`~flagquantum.qec.PhenomenologicalNoise` fields are documented with. It
    is ``"x"`` for a measurement mechanism, which is not a Pauli fault at all,
    because the field is part of every record and inventing a fourth value for
    the one kind that has no choice would make the two kinds harder to compare.
    """

    kind: Literal["data", "measurement"]
    round_index: int
    wire: int
    probability: float
    fault: Literal["x", "z", "y"] = "x"


def _check_rate_order(checks: tuple[CodeCheck, ...]) -> tuple[int, ...]:
    """Return each declared check's index in the matrix row order.

    ``measurement_flip_per_check`` is indexed the way upstream's ``pm_per_check``
    is and the way :func:`css_code_matrices` builds its rows: every Z-type check
    first, in declaration order, and then every X-type check. A code is free to
    declare its checks in any order -- the rotated surface code declares them in
    lattice order, which interleaves the two types -- so the position a check has
    in that declaration is not the index its rate is stated at, and this is the
    translation between the two. The two lists partition the checks rather than
    overlapping: :class:`~flagquantum.qec.CodeCheck` refuses a stabilizer that is
    neither pure X-type nor pure Z-type, so every check lands in exactly one of
    them and no check is left without a row.
    """

    z_positions = [
        position for position, check in enumerate(checks) if check.stabilizer.z_wires
    ]
    x_positions = [
        position for position, check in enumerate(checks) if check.stabilizer.x_wires
    ]
    order = [0] * len(checks)
    for row, position in enumerate((*z_positions, *x_positions)):
        order[position] = row
    return tuple(order)


def _mechanisms(
    circuit: MemoryCircuit, noise: PhenomenologicalNoise
) -> tuple[_Mechanism, ...]:
    """Return every noise location a probability makes possible, in order.

    Data faults come first — one per round per data wire per fault family — and
    measurement flips second, one per round per check. Every loop walks the code's
    own tuples in order, which is the order the emitted program measures in. A
    location whose **own** probability is zero cannot flip anything, so it is not
    enumerated at all: a caller that counts mechanisms then counts exactly what can
    happen, and a per-element vector is how a caller states that one location is
    quiet while its neighbours are not.

    The three fault families are enumerated separately rather than as one
    depolarizing location, because a code-capacity model detects them differently
    and a caller states them as three separate rates. This matches the matrix
    route, which has always enumerated all three; a data location is the same
    location whether the model is built from the program or from the matrices, and
    the two routes must not disagree about which faults exist.

    The probability a location is enumerated with is the element of the vector
    that names it, or the scalar when no vector is stated, and the element is
    resolved through the same accessors the matrix route uses, so the two routes
    cannot read one vector two ways. The per-check vector is indexed in the matrix
    row order rather than in the code's declaration order, which is what
    :func:`_check_rate_order` translates.

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

    Raises:
        ValueError: If a per-element rate vector the noise states does not name
            every data wire or every check the code declares.
    """

    data_wires = circuit.code.data_wires
    if len(set(data_wires)) != len(data_wires):
        raise ValueError(
            "the code declares a repeated data wire, so a mechanism at that "
            "location would be enumerated twice and merged with itself"
        )
    checks = tuple(circuit.code.checks)
    ancilla_wires = [check.ancilla_wire for check in checks]
    if len(set(ancilla_wires)) != len(ancilla_wires):
        repeated = sorted(
            wire for wire in set(ancilla_wires) if ancilla_wires.count(wire) > 1
        )
        raise ValueError(
            f"the code declares a repeated check ancilla wire ({repeated[0]}): "
            "two checks record one syndrome bit, so the model would read the "
            "same bit for both"
        )
    data_flip_rates = noise.data_flip_rates(num_qubits=len(data_wires))
    phase_flip_rates = noise.phase_flip_rates(num_qubits=len(data_wires))
    both_flip_rates = noise.both_flip_rates(num_qubits=len(data_wires))
    measurement_flip_rates = noise.measurement_flip_rates(num_checks=len(checks))
    rate_order = _check_rate_order(checks)
    mechanisms: list[_Mechanism] = []
    families: tuple[tuple[Literal["x", "z", "y"], tuple[float, ...]], ...] = (
        ("x", data_flip_rates),
        ("z", phase_flip_rates),
        ("y", both_flip_rates),
    )
    for round_index in range(circuit.rounds):
        for position, wire in enumerate(data_wires):
            for fault, rates in families:
                rate = rates[position]
                if rate:
                    mechanisms.append(
                        _Mechanism(
                            kind="data",
                            round_index=round_index,
                            wire=wire,
                            probability=rate,
                            fault=fault,
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
                        wire=check.ancilla_wire,
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


def _forced_flip(round_index: int, wire: int, *, gates: tuple[str, ...]) -> str:
    """Return the guarded single-qubit sequence that forces one error.

    ``gates`` is the sequence of the hybrid language's single-qubit gates whose
    product is the fault. The language carries ``H`` and ``X`` and no ``Z``, and
    conjugating ``X`` by ``H`` is exactly ``Z``, so a Z fault is written as those
    three gates rather than as a gate the capture layer would refuse. A Y fault is
    ``X`` followed by that same conjugation of ``X``, which is ``XZ``: the two
    differ by the global phase ``i``, and a Pauli frame does not see a global
    phase.
    """

    lines = [f"{_FLIP_INDENT}if round_index == {round_index}:"]
    lines.extend(f"{_FLIP_INDENT}    qp.{gate}(wires={wire})" for gate in gates)
    return "\n".join(lines) + "\n"


def _inject_data_flip(
    circuit: MemoryCircuit,
    *,
    round_index: int,
    wire: int,
    fault: Literal["x", "z", "y"] = "x",
) -> str:
    """Return ``circuit``'s source with one Pauli fault forced on a data wire.

    The fault is the one ``fault`` names and not the one that the experiment's
    readout basis makes visible. Those are the same question only in the Z basis:
    the model states an X fault, a Z fault and a Y fault as three independent
    locations, and each is injected as itself whichever basis the wire is read
    out in. Substituting one for another here would make ``data_flip`` open an
    X-basis experiment's observable, which is a claim the noise record does not
    make and the caller cannot see.

    The flip is guarded by ``if round_index == <round_index>:`` and inserted at
    the start of that round, before any of the round's CNOTs, so the error opens
    the frame the round's detectors compare against.
    """

    checked_round = _checked_round(circuit, round_index, label="data-flip round")
    checked_wire = _checked_wire(circuit.code.data_wires, wire, label="data-flip wire")
    if fault not in _FAULT_GATES:
        raise ValueError(
            f"unknown data fault {fault!r}: the model states an X, a Z and a Y "
            "fault, spelled 'x', 'z' and 'y'"
        )
    offset = _single_anchor(
        circuit.source, _ROUND_LOOP_ANCHOR, label="the round loop anchor"
    )
    end = offset + len(_ROUND_LOOP_ANCHOR)
    gates = _FAULT_GATES[fault]
    return (
        f"{circuit.source[:end]}"
        f"{_forced_flip(checked_round, checked_wire, gates=gates)}"
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
        f"{_forced_flip(checked_round, checked_wire, gates=_MEASUREMENT_FLIP_GATES)}"
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
