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

Both routes carry the same family of faults. On the matrix route the three data
faults are the three Pauli errors: an X fault reaches the Z-type detectors and
the Z-type logical operators, a Z fault reaches the X-type detectors and the
X-type logical operators, and a Y fault reaches both. Each data fault in round
``r`` shows up in the detector band of round ``r`` and in the band after it, and
a measurement fault shows up in its own check's detector and in the next round's,
which is the code-capacity experiment stated as detector differences.

The circuit route enumerates the same four families -- the X, Z and Y data faults
and the measurement fault -- and forces each one through the program instead of
reading it off a matrix. The injection writes its guard into the experiment's own
source, whose language carries ``h`` and ``x`` as its only single-qubit gates that
take no parameter, so a Z fault is forced as the X fault conjugated by ``h`` and a
Y fault as that Z fault followed by the X fault; both are exact gate identities
rather than approximations. What a fault then flips is the program's answer rather
than a plan's, and the Z-basis layout makes it a narrower answer than the matrix
route's: a Z fault never flips the Z-type logical observable, and it reaches only
the X-type checks, which have a detector for every round after the first. A Z
fault at the round-zero boundary is therefore no mechanism at all on a Z-memory
circuit -- the band it would flip has no reference to be compared against -- while
the same fault one round later is one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import Literal

import torch

from ..compiler._hybrid import INDEX, capture_source, lower_dynamic_program
from ..runtime.dynamic.hybrid_session import execute_hybrid_dynamic_session
from .circuit import MeasurementRef, MemoryCircuit
from .codes import CodeCheck, StabilizerCode
from .noise import PhenomenologicalNoise

__all__ = ("CssCodeMatrices", "css_code_matrices")

Entry = tuple[float, tuple[int, ...], tuple[int, ...]]

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
        if mechanism.kind in _FAULT_GATES:
            source = _inject_data_flip(
                circuit,
                round_index=mechanism.round_index,
                wire=mechanism.wire,
                kind=mechanism.kind,
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

# The gates a forced data fault is composed of, one entry per family the noise
# record states. The injection writes into a source whose language carries `h` and
# `x` as its only parameter-free single-qubit gates, so the two families that are
# not a bit flip are built out of those: `h` conjugates the `x` into a `z`, and a
# `y` is that `z` followed by the `x` again, since Z X = iY. Both are exact
# identities, so a forced fault is the Pauli it is named for and not an
# approximation of it; the global phase `i` cannot reach a flip set, which is what
# the signature is read from.
_FAULT_GATES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "data": ("X",),
        "phase": ("H", "X", "H"),
        "both": ("H", "X", "H", "X"),
    }
)


@dataclass(frozen=True)
class _Mechanism:
    """One physical noise location, by kind, round, and wire.

    The record carries the coordinates the matching injector needs rather than a
    rendered source. A caller that fires a set of mechanisms at once has to
    re-inject them into the one program it executes, and it cannot rebuild a
    source from a string alone; carrying coordinates also keeps injection in
    exactly one place. ``kind`` names the noise record's own field: ``"data"``
    for an X fault on a data wire, ``"phase"`` for a Z fault and ``"both"`` for a
    Y fault, where ``wire`` is that data wire, and ``"measurement"`` for a fault
    on a check's syndrome measurement, where ``wire`` is that check's ancilla. The
    kind is a literal rather than a free string, so a caller that builds a record
    by hand is told at the type checker which flips exist.
    """

    kind: _MechanismKind
    round_index: int
    wire: int
    probability: float


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

    The three data faults come first — one per round per data wire per family,
    the X family first and then the Z and Y families — and measurement faults
    second, one per round per check. Every loop walks the code's own tuples in
    order, which is the order the emitted program measures in. A location whose
    **own** probability is zero cannot flip anything, so it is not enumerated at
    all: a caller that counts mechanisms then counts exactly what can happen, and
    a per-element vector is how a caller states that one location is quiet while
    its neighbours are not.

    One wire can therefore carry up to three mechanisms in a round, one per
    family, and the record states each family's rate separately: a wire that is
    noisy in one Pauli and quiet in the others is enumerated once, which is what
    makes a bit-flip-only profile cost one mechanism rather than three.

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
    measurement_flip_rates = noise.measurement_flip_rates(num_checks=len(checks))
    rate_order = _check_rate_order(checks)
    mechanisms: list[_Mechanism] = []
    for kind, rates in _data_family_rates(noise, num_qubits=len(data_wires)):
        for round_index in range(circuit.rounds):
            for position, wire in enumerate(data_wires):
                rate = rates[position]
                if rate:
                    mechanisms.append(
                        _Mechanism(
                            kind=kind,
                            round_index=round_index,
                            wire=wire,
                            probability=rate,
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
    circuit: MemoryCircuit, *, round_index: int, wire: int, kind: str = "data"
) -> str:
    """Return ``circuit``'s source with one data fault forced on a data wire.

    The fault is guarded by ``if round_index == <round_index>:`` and inserted at
    the start of that round, before any of the round's CNOTs, so the error opens
    the frame the round's detectors compare against. ``kind`` selects the Pauli
    family, which is which gates :func:`_forced_flip` emits; the default is the
    bit flip, so a caller that names no family is injecting the fault this
    function was written for.
    """

    checked_round = _checked_round(circuit, round_index, label="data-flip round")
    checked_wire = _checked_wire(circuit.code.data_wires, wire, label="data-flip wire")
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
