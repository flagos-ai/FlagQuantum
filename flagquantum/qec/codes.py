"""Code-independent stabilizer-code descriptions.

A code declares its qubit layout, its checks, its per-basis ancilla bands, and
its logical observables. The frozen repetition profile keeps its own records;
this module describes a code as a value so that circuit generation, detector
layout, and decoding can be derived from it rather than pinned to one instance.
A record can also be built by name through the family's own registry, so a
caller that knows which code it wants does not have to know which class
declares it.

Two routes reach such a record. A family declares its own layout from a distance
or an index -- the repetition profile, the rotated surface patch, the ZXXZ
surface patch, the Steane code, the triangular colour patch and the
square-lattice torus all do -- and the matrix route instead takes the four
matrices a Calderbank-Shor-Steane code is already written down as, which is how
a code this package does not declare arrives. The two routes meet:
:func:`triangular_colour_code` and :func:`toric_code` derive a family's matrices
and hand back the same matrix record a caller would otherwise have written out
by hand. Every route satisfies the same protocol, so no consumer can tell which
one it was handed.

A check names one ancilla and the CNOT pairs that couple it to the stabilizer's
support, and the direction of each pair is fixed by the factor that qubit
carries. A pure Z-type check therefore uses one direction alone, a pure X-type
check the other, and a mixed X-and-Z check -- which is what every check of the
ZXXZ surface patch is -- uses both directions through the same ancilla.
"""

from __future__ import annotations

import inspect
import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    TypeVar,
    cast,
    runtime_checkable,
)

from .gf2 import in_span, reduce_rows, reduce_vector
from .pauli import Pauli

if TYPE_CHECKING:
    # The matrix route's record, which this module's derived families build and
    # which is not imported at run time: it rests on this module's protocol, so a
    # module-level import here would close a cycle between the two.
    from .css_code import CssCode


def ancilla_bands(
    checks: tuple[CodeCheck, ...] | list[CodeCheck],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split a code's check ancillas into the X band and the Z band.

    A check's ancilla measures one basis, which its stabilizer's type fixes, so
    the two bands are a function of the checks and are derived here rather than
    declared beside them. That is deliberate: a record that stated its bands as
    well as its checks could state two answers, and the drift would be silent
    because every consumer reads one of them. A record that derives them cannot.

    An ancilla that measures neither basis -- a flag or an idle ancilla -- is in
    neither band, so the two bands need not cover a record's declared ancillas.
    Both bands are returned in ascending qubit order, which is the order a
    detector band and a schedule are read in.

    Args:
        checks: The checks whose ancillas are being split.

    Returns:
        The ``(x_qubits, z_qubits)`` pair, each sorted and without duplicates.
    """

    x_qubits = tuple(
        sorted({check.ancilla_qubit for check in checks if check.stabilizer.x_qubits})
    )
    z_qubits = tuple(
        sorted({check.ancilla_qubit for check in checks if check.stabilizer.z_qubits})
    )
    overlap = set(x_qubits) & set(z_qubits)
    if overlap:
        raise ValueError(
            "ancillas "
            + ", ".join(str(qubit) for qubit in sorted(overlap))
            + " measure a check of each basis, so they belong to neither band: one "
            "ancilla measures one basis"
        )
    return x_qubits, z_qubits


def ancilla_band_counts(
    checks: tuple[CodeCheck, ...] | list[CodeCheck],
) -> tuple[int, int]:
    """Count the ancillas in each band, tolerating an ancilla that reads both.

    :func:`ancilla_bands` answers with the distinct ancilla each check measures,
    which is the set a detector band is laid out against, and it refuses an
    ancilla that reads both bases because one ancilla cannot be read in two
    bands. This accessor answers the same question -- how wide each band is --
    and differs only in that it does not refuse: an ancilla carrying a mixed
    check is counted in both bands instead of being rejected, so the count stays
    defined exactly where the band sets are not.

    That is what makes the mixed frame representable. Its checks read both bases
    through one ancilla, so the bands it defines overlap, and a record that
    reports its two widths is stating a number that exists. A record whose
    checks each measure one basis gets the same pair from either reading,
    because one ancilla then measures one check, and this is where a record's
    stated widths are compared against the widths its checks define.

    Args:
        checks: The checks whose per-band ancilla counts are being read.

    Returns:
        The ``(x_count, z_count)`` pair, each the number of distinct ancillas in
        that band.
    """

    return (
        len({check.ancilla_qubit for check in checks if check.stabilizer.x_qubits}),
        len({check.ancilla_qubit for check in checks if check.stabilizer.z_qubits}),
    )


@dataclass(frozen=True)
class CodeCheck:
    """One stabilizer check with its ancilla and its CNOT coupling.

    Each entry of ``cnot_qubits`` is a ``(control, target)`` pair, and every pair
    couples the check's single ancilla to one data qubit of the stabilizer's
    support. Which way a pair points is fixed by the stabilizer factor it
    carries rather than left to the caller:

    * A **Z factor** on a data qubit is read by coupling that data qubit into the
      ancilla with the data qubit controlling, so it comes first:
      ``(data_qubit, ancilla)``. That is the Z-type gadget: the ancilla is
      prepared in ``|0>`` and read in the Z basis.
    * An **X factor** is read by conjugating the ancilla with ``H`` and coupling
      it out into that data qubit with the ancilla controlling:
      ``(ancilla, data_qubit)``. That is the X-type gadget: the same ``H``
      prepares the ancilla in ``|+>`` and reads it back in the Z basis.

    Both gadgets leave the ancilla's Z-basis readout equal to the check's
    eigenvalue, which is what makes the two symmetric here. So ``cnot_qubits`` is
    the Z-factor pairs in the stabilizer's own qubit order, then the X-factor
    pairs in the stabilizer's own qubit order, and the boundary between the two
    runs is what says which pairs the ``H`` gates wrap. The emitted gadget is the
    Z pairs, then ``H``, then the X pairs, then ``H``, then a Z-basis readout of
    the ancilla -- which leaves that readout equal to the check's eigenvalue.

    **One ancilla measures a mixed stabilizer.** A pure Z-type check has an
    empty X run and no ``H`` gates at all; a pure X-type check has an empty Z
    run and is the ``H``-wrapped gadget alone. Those are the two shapes this
    record described before it described the mixed one, and they are not two
    rules beside a third: they are the two ends of the one rule above, and they
    are emitted by it unchanged.

    **Why the X run needs the H gates and the Z run does not.** Reading a Z
    factor through the ancilla is a CNOT in one direction, and reading an X
    factor is the same CNOT conjugated by ``H`` on the ancilla, because
    ``H X H = Z``. A mixed stabilizer needs both directions through the *same*
    ancilla, so it needs the ``H`` pair around the X half only -- never a second
    ancilla, and never an ``H`` that would also conjugate the Z half and turn
    the check into a different operator.
    """

    index: int
    stabilizer: Pauli
    ancilla_qubit: int
    cnot_qubits: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("check index must be non-negative")
        if not isinstance(self.stabilizer, Pauli):
            raise TypeError("check stabilizer must be a Pauli operator")
        if self.ancilla_qubit < 0:
            raise ValueError("check ancilla qubit must be non-negative")
        if not self.cnot_qubits:
            raise ValueError("check must declare at least one CNOT")
        for control, target in self.cnot_qubits:
            if control < 0 or target < 0:
                raise ValueError("check CNOT qubits must be non-negative")
        if not self.stabilizer.support:
            raise ValueError(
                "check stabilizer must have a non-empty support, because an "
                "identity check measures nothing and its ancilla has nothing to "
                "read"
            )
        self._validate_z_run()
        self._validate_x_run()

    def _validate_z_run(self) -> None:
        """Require the leading run to be the Z factors, each into the ancilla."""

        z_qubits = self.stabilizer.z_qubits
        run = self.cnot_qubits[: len(z_qubits)]
        for _, target in run:
            if target != self.ancilla_qubit:
                raise ValueError(
                    "check CNOT targets an X-factor pair, which must come after "
                    "the Z-factor pairs, or couples a qubit the stabilizer does "
                    "not carry a Z factor on"
                )
        controls = tuple(sorted(control for control, _ in run))
        if self.ancilla_qubit in controls:
            raise ValueError(
                "check CNOT controls must be data qubits, not the ancilla, "
                "because a Z-factor pair draws a data qubit into the ancilla"
            )
        if controls != z_qubits:
            raise ValueError(
                "check CNOT controls must match the stabilizer's Z-factor support"
            )

    def _validate_x_run(self) -> None:
        """Require the trailing run to be the X factors, each out of the ancilla."""

        x_qubits = self.stabilizer.x_qubits
        run = self.cnot_qubits[len(self.stabilizer.z_qubits) :]
        for control, _ in run:
            if control != self.ancilla_qubit:
                raise ValueError(
                    "check CNOT controls must be the declared ancilla, because an "
                    "X-factor pair is controlled by the declared ancilla and its "
                    "target data qubits are the stabilizer's X-factor support"
                )
        targets = tuple(sorted(target for _, target in run))
        if targets != x_qubits:
            raise ValueError(
                "check CNOT targets must match the stabilizer's X-factor support"
            )


@runtime_checkable
class StabilizerCode(Protocol):
    """A code that declares its qubit layout, checks, and logical observables.

    The ancilla count is stated twice on purpose: once as a total, and once per
    basis. Upstream's code record carries the same three accessors as pure
    virtuals -- ``get_num_ancilla_qubits``, ``get_num_ancilla_x_qubits`` and
    ``get_num_ancilla_z_qubits`` -- and the split is what a syndrome-extraction
    round and a detector band are laid out against, so a total alone does not say
    which ancillas a round has to visit. :func:`ancilla_bands` derives the two
    bands from ``checks``, and the records in this module read their counts from
    it, so a record cannot report a split its own checks contradict.

    ``num_x_stabilizers`` and ``num_z_stabilizers`` are the two further counts
    upstream's record declares beside those three, as the pure virtuals
    ``get_num_x_stabilizers`` and ``get_num_z_stabilizers``. Each of the three
    code classes upstream ships answers the same number to a stabilizer count as
    to the band count of the same basis, because one ancilla measures one
    stabilizer there exactly as it does here; the two are therefore one quantity
    in this repository rather than two, read once through
    :func:`ancilla_bands`, and these two accessors are aliases of the two band
    accessors rather than a second derivation. A caller reading upstream's
    accessor list finds the same five counts here, and a record that stated the
    same quantity a second time could state a different answer to it.
    """

    @property
    def distance(self) -> int: ...

    @property
    def num_data_qubits(self) -> int: ...

    @property
    def num_ancilla_qubits(self) -> int: ...

    @property
    def num_ancilla_x_qubits(self) -> int: ...

    @property
    def num_ancilla_z_qubits(self) -> int: ...

    @property
    def num_x_stabilizers(self) -> int: ...

    @property
    def num_z_stabilizers(self) -> int: ...

    @property
    def data_qubits(self) -> tuple[int, ...]: ...

    @property
    def ancilla_qubits(self) -> tuple[int, ...]: ...

    @property
    def checks(self) -> tuple[CodeCheck, ...]: ...

    @property
    def stabilizers(self) -> tuple[Pauli, ...]: ...

    @property
    def logical_observables(self) -> tuple[Pauli, ...]: ...


@dataclass(frozen=True)
class RepetitionCode:
    """The bit-flip repetition code with ``distance`` data qubits.

    Data qubits occupy qubits ``0..distance-1`` and check ancillas occupy qubits
    ``distance..2*distance-2``. Check ``c`` measures ``Z_c Z_{c+1}``, so the code
    detects bit flips. The single declared logical observable is ``Z`` on every
    data qubit, which makes its readout representative the parity of the terminal
    data measurements.
    """

    distance: int = 3

    def __post_init__(self) -> None:
        if isinstance(self.distance, bool) or not isinstance(self.distance, Integral):
            raise TypeError("repetition-code distance must be an integer")
        if self.distance < 2:
            raise ValueError("repetition-code distance must be at least two")

    @property
    def num_data_qubits(self) -> int:
        return self.distance

    @property
    def num_ancilla_qubits(self) -> int:
        return self.distance - 1

    @property
    def num_ancilla_x_qubits(self) -> int:
        return ancilla_band_counts(self.checks)[0]

    @property
    def num_ancilla_z_qubits(self) -> int:
        return ancilla_band_counts(self.checks)[1]

    @property
    def num_x_stabilizers(self) -> int:
        return self.num_ancilla_x_qubits

    @property
    def num_z_stabilizers(self) -> int:
        return self.num_ancilla_z_qubits

    @property
    def data_qubits(self) -> tuple[int, ...]:
        return tuple(range(self.distance))

    @property
    def ancilla_qubits(self) -> tuple[int, ...]:
        return tuple(range(self.distance, 2 * self.distance - 1))

    @property
    def checks(self) -> tuple[CodeCheck, ...]:
        return tuple(
            CodeCheck(
                index=index,
                stabilizer=Pauli(z_qubits=(index, index + 1)),
                ancilla_qubit=self.distance + index,
                cnot_qubits=(
                    (index, self.distance + index),
                    (index + 1, self.distance + index),
                ),
            )
            for index in range(self.distance - 1)
        )

    @property
    def stabilizers(self) -> tuple[Pauli, ...]:
        return tuple(check.stabilizer for check in self.checks)

    @property
    def logical_observables(self) -> tuple[Pauli, ...]:
        return (Pauli(z_qubits=self.data_qubits),)


def _surface_ancilla_sites(distance: int) -> tuple[tuple[int, int, bool], ...]:
    """Return the rotated-surface-code ancilla sites as ``(a, b, is_x)``.

    Data qubits sit on the lattice ``(i, j)`` with ``0 <= i, j <= distance - 1``
    and an ancilla sits on ``(a, b)`` with ``0 <= a, b <= distance``. An X-type
    ancilla needs an interior column and an odd lattice parity, a Z-type ancilla
    needs an interior row and an even lattice parity, which is the stabilizer
    assignment the rotated layout requires: the parity that would place an
    ancilla outside the patch is what truncates the boundary checks.

    Sites are reported in lattice order so the check and ancilla indices are
    reproducible.
    """

    sites: list[tuple[int, int, bool]] = []
    for a in range(distance + 1):
        for b in range(distance + 1):
            if 1 <= a <= distance - 1 and (a + b) % 2 == 1:
                sites.append((a, b, True))
            elif 1 <= b <= distance - 1 and (a + b) % 2 == 0:
                sites.append((a, b, False))
    return tuple(sites)


@dataclass(frozen=True)
class RotatedSurfaceCode:
    """The rotated surface code of odd ``distance`` read out as Z memory.

    Data qubits occupy qubits ``0..distance**2 - 1``, indexed so that lattice
    site ``(i, j)`` is qubit ``j * distance + i``. Ancillas follow the data qubits
    in lattice order. ``(i, j)`` is a data qubit for ``0 <= i, j <= distance - 1``
    and ``(a, b)`` is an ancilla for ``0 <= a, b <= distance``; an X-type ancilla
    sits on an interior column with odd lattice parity, a Z-type ancilla on an
    interior row with even lattice parity, and its support is the up to four data
    qubits diagonally adjacent to it. That assignment is the rotated layout: a
    check that would fall outside the patch is truncated, which is what leaves
    the boundary checks at weight two.

    The declared logical observable is ``Z`` on the data row ``j == 0``, which is
    one complete row of the patch, and the two logical operators of the patch
    therefore read out as Z memory. The X-type checks are still measured every
    round, because they detect the Z errors this readout is vulnerable to; they
    are simply not deterministic in the initial all-zero state.

    This record describes the checks and the observable. It does not choose a
    decoding strategy, a noise model, or an error-correction threshold.
    """

    distance: int = 3

    def __post_init__(self) -> None:
        if isinstance(self.distance, bool) or not isinstance(self.distance, Integral):
            raise TypeError("rotated-surface-code distance must be an integer")
        if self.distance < 2:
            raise ValueError("rotated-surface-code distance must be at least two")

    @property
    def num_data_qubits(self) -> int:
        return self.distance * self.distance

    @property
    def num_ancilla_qubits(self) -> int:
        return len(_surface_ancilla_sites(self.distance))

    @property
    def num_ancilla_x_qubits(self) -> int:
        return ancilla_band_counts(self.checks)[0]

    @property
    def num_ancilla_z_qubits(self) -> int:
        return ancilla_band_counts(self.checks)[1]

    @property
    def num_x_stabilizers(self) -> int:
        return self.num_ancilla_x_qubits

    @property
    def num_z_stabilizers(self) -> int:
        return self.num_ancilla_z_qubits

    @property
    def data_qubits(self) -> tuple[int, ...]:
        return tuple(range(self.num_data_qubits))

    @property
    def ancilla_qubits(self) -> tuple[int, ...]:
        first = self.num_data_qubits
        return tuple(first + index for index in range(self.num_ancilla_qubits))

    @property
    def checks(self) -> tuple[CodeCheck, ...]:
        distance = self.distance
        checks: list[CodeCheck] = []
        for index, (a, b, is_x) in enumerate(_surface_ancilla_sites(distance)):
            ancilla = self.num_data_qubits + index
            support = tuple(
                sorted(
                    j * distance + i
                    for i in range(distance)
                    for j in range(distance)
                    if i in (a - 1, a) and j in (b - 1, b)
                )
            )
            if is_x:
                stabilizer = Pauli(x_qubits=support)
                cnot_qubits = tuple((ancilla, wire) for wire in support)
            else:
                stabilizer = Pauli(z_qubits=support)
                cnot_qubits = tuple((wire, ancilla) for wire in support)
            checks.append(
                CodeCheck(
                    index=index,
                    stabilizer=stabilizer,
                    ancilla_qubit=ancilla,
                    cnot_qubits=cnot_qubits,
                )
            )
        return tuple(checks)

    @property
    def stabilizers(self) -> tuple[Pauli, ...]:
        return tuple(check.stabilizer for check in self.checks)

    @property
    def logical_observables(self) -> tuple[Pauli, ...]:
        return (Pauli(z_qubits=tuple(range(self.distance))),)


_STEANE_CHECK_SUPPORTS: tuple[tuple[int, ...], ...] = (
    (3, 4, 5, 6),
    (1, 2, 5, 6),
    (0, 2, 4, 6),
)
_STEANE_LOGICAL_SUPPORT: tuple[int, ...] = (0, 1, 2)


@dataclass(frozen=True)
class SteaneCode:
    """The seven-qubit Steane code, declared by its checks and its logicals.

    Data qubits occupy qubits ``0..6`` and the six check ancillas follow them on
    qubits ``7..12``: checks ``0..2`` are Z-type and checks ``3..5`` are X-type,
    each with the same support, which is what makes the code Calderbank-Shor-
    Steane. The supports are the three non-zero parity constraints of the
    ``[7, 4, 3]`` Hamming code, so each check has weight four and the code has
    distance three.

    Both logical operators are declared, on the same three data qubits: ``Z`` on
    ``(0, 1, 2)`` and ``X`` on ``(0, 1, 2)``. They anticommute, since they overlap
    so an odd number of qubits flips the is a product of checks. This is the
    smallest code this package declares whose two fault families are both
    non-trivial, which is why it is the record the matrix route is exercised on.

    The record states the checks and the observables. It does not choose a
    decoding strategy or a noise model, and it declines to describe a
    preparation, so a memory circuit built from it is Z memory like every other
    record here.
    """

    @property
    def distance(self) -> int:
        return 3

    @property
    def num_data_qubits(self) -> int:
        return 7

    @property
    def num_ancilla_qubits(self) -> int:
        return 6

    @property
    def num_ancilla_x_qubits(self) -> int:
        return ancilla_band_counts(self.checks)[0]

    @property
    def num_ancilla_z_qubits(self) -> int:
        return ancilla_band_counts(self.checks)[1]

    @property
    def num_x_stabilizers(self) -> int:
        return self.num_ancilla_x_qubits

    @property
    def num_z_stabilizers(self) -> int:
        return self.num_ancilla_z_qubits

    @property
    def data_qubits(self) -> tuple[int, ...]:
        return tuple(range(7))

    @property
    def ancilla_qubits(self) -> tuple[int, ...]:
        return tuple(range(7, 13))

    @property
    def checks(self) -> tuple[CodeCheck, ...]:
        checks: list[CodeCheck] = []
        for index, support in enumerate(_STEANE_CHECK_SUPPORTS):
            ancilla = 7 + index
            checks.append(
                CodeCheck(
                    index=index,
                    stabilizer=Pauli(z_qubits=support),
                    ancilla_qubit=ancilla,
                    cnot_qubits=tuple((wire, ancilla) for wire in support),
                )
            )
        for offset, support in enumerate(_STEANE_CHECK_SUPPORTS):
            index = len(_STEANE_CHECK_SUPPORTS) + offset
            ancilla = 7 + index
            checks.append(
                CodeCheck(
                    index=index,
                    stabilizer=Pauli(x_qubits=support),
                    ancilla_qubit=ancilla,
                    cnot_qubits=tuple((ancilla, wire) for wire in support),
                )
            )
        return tuple(checks)

    @property
    def stabilizers(self) -> tuple[Pauli, ...]:
        return tuple(check.stabilizer for check in self.checks)

    @property
    def logical_observables(self) -> tuple[Pauli, ...]:
        return (
            Pauli(z_qubits=_STEANE_LOGICAL_SUPPORT),
            Pauli(x_qubits=_STEANE_LOGICAL_SUPPORT),
        )


_DISTANCE_SEARCH_WEIGHT = 3
# The weight up to which this module proves the distance it reports for a family it
# derives from a lattice, and the default those builders search to.
#
# A Calderbank-Shor-Steane code has one distance per Pauli family and neither has a
# closed form, so a derived family looks for a logical operator of the least weight
# there is and refuses to report a distance when the search reaches this weight
# without finding one. Three admits every distance-three code, which is what the
# families derived here are, and it is what makes the default affordable: the search
# tries every qubit subset up to the bound, so raising the bound to four costs an
# order of magnitude. Measured on the repetition profile read as a code with no
# low-weight logical operator, a bound of three takes 0.36 s at 100 data qubits and
# 1.3 s at 140, while a bound of four takes 7.2 s and 37.5 s. A code of higher
# distance is derived by asking for a bound the search reaches, which is where that
# cost becomes the caller's decision rather than the default's.

BinaryMatrix = Sequence[Sequence[int]]
# The matrices a Calderbank-Shor-Steane code is written down as: one inner sequence
# per check or logical operator, holding one ``0`` or ``1`` per data qubit.


def _block_width(rows: tuple[tuple[int, ...], ...], *, name: str) -> int | None:
    """Return the number of columns ``rows`` states, or ``None`` when it has no row."""

    if not rows:
        return None
    width = len(rows[0])
    if width == 0:
        raise ValueError(
            f"{name} has {len(rows)} row(s) but no column, so it states an operator "
            "over no data qubit"
        )
    if any(len(row) != width for row in rows):
        raise ValueError(
            f"{name} rows disagree about how many data qubits the code has, and every "
            "matrix of a code is read over the same data qubits"
        )
    return width


def _matrix_rows(block: object, *, name: str) -> tuple[tuple[int, ...], ...]:
    """Read ``block`` as a matrix of ``0``s and ``1``s, one row per operator."""

    if isinstance(block, (str, bytes)) or not isinstance(block, Sequence):
        raise TypeError(
            f"{name} must be a matrix of 0s and 1s with one row per check or "
            "logical operator, which is a sequence of sequences: read a tensor "
            "block with .tolist() first"
        )
    rows: list[tuple[int, ...]] = []
    for index, row in enumerate(block):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise TypeError(f"{name} row {index} must be a sequence of 0s and 1s")
        values: list[int] = []
        for entry in row:
            if isinstance(entry, bool) or not isinstance(entry, Integral):
                raise TypeError(
                    f"{name} row {index} carries {entry!r}, and a parity-check or "
                    "logical-operator matrix has one 0-or-1 entry per data qubit"
                )
            value = int(entry)
            if value not in (0, 1):
                raise ValueError(
                    f"{name} row {index} carries {value}, and a parity-check or "
                    "logical-operator matrix has one 0-or-1 entry per data qubit"
                )
            values.append(value)
        rows.append(tuple(values))
    return tuple(rows)


def _mask(row: tuple[int, ...]) -> int:
    """Return one row as a bit vector over its columns."""

    bits = 0
    for position, value in enumerate(row):
        if value:
            bits |= 1 << position
    return bits


def _supports(rows: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    """Return the qubit support of every row, in ascending qubit order."""

    return tuple(
        tuple(position for position, value in enumerate(row) if value) for row in rows
    )


def _odd_overlap(vector: int, row: int) -> bool:
    """Whether the two bit vectors agree on an odd number of columns."""

    return bin(vector & row).count("1") % 2 == 1


def _minimum_logical_weight(
    *,
    width: int,
    commuting_rows: tuple[int, ...],
    stabilizer_rows: tuple[int, ...],
    bound: int,
) -> int | None:
    """Return the least weight of a logical operator of one Pauli family.

    An operator of the family commutes with every check of the other family, which
    is ``commuting_rows``, and is not a combination of the checks of its own
    family, which is ``stabilizer_rows``. Qubit subsets are tried in increasing
    weight, so the first operator found has the least weight there is, and the
    search stops at ``bound``.

    The stabilizer span is reduced to its pivots once, because every candidate is
    tested against the same span.
    """

    pivots = reduce_rows(list(stabilizer_rows))
    for weight in range(1, bound + 1):
        for positions in itertools.combinations(range(width), weight):
            vector = 0
            for position in positions:
                vector |= 1 << position
            if any(_odd_overlap(vector, row) for row in commuting_rows):
                continue
            if reduce_vector(vector, pivots):
                return weight
    return None


def _refuse_non_logicals(
    *,
    rows: tuple[tuple[int, ...], ...],
    family: str,
    as_pauli: tuple[Pauli, ...],
    opposite_checks: tuple[Pauli, ...],
    opposite_family: str,
    own_masks: tuple[int, ...],
    own_family: str,
) -> None:
    """Refuse every row of one logical family that is not a logical operator."""

    for index, logical in enumerate(as_pauli):
        for check_index, check in enumerate(opposite_checks):
            if not logical.commutes_with(check):
                raise ValueError(
                    f"{family} row {index} anticommutes with {opposite_family} row "
                    f"{check_index}, so it does not preserve the code space and its "
                    "outcome cannot be read out"
                )
        if in_span(_mask(rows[index]), list(own_masks)):
            raise ValueError(
                f"{family} row {index} is a combination of the {own_family} rows, so "
                "it is a stabilizer whose outcome is fixed and the terminal readout "
                "would report a constant rather than the operator's eigenvalue"
            )


def triangular_colour_code(distance: int) -> CssCode:
    """Return the triangular 6.6.6 colour code of the requested distance.

    A colour code is written down on a trivalent lattice whose faces are
    three-colourable, and every face carries one X-type and one Z-type stabilizer
    over the data qubits at its vertices. That is what makes the family self-dual,
    which is why the two check blocks below are one matrix, and it is also why the
    family is not a matching code: a face in the bulk has six vertices, so one data
    fault inside it can flip more detectors than an edge can join.

    **The patch is a triangle and its side is the distance.** A point ``(row,
    column)`` with ``0 <= column <= row <= bound`` is either a data vertex or a face
    centre, and the face centres are the points where ``column % 3 == 2 - row % 3``.
    Every other point is a data vertex. A face is the six in-patch points one step
    diagonally or orthogonally away from it, so a face in the bulk has weight six
    and a face on the triangle's edge has weight four, where the boundary cuts the
    hexagon away. The declared logical operator is the data vertices on the
    ``column == 0`` side, which is the patch's shortest logical string; it commutes
    with every face and overlaps itself on an odd number of vertices, so it is a
    logical operator of both families rather than of one.

    **The requested distance is proved rather than stated.** The record handed back
    is the matrix route's :class:`~flagquantum.qec.CssCode`, so the distance it
    states is searched for over the matrices derived here: the least weight of a
    logical operator of each family is found and the smaller of the two becomes the
    record's ``distance``. The bound that search is given is the requested distance
    itself, since a smaller one would refuse the code whose distance is the number
    asked for, and that bound is the whole cost of this function: the search tries
    every qubit subset up to the distance, so the family is affordable at the small
    distances an experiment is built from and is not a way to write down a large
    patch.

    **At distance three this is the Steane code, relabelled.** The smallest patch
    has seven data vertices over three faces, and its three weight-four checks are
    the Steane code's three weight-four checks under a permutation of the data
    qubits -- the identity is not that permutation, which
    ``tests/qec/test_colour_code.py`` exhibits rather than asserts. The two records
    are therefore one code in two descriptions at distance three rather than two
    seven-qubit codes, and their qubit numbers are not comparable position by
    position. Nothing here reads that equivalence, because the distance is proved
    from this patch's own matrices, and the family grows away from it: at distance
    five the patch has nineteen data vertices and checks of weight four and six,
    which no Steane-style description reaches.

    Args:
        distance: The code distance, an odd integer of at least three.

    Returns:
        The code record, with one logical qubit and both family distances equal to
        ``distance``.

    Raises:
        TypeError: If ``distance`` is not an integer.
        ValueError: If ``distance`` is even or below three, or if no logical
            operator of a family reaches the requested weight.

    Examples:
        >>> colour = triangular_colour_code(3)
        >>> colour.num_data_qubits, colour.num_ancilla_qubits, colour.distance
        (7, 6, 3)
        >>> colour.stabilizers[0] == Pauli(z_qubits=(0, 1, 2, 3))
        True
    """

    if isinstance(distance, bool) or not isinstance(distance, Integral):
        raise TypeError("colour-code distance must be an integer")
    if distance % 2 == 0:
        raise ValueError(
            f"the triangular colour patch of distance {distance} has no whole side: "
            "the side spans (distance - 1) lattice periods of three rows, so the "
            "distance must be odd"
        )
    if distance < 3:
        raise ValueError(
            f"the triangular colour patch of distance {distance} has no face and "
            "therefore no check, so it is not a code"
        )

    bound = 3 * (distance - 1) // 2
    points = tuple(
        (row, column) for row in range(bound + 1) for column in range(row + 1)
    )
    # A face centre is the point whose column sits two places back from its row in
    # the three-colouring, in the row coordinates the patch is written in.
    faces = tuple(point for point in points if point[1] % 3 == 2 - point[0] % 3)
    vertices = tuple(point for point in points if point[1] % 3 != 2 - point[0] % 3)
    wire_of = {vertex: index for index, vertex in enumerate(vertices)}

    offsets = ((-1, -1), (-1, 0), (0, -1), (0, 1), (1, 0), (1, 1))
    supports = tuple(
        tuple(
            sorted(
                wire_of[(row + down, column + across)]
                for down, across in offsets
                if (row + down, column + across) in wire_of
            )
        )
        for row, column in faces
    )
    side = tuple(sorted(wire_of[vertex] for vertex in vertices if vertex[1] == 0))

    width = len(vertices)
    checks = tuple(
        tuple(1 if wire in support else 0 for wire in range(width))
        for support in supports
    )
    logical = (tuple(1 if wire in side else 0 for wire in range(width)),)
    # The record is built on this module's protocol, so it is imported here rather
    # than at module level, where it would close a cycle between the two modules.
    from .css_code import CssCode

    return CssCode(
        hz=checks,
        hx=checks,
        lz=logical,
        lx=logical,
        distance_search_weight=int(distance),
    )


def toric_code(linear_size: int) -> CssCode:
    """Return the square-lattice toric code of the requested linear size.

    A toric code is written down on a torus rather than on a patch. The lattice is
    a ``linear_size``-by-``linear_size`` grid whose opposite sides are identified,
    every vertex carries one X-type stabilizer over the four edges meeting it, and
    every face carries one Z-type stabilizer over the four edges around it. The
    identification is what the family is: a periodic lattice has no boundary, so
    no face is cut down by one and every check has weight four at every size.

    **The patch is a torus, so the code carries two logical qubits rather than
    one.** A logical operator is a cycle of edges that wraps one of the two
    directions without bounding a face, and each direction carries one of each
    family. The Z-type family below is therefore two operators -- the ring of
    horizontal edges in row zero and the ring of vertical edges in column zero --
    and the X-type family is the matching pair of cuts, which cross the first ring
    and the second ring respectively. This is the first record in this package
    whose matrices leave ``k = 2``: ``css_code_matrices`` reports four logical
    operators for it, and a memory circuit built from it declares two observables.
    ``hz`` and ``hx`` are not one matrix here, because a star spans the edges at a
    vertex and a face spans the edges around a face; the two families are
    transposes of each other in the lattice sense and are computed from the same
    edge numbering rather than kept as two tables in step.

    **The linear size is the distance, and the distance is proved rather than
    stated.** The record handed back is the matrix route's
    :class:`~flagquantum.qec.CssCode`, so the distance it states is searched for
    over the matrices derived here: the least weight of a logical operator of each
    family is found -- the two rings and the two cuts, each of weight
    ``linear_size`` -- and the smaller of the two becomes the record's ``distance``.
    The search is given the linear size as its bound. That bound is the whole cost
    of this function, because the search tries every qubit subset up to the bound:
    measured, the record builds in 0.00 s at size two, three and four,
    0.34 s at five and 23.1 s at six, on 2 * size ** 2 data qubits. It is
    therefore a way to write down the small tori an experiment is built from and
    not a way to write down a large one, and a caller who wants a larger lattice
    wants a decoder that reads its distance from the lattice instead -- which is
    the work of its own row rather than this route's.

    Args:
        linear_size: The number of vertices along one side of the torus, an
            integer of at least two. The patch has ``2 * linear_size ** 2`` data
            qubits, ``2 * linear_size ** 2`` checks and two logical qubits.

    Returns:
        The code record, with two logical qubits and both family distances equal
        to ``linear_size``.

    Raises:
        TypeError: If ``linear_size`` is not an integer.
        ValueError: If ``linear_size`` is below two, or if no logical operator of
            a family reaches the requested weight.

    Examples:
        >>> torus = toric_code(2)
        >>> torus.num_data_qubits, torus.num_ancilla_qubits, torus.distance
        (8, 8, 2)
        >>> torus.stabilizers[0] == Pauli(z_qubits=(0, 1, 4, 6))
        True
    """

    if isinstance(linear_size, bool) or not isinstance(linear_size, Integral):
        raise TypeError("toric linear size must be an integer")
    if linear_size < 2:
        raise ValueError(
            f"the torus of linear size {linear_size} has no face and therefore no "
            "check, so it is not a code"
        )
    size = int(linear_size)
    width = 2 * size * size

    def edge(side: int, row: int, column: int) -> int:
        """The qubit of one edge, with both torus coordinates wrapped."""

        return side * size * size + (row % size) + size * (column % size)

    def matrix_row(support: frozenset[int]) -> tuple[int, ...]:
        return tuple(1 if wire in support else 0 for wire in range(width))

    # A star is the four edges at a vertex: the two horizontal edges to its right
    # and the two vertical edges below it. A face is the four edges around a face:
    # the two horizontal edges of its top row and the two vertical edges of its
    # left column. Both are read off the one edge numbering above, so wrapping is
    # the modulus in that function and not a special case here.
    stars = tuple(
        matrix_row(
            frozenset(
                (
                    edge(0, row, column),
                    edge(0, row, column + 1),
                    edge(1, row, column),
                    edge(1, row + 1, column),
                )
            )
        )
        for row in range(size)
        for column in range(size)
    )
    faces = tuple(
        matrix_row(
            frozenset(
                (
                    edge(0, row - 1, column),
                    edge(0, row, column),
                    edge(1, row, column - 1),
                    edge(1, row, column),
                )
            )
        )
        for row in range(size)
        for column in range(size)
    )
    row_ring = frozenset(edge(0, 0, column) for column in range(size))
    column_ring = frozenset(edge(1, row, 0) for row in range(size))
    column_cut = frozenset(edge(0, row, 0) for row in range(size))
    row_cut = frozenset(edge(1, 0, column) for column in range(size))
    # The record is built on this module's protocol, so it is imported here rather
    # than at module level, where it would close a cycle between the two modules.
    from .css_code import CssCode

    return CssCode(
        hz=faces,
        hx=stars,
        lz=(matrix_row(row_ring), matrix_row(column_ring)),
        lx=(matrix_row(column_cut), matrix_row(row_cut)),
        distance_search_weight=size,
    )


# ---------------------------------------------------------------------------
# The code registry
# ---------------------------------------------------------------------------

#: The members a record has to carry to be registered, read off the protocol
#: rather than restated: the protocol is what a consumer holds a record to, and a
#: second hand-written list here would be a second statement of the same
#: contract, free to drift from the one callers are type-checked against.
_STABILIZER_CODE_MEMBERS: tuple[str, ...] = tuple(
    sorted(name for name in vars(StabilizerCode) if not name.startswith("_"))
)

#: One name per registered record. The value is the class and not an instance,
#: because the options are the caller's: ``get_code`` builds the record, so a
#: registered instance would have frozen a distance nobody chose.
_CODES: dict[str, type[StabilizerCode]] = {}

_CodeT = TypeVar("_CodeT")


def register_code(name: str, *, replace: bool = False) -> Callable[[_CodeT], _CodeT]:
    """Name a stabilizer-code record in the registry.

    Registration is what lets a caller reach a code by name instead of by class,
    which is the shape upstream's own factory has: ``get_code(name, options)``
    builds a code and ``get_available_codes()`` lists the names. The class is
    registered and returned unchanged, so a registered record keeps its name and
    its module and stays directly constructible.

    Args:
        name: The name :func:`get_code` will build this record from. It must be a
            non-empty string with no whitespace in it, so that a name is one
            token and can be printed in a refusal.
        replace: Whether to overwrite an existing registration. Replacing is
            refused by default, for the same reason it is refused for a decoder:
            two records answering to one name is a choice the registry cannot
            make on the caller's behalf, and the second registration is usually a
            typo rather than an intent.

    Returns:
        The decorator that registers a class and returns it unchanged.

    Raises:
        TypeError: The name is not a string, or the registered object does not
            carry every member of the :class:`StabilizerCode` protocol.
        ValueError: The name is empty or carries whitespace, or it is already
            registered and ``replace`` is not set.
    """

    if not isinstance(name, str):
        raise TypeError("a code name must be a string")
    if not name or name != name.strip() or any(char.isspace() for char in name):
        raise ValueError(
            f"a code name must be non-empty and carry no whitespace, not {name!r}"
        )
    if name in _CODES and not replace:
        raise ValueError(
            f"the name {name!r} is already registered against "
            f"{getattr(_CODES[name], '__name__', _CODES[name])!r}; pass "
            "replace=True to overwrite it"
        )

    def decorate(candidate: _CodeT) -> _CodeT:
        missing = [
            member
            for member in _STABILIZER_CODE_MEMBERS
            if not hasattr(candidate, member)
        ]
        if missing:
            raise TypeError(
                f"{getattr(candidate, '__name__', candidate)!r} cannot be registered "
                f"as {name!r}: a stabilizer-code record declares "
                + ", ".join(_STABILIZER_CODE_MEMBERS)
                + f", and {', '.join(missing)} "
                + ("is" if len(missing) == 1 else "are")
                + " missing"
            )
        # The members were just checked, which is the only form of the protocol
        # check a class object admits: isinstance reads an instance, and no
        # instance exists until a caller asks for one.
        _CODES[name] = cast("type[StabilizerCode]", candidate)
        return candidate

    return decorate


def code_names() -> tuple[str, ...]:
    """Return the registered code names, sorted, so a refusal can list them."""

    return tuple(sorted(_CODES))


def get_code(name: str, **options: Any) -> StabilizerCode:
    """Build the stabilizer-code record registered under ``name``.

    The options are the fields the record itself declares, forwarded to its
    constructor, so a name reaches the same record a direct construction would
    and there is no second place a distance or a patch size can be set. A name
    whose record takes no fields is built with no options, which is what makes
    ``get_code("steane")`` and ``get_code("rotated_surface", distance=5)`` the
    same call shape.

    Args:
        name: A name from :func:`code_names`.
        **options: Field values for the record, checked against the record's own
            fields before the constructor runs so an unknown field is a refusal
            that lists the ones it takes rather than a traceback from inside it.
            A value the record refuses is refused by the record, with the
            message the record states.

    Returns:
        The record, ready for :func:`flagquantum.qec.build_memory_circuit`.

    Raises:
        ValueError: The name is not registered.
        TypeError: The name is not a string, or an option is not a field of the
            record the name selects.

    Note:
        Upstream's factory also has an overload that builds a code from a list of
        stabilizers the caller supplies. That overload is not here, because a
        record built from arbitrary stabilizers has to decide each check's
        ancilla and coupling direction on the caller's behalf, and a code whose
        generator matrix is known is already served by
        :meth:`flagquantum.qec.DetectorErrorModel.from_code_matrices`, which
        reads the matrix rather than inventing a gadget for it.
    """

    try:
        record = _CODES[name]
    except KeyError:
        known = ", ".join(code_names())
        raise ValueError(
            f"no code is registered as {name!r}; the registry holds {known}"
        ) from None
    fields = [
        parameter
        for parameter in inspect.signature(record).parameters.values()
        if parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
        and parameter.name != "self"
    ]
    accepted = {parameter.name for parameter in fields}
    unknown = sorted(set(options) - accepted)
    if unknown:
        raise TypeError(
            f"{getattr(record, '__name__', record)!r} takes no option "
            + ", ".join(repr(field) for field in unknown)
            + "; it takes "
            + (", ".join(sorted(accepted)) if accepted else "no options")
        )
    return record(**options)


register_code("repetition")(RepetitionCode)
register_code("rotated_surface")(RotatedSurfaceCode)
register_code("steane")(SteaneCode)


__all__ = (
    "CssCode",
    "CodeCheck",
    "RepetitionCode",
    "RotatedSurfaceCode",
    "StabilizerCode",
    "SteaneCode",
    "ancilla_band_counts",
    "ancilla_bands",
    "code_names",
    "get_code",
    "register_code",
    "toric_code",
    "triangular_colour_code",
)


def __getattr__(name: str) -> Any:
    """Publish the matrix record from its own module, on first use.

    :class:`~flagquantum.qec.CssCode` is built on this module's protocol, so a
    module-level import of it here would close a cycle between the two modules.
    The name is resolved on first use instead, which is what lets a caller import
    it from either home.
    """

    if name == "CssCode":
        from .css_code import CssCode

        return CssCode
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
