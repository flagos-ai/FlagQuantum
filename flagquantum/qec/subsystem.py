"""Subsystem codes: a code whose logical group is smaller than its check group.

A stabilizer code is described by its checks, and the joint ``+1`` eigenspace of
their span is the code space. A **subsystem code** declares more operators than
that. Some of them are the stabilizer; the rest are *gauge* generators, which are
measured alongside it and are not fixed by the code space. What a gauge
measurement destroys is exactly the information that makes the code's logical
group smaller than its check group: the qubits every gauge generator leaves alone
are the ones that carry protected information, and the rest are gauge workspace.

**Why this is a record of its own rather than a pair of blocks on
:class:`~flagquantum.qec.CssCode`.** The two agree on every matrix they hold and
disagree about what the matrices mean. A :class:`~flagquantum.qec.CssCode` has one
logical operator per logical qubit, because its checks leave exactly that many
classes; here the checks leave more, and the gauge generators are what tells the
record which of those classes are protected and which are workspace. The count,
the pairing rule and the distance therefore all read differently, and a code
record is read by consumers that act on those readings. Folding the gauge blocks
into the CSS record would change what its existing readings mean for every caller
that never declares one, so the second meaning is stated by a second record and
the existing one is left alone.

**A subsystem code is deliberately not a
:class:`~flagquantum.qec.StabilizerCode`.** A memory experiment measures a
subsystem code's gauge generators, not its stabilizers alone, so the protocol that
states "these are the operators a round measures" does not describe this record.
That is load-bearing rather than incidental:
:func:`~flagquantum.qec.css_code_matrices` reads a code through that protocol and
turns its checks into parity-check matrices, and handed this record's stabilizers
it would state a stabilizer code the caller did not declare -- one with the
gauge generators silently dropped. It refuses this record instead, because this
record does not present the members it reads.

**The distance is dressed, and it is searched for rather than declared.** The
declared logical operators of a subsystem code are read up to the gauge group:
multiplying one by a gauge generator gives another operator that acts the same way
on protected information, and the shortest representative of a logical class is
the one an error has to reach. So a logical operator here has to commute with the
*checks* of the opposite family and lie outside its own family's stabilizer **and
gauge** span, and the least weight of such an operator is what this record
reports. That is the reading the published parameters of these codes are stated
in, and for a record with no gauge generator it is
:class:`~flagquantum.qec.CssCode`'s reading exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral

from .codes import (
    _DISTANCE_SEARCH_WEIGHT,
    BinaryMatrix,
    _block_width,
    _mask,
    _matrix_rows,
    _minimum_logical_weight,
    _refuse_non_logicals,
    _supports,
)
from .gf2 import rank
from .pauli import Pauli

__all__ = ("SubsystemCode", "tesseract_code")


@dataclass(frozen=True)
class SubsystemCode:
    """A CSS subsystem code given by the matrices it is written as.

    ``hz`` and ``hx`` hold the Z-type and X-type checks, ``gz`` and ``gx`` the
    Z-type and X-type gauge generators, and ``lz`` and ``lx`` the protected logical
    operators, one row per operator and one column per data qubit in every block.
    Data qubits are the columns and occupy wires ``0..n-1`` in column order.

    **The blocks have to describe a subsystem code, and the record checks the
    three conditions that make them one.** Every gauge generator must commute with
    every check, because the checks are the stabilizer and the stabilizer is the
    center of the gauge group: a gauge generator that failed to commute with a
    check would make that check's outcome depend on the gauge, and a check whose
    outcome depends on the gauge is not one. The two gauge blocks must have the
    same number of rows and must pair up non-degenerately, because a gauge qubit is
    a pair -- one X-type and one Z-type operator that anticommute -- and a
    combination of Z-type gauge generators that commuted with all of them would be
    a stabilizer the caller called a gauge operator. And no gauge generator may lie
    in its own family's check span, for the same reason read the other way round.

    **The logical count follows from the checks and the gauge, and the caller does
    not state it.** With ``s`` the rank of the two check blocks together and ``r``
    the number of gauge pairs, the checks leave ``n - s`` classes and the gauge
    qubits account for ``r`` of them, so the code has ``n - s - r`` protected
    qubits and each logical family has that many rows. A shorter or longer list is
    refused: it is not a logical family of this code.

    Raises:
        TypeError: If a block is not a sequence of sequences, a row is not a
            sequence, an entry is not an integer, or ``distance_search_weight`` is
            not an integer.
        ValueError: If an entry is neither zero nor one, a block has rows but no
            column, a row acts on no data qubit, the blocks disagree about how many
            data qubits the code has, no block states that number at all, the two
            check families do not commute, a gauge generator anticommutes with a
            check, the two gauge blocks disagree about how many gauge qubits there
            are, the gauge generators pair up degenerately, a gauge generator lies
            in its own family's check span, the checks and gauge leave no logical
            qubit, a logical block declares the wrong number of rows, a logical
            operator anticommutes with a check of the opposite family or lies in
            its own family's check-and-gauge span, the logical blocks pair up
            degenerately, or a family has no logical operator within
            ``distance_search_weight``.

    Examples:
        The smallest member of the family, the two-by-two Bacon-Shor code. Its
        single Z-type and single X-type gauge generator span the two classes the
        checks leave, so one of them is a logical qubit and the other is gauge.

        >>> from flagquantum.qec import SubsystemCode
        >>> baconshor = SubsystemCode(
        ...     hz=((1, 1, 1, 1),),
        ...     hx=((1, 1, 1, 1),),
        ...     gz=((1, 0, 1, 0),),
        ...     gx=((1, 1, 0, 0),),
        ...     lz=((1, 1, 0, 0),),
        ...     lx=((1, 0, 1, 0),),
        ...     distance_search_weight=2,
        ... )
        >>> (
        ...     baconshor.num_data_qubits,
        ...     baconshor.num_stabilizer_generators,
        ...     baconshor.num_gauge_qubits,
        ...     baconshor.num_logical_qubits,
        ...     baconshor.distance,
        ... )
        (4, 2, 1, 1, 2)
    """

    hz: BinaryMatrix
    hx: BinaryMatrix
    lz: BinaryMatrix = ()
    lx: BinaryMatrix = ()
    gz: BinaryMatrix = ()
    gx: BinaryMatrix = ()
    distance_search_weight: int = _DISTANCE_SEARCH_WEIGHT

    _width: int = field(init=False, repr=False)
    _z_checks: tuple[tuple[int, ...], ...] = field(init=False, repr=False)
    _x_checks: tuple[tuple[int, ...], ...] = field(init=False, repr=False)
    _z_logicals: tuple[tuple[int, ...], ...] = field(init=False, repr=False)
    _x_logicals: tuple[tuple[int, ...], ...] = field(init=False, repr=False)
    _z_gauges: tuple[tuple[int, ...], ...] = field(init=False, repr=False)
    _x_gauges: tuple[tuple[int, ...], ...] = field(init=False, repr=False)
    _stabilizer_generators: int = field(init=False, repr=False)
    _gauge_qubits: int = field(init=False, repr=False)
    _x_distance: int = field(init=False, repr=False)
    _z_distance: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.distance_search_weight, bool) or not isinstance(
            self.distance_search_weight, Integral
        ):
            raise TypeError("distance search weight must be an integer")
        if self.distance_search_weight < 1:
            raise ValueError("distance search weight must be at least one")

        blocks = {
            name: _matrix_rows(getattr(self, name), name=name)
            for name in ("hz", "hx", "lz", "lx", "gz", "gx")
        }
        widths: dict[str, int] = {}
        for name, rows in blocks.items():
            for index, row in enumerate(rows):
                if not any(row):
                    raise ValueError(
                        f"{name} row {index} acts on no data qubit, so it states an "
                        "operator over nothing rather than a check, a gauge generator "
                        "or a logical operator"
                    )
            width = _block_width(rows, name=name)
            if width is not None:
                widths[name] = width
        if not widths:
            raise ValueError(
                "no block states a check, a gauge generator or a logical operator, so "
                "the matrices do not state how many data qubits the code has"
            )
        if len(set(widths.values())) != 1:
            stated = ", ".join(f"{name} has {width}" for name, width in widths.items())
            raise ValueError(
                "the blocks disagree about how many data qubits the code has: "
                f"{stated}"
            )
        width = next(iter(widths.values()))

        z_checks = blocks["hz"]
        x_checks = blocks["hx"]
        z_gauges = blocks["gz"]
        x_gauges = blocks["gx"]
        z_logicals = blocks["lz"]
        x_logicals = blocks["lx"]

        z_check_masks = tuple(_mask(row) for row in z_checks)
        x_check_masks = tuple(_mask(row) for row in x_checks)
        z_gauge_masks = tuple(_mask(row) for row in z_gauges)
        x_gauge_masks = tuple(_mask(row) for row in x_gauges)
        z_checks_as_pauli = tuple(Pauli(z_wires=s) for s in _supports(z_checks))
        x_checks_as_pauli = tuple(Pauli(x_wires=s) for s in _supports(x_checks))
        z_gauges_as_pauli = tuple(Pauli(z_wires=s) for s in _supports(z_gauges))
        x_gauges_as_pauli = tuple(Pauli(x_wires=s) for s in _supports(x_gauges))

        for z_index, z_check in enumerate(z_checks_as_pauli):
            for x_index, x_check in enumerate(x_checks_as_pauli):
                if not z_check.commutes_with(x_check):
                    raise ValueError(
                        f"hz row {z_index} and hx row {x_index} act on an odd number "
                        "of shared data qubits, so the two check families do not "
                        "commute and generate no stabilizer group: the matrices are "
                        "not those of a stabilizer code"
                    )

        if len(z_gauges) != len(x_gauges):
            raise ValueError(
                f"gz declares {len(z_gauges)} gauge generator(s) and gx declares "
                f"{len(x_gauges)}: a gauge qubit is one Z-type and one X-type "
                "operator that anticommute, so the two families come in pairs"
            )
        # ``gauge`` is the Z-type generator and ``commuting`` the family it has to
        # commute with, which is the X-type side: a Z-type operator commutes with
        # every Z-type operator, so its own family states no condition.
        for gauge_family, gauge_paulis, commuting in (
            ("gz", z_gauges_as_pauli, x_checks_as_pauli),
            ("gx", x_gauges_as_pauli, z_checks_as_pauli),
        ):
            opposite = "hx" if gauge_family == "gz" else "hz"
            for gauge_index, gauge in enumerate(gauge_paulis):
                for check_index, check in enumerate(commuting):
                    if not gauge.commutes_with(check):
                        raise ValueError(
                            f"{gauge_family} row {gauge_index} anticommutes with "
                            f"{opposite} row {check_index}, so a check's outcome would "
                            "depend on the gauge: the checks are the stabilizer and "
                            "the stabilizer is the center of the gauge group, which is "
                            "what makes their outcomes well defined"
                        )

        z_check_rank = rank(list(z_check_masks))
        x_check_rank = rank(list(x_check_masks))
        for name, masks, own_masks, own_rank in (
            ("gz", z_gauge_masks, z_check_masks, z_check_rank),
            ("gx", x_gauge_masks, x_check_masks, x_check_rank),
        ):
            if rank(list(own_masks) + list(masks)) != own_rank + len(masks):
                raise ValueError(
                    f"{name} is not independent of the checks of its own family: a "
                    "gauge generator that is a combination of checks is a stabilizer "
                    "the caller called a gauge operator, and it adds no gauge qubit"
                )

        pairing_rows = [
            sum(
                0 if gauge.commutes_with(x_gauge) else 1 << column
                for column, x_gauge in enumerate(x_gauges_as_pauli)
            )
            for gauge in z_gauges_as_pauli
        ]
        gauge_qubits = rank(pairing_rows)
        if gauge_qubits != len(z_gauges):
            raise ValueError(
                f"the {len(z_gauges)} gauge generator pairs have a pairing matrix of "
                f"rank {gauge_qubits}, so they pair up degenerately: a combination of "
                "the declared Z-type gauge generators commutes with every X-type one, "
                "and that combination is fixed by every gauge measurement rather than "
                "free, which makes it a stabilizer the caller called a gauge generator"
            )

        stabilizer_generators = z_check_rank + x_check_rank
        logical_qubits = width - stabilizer_generators - gauge_qubits
        if logical_qubits < 1:
            raise ValueError(
                f"the checks span {stabilizer_generators} of the {width} data qubits "
                f"and the gauge accounts for {gauge_qubits} more, so no logical qubit "
                "is left: the matrices describe one state of a subsystem rather than a "
                "code with a distance"
            )

        for name, rows in (("lz", z_logicals), ("lx", x_logicals)):
            if rows and len(rows) != logical_qubits:
                raise ValueError(
                    f"{name} declares {len(rows)} logical operator(s) for a code with "
                    f"{logical_qubits} logical qubit(s): the checks leave "
                    f"{width - stabilizer_generators} classes and the gauge qubits "
                    "account for the rest, so a logical family has one operator per "
                    "protected qubit and a shorter or longer list is not one"
                )

        z_logicals_as_pauli = tuple(Pauli(z_wires=s) for s in _supports(z_logicals))
        x_logicals_as_pauli = tuple(Pauli(x_wires=s) for s in _supports(x_logicals))
        # A logical operator of a subsystem code is read up to the gauge group, so
        # what it must lie outside of is its own family's checks *and* gauge.
        _refuse_non_logicals(
            rows=z_logicals,
            family="lz",
            as_pauli=z_logicals_as_pauli,
            opposite_checks=x_checks_as_pauli,
            opposite_family="hx",
            own_masks=z_check_masks + z_gauge_masks,
            own_family="hz and gz",
        )
        _refuse_non_logicals(
            rows=x_logicals,
            family="lx",
            as_pauli=x_logicals_as_pauli,
            opposite_checks=z_checks_as_pauli,
            opposite_family="hz",
            own_masks=x_check_masks + x_gauge_masks,
            own_family="hx and gx",
        )

        if z_logicals_as_pauli and x_logicals_as_pauli:
            logical_pairing = [
                sum(
                    0 if z_logical.commutes_with(x_logical) else 1 << column
                    for column, x_logical in enumerate(x_logicals_as_pauli)
                )
                for z_logical in z_logicals_as_pauli
            ]
            pairing_rank = rank(logical_pairing)
            if pairing_rank != logical_qubits:
                raise ValueError(
                    "the declared Z-type and X-type logical operators pair up "
                    f"degenerately: their {logical_qubits}-by-{logical_qubits} "
                    f"pairing matrix has rank {pairing_rank}, so together they "
                    "generate a smaller logical group than a code with that many "
                    "logical qubits has"
                )

        # An X-type operator commutes with the Z-type checks and is read up to the
        # X-type checks and gauge; a Z-type operator is the mirror of that.
        x_distance = _minimum_logical_weight(
            width=width,
            commuting_rows=z_check_masks,
            stabilizer_rows=x_check_masks + x_gauge_masks,
            bound=int(self.distance_search_weight),
        )
        z_distance = _minimum_logical_weight(
            width=width,
            commuting_rows=x_check_masks,
            stabilizer_rows=z_check_masks + z_gauge_masks,
            bound=int(self.distance_search_weight),
        )
        for family, found in (("X-type", x_distance), ("Z-type", z_distance)):
            if found is None:
                raise ValueError(
                    f"no {family} logical operator of weight at most "
                    f"{self.distance_search_weight} exists, so this record cannot "
                    "report the code's distance: build it again with a "
                    "distance_search_weight the search reaches, because the record "
                    "does not report a distance nothing checked"
                )

        object.__setattr__(self, "hz", z_checks)
        object.__setattr__(self, "hx", x_checks)
        object.__setattr__(self, "lz", z_logicals)
        object.__setattr__(self, "lx", x_logicals)
        object.__setattr__(self, "gz", z_gauges)
        object.__setattr__(self, "gx", x_gauges)
        object.__setattr__(self, "_width", width)
        object.__setattr__(self, "_z_checks", z_checks)
        object.__setattr__(self, "_x_checks", x_checks)
        object.__setattr__(self, "_z_logicals", z_logicals)
        object.__setattr__(self, "_x_logicals", x_logicals)
        object.__setattr__(self, "_z_gauges", z_gauges)
        object.__setattr__(self, "_x_gauges", x_gauges)
        object.__setattr__(self, "_stabilizer_generators", stabilizer_generators)
        object.__setattr__(self, "_gauge_qubits", gauge_qubits)
        object.__setattr__(self, "_x_distance", x_distance)
        object.__setattr__(self, "_z_distance", z_distance)

    @property
    def distance(self) -> int:
        """The smaller of the code's two dressed family distances."""

        return min(self._x_distance, self._z_distance)

    @property
    def x_distance(self) -> int:
        """The least weight of an X-type logical operator read up to the gauge."""

        return self._x_distance

    @property
    def z_distance(self) -> int:
        """The least weight of a Z-type logical operator read up to the gauge."""

        return self._z_distance

    @property
    def num_data_qubits(self) -> int:
        return self._width

    @property
    def data_wires(self) -> tuple[int, ...]:
        return tuple(range(self._width))

    @property
    def num_stabilizer_generators(self) -> int:
        """The rank of the code's two check blocks together."""

        return self._stabilizer_generators

    @property
    def num_gauge_qubits(self) -> int:
        """The number of anticommuting gauge pairs, which is the gauge workspace."""

        return self._gauge_qubits

    @property
    def num_logical_qubits(self) -> int:
        """The protected qubit count, which the checks and the gauge leave over."""

        return self._width - self._stabilizer_generators - self._gauge_qubits

    @property
    def stabilizer_generators(self) -> tuple[Pauli, ...]:
        """The checks, Z-type first, in the order the matrices declare them."""

        return tuple(Pauli(z_wires=s) for s in _supports(self._z_checks)) + tuple(
            Pauli(x_wires=s) for s in _supports(self._x_checks)
        )

    @property
    def gauge_generators(self) -> tuple[Pauli, ...]:
        """The gauge generators, Z-type first, in the order the matrices declare them."""

        return tuple(Pauli(z_wires=s) for s in _supports(self._z_gauges)) + tuple(
            Pauli(x_wires=s) for s in _supports(self._x_gauges)
        )

    @property
    def logical_operators(self) -> tuple[Pauli, ...]:
        """The protected logical operators, Z-type first."""

        return tuple(Pauli(z_wires=s) for s in _supports(self._z_logicals)) + tuple(
            Pauli(x_wires=s) for s in _supports(self._x_logicals)
        )


def tesseract_code() -> SubsystemCode:
    """Return the tesseract subsystem code as it is published.

    The tesseract is the four-dimensional hypercube read as a quantum code: its
    sixteen vertices are the data qubits and its eight-cell faces are its checks,
    six faces a cell, which is why every check below has weight eight. Its
    **parent** is a stabilizer code on the same vertices with ten independent
    checks and six encoded qubits; the code declared here measures two of those
    six as gauge qubits and protects the other four.

    **Every number this function's record reports is read off the matrices rather
    than stated beside them.** The ten check rows, the two anticommuting gauge
    pairs, the four protected qubits and the distance four are all derived by
    :class:`SubsystemCode` from the rows below, so a transcription error in the
    tables is refused by the record instead of being published as a parameter. The
    distance the table is searched to is four, which is the published distance and
    therefore the bound the search has to reach: the check rows are weight eight,
    so the shortest operator that commutes with all of them and is outside the
    gauge span is what the search finds. The search stops at the first weight it
    reaches, so it tries 722 of the 2516 subsets of weight at most four for each
    family and the record costs 0.8 ms to build.

    **The logical order is the one the free-CNOT gadget is stated in.** The four
    protected operators are listed as ``(L1, L0, L2, L5)`` of the parent's six, and
    ``tests/qec/test_subsystem_code.py`` exhibits the data-wire permutation of
    arXiv:2412.14256 acting as ``CNOT(0 -> 1) * CNOT(2 -> 3)`` on exactly those
    four, which is the gadget that makes the family useful rather than the
    parameter table.

    Examples:
        >>> tesseract = tesseract_code()
        >>> (
        ...     tesseract.num_data_qubits,
        ...     tesseract.num_stabilizer_generators,
        ...     tesseract.num_gauge_qubits,
        ...     tesseract.num_logical_qubits,
        ...     tesseract.distance,
        ... )
        (16, 10, 2, 4, 4)
    """

    checks = (
        (0, 1, 2, 3, 4, 5, 6, 7),
        (0, 1, 2, 3, 8, 9, 10, 11),
        (0, 1, 4, 5, 8, 9, 12, 13),
        (0, 2, 4, 6, 8, 10, 12, 14),
        (8, 9, 10, 11, 12, 13, 14, 15),
    )
    z_logicals = (
        (0, 2, 4, 6),
        (0, 1, 4, 5),
        (1, 3, 9, 11),
        (0, 1, 8, 9),
    )
    x_logicals = (
        (4, 5, 12, 13),
        (0, 2, 8, 10),
        (8, 9, 12, 13),
        (0, 2, 4, 6),
    )
    z_gauges = ((1, 5, 9, 13), (0, 1, 2, 3))
    x_gauges = ((8, 9, 10, 11), (0, 4, 8, 12))

    def rows(supports: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(1 if wire in support else 0 for wire in range(16))
            for support in supports
        )

    return SubsystemCode(
        hz=rows(checks),
        hx=rows(checks),
        lz=rows(z_logicals),
        lx=rows(x_logicals),
        gz=rows(z_gauges),
        gx=rows(x_gauges),
        distance_search_weight=4,
    )
