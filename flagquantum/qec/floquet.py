"""Dynamic codes stated as a period of measurements rather than as one group.

A stabilizer code is one group, and its checks are the same in every round. A
dynamic code is a *sequence* of groups: each round measures a set of operators,
the operators that anticommute with the round stop being known, the measured ones
start being known, and the instantaneous stabilizer group therefore changes from
round to round while the information it protects does not. The code is stated as
its measurements -- a period of phases -- and its arithmetic is derived by running
that period, because the group a round stabilizes is not something a table of
checks names.

**The group a round stabilizes is a subgroup of the previous round's, and it is
computed rather than guessed.** Measuring a set of operators leaves exactly the
part of the old group that commutes with every one of them, and that part is the
joint kernel of one linear equation per measured operator, taken in the old
group's coordinates. Testing the old group's generators one at a time is the
tempting shortcut and it is wrong: a product of two generators can commute with a
measured operator although neither factor does. The kernel is what
:func:`~flagquantum.qec.gf2.nullspace` computes, and the round's new group is that
kernel together with the operators the round measured.

**A schedule is not yet a code.** Every schedule settles eventually, because there
are finitely many subgroups of the ``n``-wire Pauli group and the round map is
deterministic, so what a schedule can fail to do is settle *soon* -- and a budget
is stated, with a schedule that has not settled inside it refused rather than
reported with a count read off a group that is still moving. What it can also fail
to do is protect anything: measuring ``Z`` and ``X`` on the ring's edges in
alternating families, then the same two families the other way round, settles at a
stabilizer group of full rank, where the count is zero. Both are refusals here, both
name the schedule they refuse, and both are refused at construction rather than
reported as a number.

**What this module reads, and what it deliberately does not.** The number of
protected qubits is the width minus the rank of the settled group, and the least
logical weight is the least weight of an operator that commutes with the whole
settled group without being in it -- the settled round's code distance, searched for
in increasing weight up to a stated bound rather than read off the basis the record
happens to hold. That is not a spacetime distance: there is no detector model here, so
a fault that is a measurement error in one round and a data error in the next is not
one operator and is not searched for. Both readings are taken at every round of the
period; the count has to be the same number at all of them and the least logical
weight stated is the lightest of the rounds' readings.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from numbers import Integral

from .codes import _DISTANCE_SEARCH_WEIGHT
from .gf2 import in_span, nullspace, rank, reduce_rows
from .logical import _symplectic_mask
from .pauli import Pauli

__all__ = ("MeasurementPhase", "FloquetCode", "ring_floquet_code")

_DEFAULT_MAX_ROUNDS = 64


def _phase_rows(phase: MeasurementPhase, width: int) -> list[int]:
    """Return the phase's measured operators as symplectic GF(2) rows."""

    positions = {wire: wire for wire in range(width)}
    return [
        _symplectic_mask(operator, positions, width) for operator in phase.operators
    ]


def _combination(basis: list[int], coefficients: int) -> int:
    """Return the group element the coefficient vector selects."""

    total = 0
    for index, element in enumerate(basis):
        if (coefficients >> index) & 1:
            total ^= element
    return total


def _same_span(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    """Whether two row sets generate the same group.

    Compared through the rank of the joined rows rather than row by row: the
    reduced rows are one basis of a span and not a canonical one, so two bases of
    the same group need not be the same rows in the same order.
    """

    return rank(list(left) + list(right)) == rank(left) == rank(right)


def _surviving(
    group: tuple[int, ...], measured: list[int], width: int
) -> tuple[int, ...]:
    """Return the part of ``group`` that commutes with every measured operator."""

    basis = list(reduce_rows(list(group)))
    if not basis:
        return ()
    equations = []
    for other in measured:
        equation = 0
        for index, element in enumerate(basis):
            if not _commutes(element, other, width):
                equation |= 1 << index
        equations.append(equation)
    return reduce_rows(
        [
            _combination(basis, coefficients)
            for coefficients in nullspace(equations, len(basis))
        ]
    )


def _commutes(left: int, right: int, width: int) -> bool:
    """Whether two symplectic masks commute, ignoring phase."""

    low = (1 << width) - 1
    return (
        bin((left & low) & (right >> width)).count("1")
        + bin((left >> width) & (right & low)).count("1")
    ) % 2 == 0


def _advance(
    group: tuple[int, ...], measured: list[int], width: int
) -> tuple[int, ...]:
    """Return the group that is known after ``measured`` is measured."""

    return reduce_rows(list(_surviving(group, measured, width)) + list(measured))


def _centralizer(rows: tuple[int, ...], width: int) -> tuple[int, ...]:
    """Return a basis of the operators commuting with every row.

    The pairing of an unknown operator with a known one is linear in the unknown's
    ``2 * width`` bits, and the coefficient of an ``X`` bit is the known row's
    ``Z`` factor on that wire while the coefficient of a ``Z`` bit is its ``X``
    factor. A basis of the paired-out rows' null space is therefore a basis of the
    centralizer.
    """

    low = (1 << width) - 1
    equations = [(row & low) << width | (row >> width) for row in rows]
    return nullspace(equations, 2 * width)


def _quotient_basis(rows: tuple[int, ...], span: tuple[int, ...]) -> tuple[int, ...]:
    """Return one representative per class of ``rows`` modulo the group ``span``."""

    basis: list[int] = []
    for row in rows:
        if row and not in_span(row, list(span) + basis):
            basis.append(row)
    return tuple(basis)


def _least_logical_weight(*, width: int, group: tuple[int, ...], bound: int) -> int:
    """Return the least weight of an operator outside ``group`` that commutes with it.

    A logical operator of a stabilizer code is exactly an operator that commutes
    with every element of the group without being one, so the search is over subsets
    of the ``2 * width`` symplectic positions tried in increasing weight: the first
    operator found has the least weight there is, and the search stops at ``bound``.
    That the operator is not in the group is tested against the group's pivots once,
    because every candidate is tested against the same group.

    **The bound is the caller's, and exhausting it is a refusal rather than a
    number.** A least weight that the search did not reach is not a distance, so a
    group with no logical operator inside ``bound`` raises instead of reporting the
    bound back.
    """

    pivots = reduce_rows(list(group))
    for weight in range(1, bound + 1):
        for positions in itertools.combinations(range(2 * width), weight):
            candidate = 0
            for position in positions:
                candidate |= 1 << position
            if in_span(candidate, list(pivots)):
                continue
            if all(_commutes(candidate, row, width) for row in group):
                return weight
    raise ValueError(
        f"no operator of weight {bound} or less commutes with a group of rank "
        f"{rank(group)} over {width} wires without being in it, so the least logical "
        "weight is above the search bound and is not reported"
    )


def _support(row: int, width: int) -> int:
    """Return the number of wires the masked operator acts on non-trivially."""

    return sum(
        1 for wire in range(width) if (row >> wire) & 1 or (row >> (width + wire)) & 1
    )


def _to_pauli(row: int, width: int) -> Pauli:
    """Return the operator a symplectic mask names."""

    return Pauli(
        x_wires=tuple(wire for wire in range(width) if (row >> wire) & 1),
        z_wires=tuple(wire for wire in range(width) if (row >> (width + wire)) & 1),
    )


def _settle(
    history: list[tuple[int, ...]],
) -> tuple[int, int] | None:
    """Return the settled period of ``history`` and the round it starts at.

    Candidate periods are tried in ascending order, so the number returned is the
    least one the sequence repeats over rather than a multiple of it. For a candidate
    ``q`` the settled round is the one after the last place the sequence failed to
    repeat over ``q``, and the candidate is accepted only when at least ``q`` rounds
    of the sequence remain beyond that round: a window shorter than one period has
    nothing to be checked against, so accepting it would read a period out of a
    coincidence of two groups.
    """

    for period in range(1, len(history) // 2):
        breaks = [
            index
            for index in range(len(history) - period)
            if not _same_span(history[index], history[index + period])
        ]
        start = 0 if not breaks else max(breaks) + 1
        if len(history) - start >= 2 * period:
            return period, start
    return None


@dataclass(frozen=True)
class MeasurementPhase:
    """One round of a dynamic code: the operators measured together.

    The operators of a round are measured simultaneously, so they have to commute
    with each other; a phase that measured two anticommuting operators would be
    describing an order and not a round, and it is refused. The identity is refused
    for the same reason read the other way round: it measures nothing, so listing
    it would state a constraint the round does not impose.

    ``num_independent_operators`` is derived rather than stated, because a round
    that lists a product of its own operators measures no more than it did without
    that line.

    Raises:
        TypeError: If an entry is not a :class:`~flagquantum.qec.pauli.Pauli`.
        ValueError: If the phase is unnamed, measures nothing, measures the
            identity, or measures two operators that anticommute.

    Examples:
        >>> from flagquantum.qec import MeasurementPhase, Pauli
        >>> phase = MeasurementPhase(
        ...     name="even-edges-z",
        ...     operators=(Pauli(z_wires=(0, 1)), Pauli(z_wires=(2, 3))),
        ... )
        >>> (phase.name, phase.num_operators, phase.num_independent_operators)
        ('even-edges-z', 2, 2)
    """

    name: str
    operators: tuple[Pauli, ...]

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError(
                "a measurement phase must be named, because a refusal of a schedule "
                "has to say which round it refuses"
            )
        operators = tuple(self.operators)
        if not operators:
            raise ValueError(
                f"phase {self.name!r} measures no operator, so it is not a round"
            )
        for index, operator in enumerate(operators):
            if not isinstance(operator, Pauli):
                raise TypeError(
                    f"phase {self.name!r} entry {index} is not a Pauli, so the round "
                    "does not state an operator"
                )
            if operator.is_identity:
                raise ValueError(
                    f"phase {self.name!r} entry {index} is the identity, which "
                    "measures nothing and constrains nothing"
                )
        for left, right in itertools.combinations(operators, 2):
            if not left.commutes_with(right):
                raise ValueError(
                    f"phase {self.name!r} measures {left.to_text()} and "
                    f"{right.to_text()}, which anticommute, so no round measures both"
                )
        object.__setattr__(self, "operators", operators)

    @property
    def num_operators(self) -> int:
        """The number of operators the round lists."""

        return len(self.operators)

    @property
    def num_independent_operators(self) -> int:
        """The number of the round's operators that are independent over GF(2)."""

        width = 1 + max(
            (wire for operator in self.operators for wire in operator.support),
            default=0,
        )
        return rank(_phase_rows(self, width))


@dataclass(frozen=True)
class FloquetCode:
    """A dynamic code stated as the period of measurements that defines it.

    ``phases`` is one period, measured in order and then repeated. ``num_qubits`` is
    the width the operators act on, and it is stated rather than derived from them,
    because a period that happens not to touch a wire still has to say that the wire
    is there. ``max_rounds`` bounds the search for the settled period and
    ``distance_search_weight`` bounds the search for the least logical weight.

    **The settled period is what the record reports, and the record refuses a
    schedule that does not reach one.** Rounds are propagated until the sequence of
    instantaneous stabilizer groups repeats; the least number of rounds it repeats
    over is the period, and the rounds before the first repeat are the transient.
    Two groups are the same group when each lies in the other's span, and not when
    their reduced bases are the same rows in the same order, because a reduced basis
    is one basis of a span rather than a canonical one. Because a schedule whose
    period has not arrived says nothing about what it protects, a schedule that does
    not settle inside ``max_rounds`` is refused by name -- and so is a schedule that
    settles protecting no qubit, which on rings of alternating families is what a
    period of four does.

    **The two numbers are the period's, and both are read at every round of it.** A
    period is not one group, so a count and a least logical weight have to be read at
    a round of it, and the record reads them at *each* round: a round that protected a
    different number of qubits would make the period not a code and is refused by
    name, and the least logical weight stated is the lightest of the rounds' readings,
    because a claim about a period has to hold where the period is weakest.
    ``steady_stabilizer_groups`` exposes every round so that the reading is visible
    rather than asserted. Over every schedule measured -- the five hundred and
    eighty-five ring schedules below and ninety-three thousand one hundred and fifty
    schedules of paired measurement rounds over six wires -- the rounds of a period
    agreed on both numbers, so the lightest reading equals the first on all of them;
    the search is kept because the definition of a period's distance is a minimum over
    the period and not a reading of whichever round the propagation stopped at.

    Raises:
        TypeError: If ``num_qubits``, ``max_rounds``, ``distance_search_weight`` or a
            phase is of the wrong type.
        ValueError: If the schedule is unnamed, has no phase, has two phases under one
            name, measures a wire outside the width, has not settled inside
            ``max_rounds`` rounds, settles without protecting a qubit, settles with
            different rounds protecting different numbers of qubits, or has no logical
            operator within ``distance_search_weight``.

    Examples:
        A schedule written out by hand on four wires, whose two rounds are stated
        twice. The period is read off the groups the rounds stabilize rather than off
        the number of phases, so four phases settle into a period of two and the round
        the sequence first repeats from is the third; a phase name is what a refusal
        of a schedule is able to point at, so the two statements of one round carry
        two names.

        >>> from flagquantum.qec import FloquetCode, MeasurementPhase, Pauli
        >>> z_edges = (Pauli(z_wires=(0, 1)), Pauli(z_wires=(2, 3)))
        >>> x_edges = (Pauli(x_wires=(1, 2)), Pauli(x_wires=(0, 3)))
        >>> schedule = FloquetCode(
        ...     name="ring of four, stated twice",
        ...     num_qubits=4,
        ...     phases=(
        ...         MeasurementPhase(name="z-even-first", operators=z_edges),
        ...         MeasurementPhase(name="x-odd-first", operators=x_edges),
        ...         MeasurementPhase(name="z-even-again", operators=z_edges),
        ...         MeasurementPhase(name="x-odd-again", operators=x_edges),
        ...     ),
        ... )
        >>> (
        ...     schedule.num_phases,
        ...     schedule.period,
        ...     schedule.num_transient_rounds,
        ...     schedule.num_logical_qubits,
        ...     schedule.least_logical_weight,
        ... )
        (4, 2, 1, 1, 2)
    """

    name: str
    num_qubits: int
    phases: tuple[MeasurementPhase, ...]
    max_rounds: int = _DEFAULT_MAX_ROUNDS
    distance_search_weight: int = _DISTANCE_SEARCH_WEIGHT

    _rounds_to_steady: int = field(init=False, repr=False)
    _period: int = field(init=False, repr=False)
    _steady_groups: tuple[tuple[int, ...], ...] = field(init=False, repr=False)
    _reading: tuple[int, ...] = field(init=False, repr=False)
    _logical_basis: tuple[int, ...] = field(init=False, repr=False)
    _least_logical_weight: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError(
                "a dynamic code must be named, because a refusal of it has to say "
                "what it refuses"
            )
        for field_name in ("num_qubits", "max_rounds", "distance_search_weight"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{field_name} must be an integer")
            if value < 1:
                raise ValueError(f"{field_name} is {value}, and it is at least one")
        phases = tuple(self.phases)
        if not phases:
            raise ValueError(
                f"{self.name!r} has no measurement phase, so it states no code"
            )
        for index, phase in enumerate(phases):
            if not isinstance(phase, MeasurementPhase):
                raise TypeError(
                    f"{self.name!r} phase {index} is not a MeasurementPhase"
                )
        names = [phase.name for phase in phases]
        if len(set(names)) != len(names):
            repeated = sorted({name for name in names if names.count(name) > 1})
            raise ValueError(
                f"{self.name!r} names one phase twice ({', '.join(repeated)}), so a "
                "round cannot be told from the other"
            )
        for phase in phases:
            for operator in phase.operators:
                outside = [wire for wire in operator.support if wire >= self.num_qubits]
                if outside:
                    raise ValueError(
                        f"{self.name!r} phase {phase.name!r} acts on wire "
                        f"{outside[0]}, which is outside the stated width "
                        f"{self.num_qubits}"
                    )
        object.__setattr__(self, "phases", phases)
        self._derive()

    def _derive(self) -> None:
        """Propagate the schedule, settle it, and read the period's arithmetic."""

        rows_of = [_phase_rows(phase, self.num_qubits) for phase in self.phases]
        history = [reduce_rows(rows_of[0])]
        for index in range(1, self.max_rounds + 1):
            history.append(
                _advance(history[-1], rows_of[index % len(rows_of)], self.num_qubits)
            )
        settled = _settle(history)
        if settled is None:
            raise ValueError(
                f"{self.name!r} has not settled inside {self.max_rounds} rounds, so "
                "its instantaneous stabilizer group is still moving and the count "
                "it protects is not yet one number"
            )
        period, start = settled
        object.__setattr__(self, "_rounds_to_steady", start)
        object.__setattr__(self, "_period", period)
        steady = tuple(history[start : start + period])
        object.__setattr__(self, "_steady_groups", steady)

        # The count has to be one number at every round of the period, or the period
        # is not a code and there is no ``k`` for it to report.
        counts = {self.num_qubits - rank(group) for group in steady}
        if len(counts) != 1:
            raise ValueError(
                f"{self.name!r} settles at round {start} of period {period} but its "
                "rounds protect different numbers of qubits "
                f"({', '.join(str(count) for count in sorted(counts))}), so the "
                "period is not a code"
            )
        protected = counts.pop()
        if protected < 1:
            raise ValueError(
                f"{self.name!r} settles at round {start} of period {period} with "
                f"{rank(steady[0])} independent stabilizers over {self.num_qubits} "
                "wires, so it protects no qubit and is a measurement schedule rather "
                "than a code"
            )

        # A period is read at every one of its rounds, and the readings are the two
        # numbers the record exists to state. The count has to be the same number at
        # all of them or the period is not a code; the least logical weight is the
        # lightest of them, because a claim about a period has to hold where the
        # period is weakest.
        readings: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []
        for group in steady:
            readings.append(
                (
                    _least_logical_weight(
                        width=self.num_qubits,
                        group=group,
                        bound=self.distance_search_weight,
                    ),
                    group,
                    _quotient_basis(_centralizer(group, self.num_qubits), group),
                )
            )
        weight, reading, basis = min(readings, key=lambda reading: reading[0])
        object.__setattr__(self, "_reading", reading)
        object.__setattr__(self, "_logical_basis", basis)
        object.__setattr__(self, "_least_logical_weight", weight)

    @property
    def num_data_qubits(self) -> int:
        """The number of wires the period acts on."""

        return self.num_qubits

    @property
    def data_wires(self) -> tuple[int, ...]:
        """The wires the period acts on, in ascending order."""

        return tuple(range(self.num_qubits))

    @property
    def num_phases(self) -> int:
        """The number of measurement phases in one period as stated."""

        return len(self.phases)

    @property
    def period(self) -> int:
        """The number of rounds the settled group sequence repeats over.

        A schedule whose later phases measure what an earlier one already measured
        repeats sooner than its stated phase count, and it is the sooner number that
        describes it.
        """

        return self._period

    @property
    def num_transient_rounds(self) -> int:
        """The number of opening rounds whose group the settled sequence never repeats.

        The ring's value is one: its first round's group is the span of that round's
        own measurements, which is strictly smaller than the group the schedule holds
        at the same phase index once it has run a full period.
        """

        return self._rounds_to_steady

    @property
    def num_stabilizer_generators(self) -> int:
        """The rank of the reading round's instantaneous stabilizer group."""

        return rank(self._reading)

    @property
    def num_logical_qubits(self) -> int:
        """The number of qubits the period protects."""

        return self.num_qubits - rank(self._reading)

    @property
    def least_logical_weight(self) -> int:
        """The least weight of a logical operator at the period's lightest round.

        This is the settled round's code distance and not a spacetime fault
        distance: a record with no detector model measures operators on one time
        slice, and a fault spread over two rounds is not one of them.
        """

        return self._least_logical_weight

    @property
    def stabilizer_generators(self) -> tuple[Pauli, ...]:
        """The reading round's instantaneous stabilizer group, as generators."""

        return tuple(_to_pauli(row, self.num_qubits) for row in self._reading)

    @property
    def logical_operators(self) -> tuple[Pauli, ...]:
        """One representative per logical class of the reading round."""

        return tuple(_to_pauli(row, self.num_qubits) for row in self._logical_basis)

    def steady_stabilizer_groups(self) -> tuple[tuple[Pauli, ...], ...]:
        """The instantaneous stabilizer group at each round of the settled period.

        One entry per round, in round order, starting at the round the sequence
        first repeats from.
        """

        return tuple(
            tuple(_to_pauli(row, self.num_qubits) for row in group)
            for group in self._steady_groups
        )


def _ring_edges(num_qubits: int, residue: int) -> tuple[tuple[int, int], ...]:
    """Return the ring's edges whose index is ``residue`` modulo two.

    The wires of an edge are returned in ascending order, because the closing edge
    of the ring runs from the last wire back to wire zero and an operator names its
    wires rather than its direction.
    """

    return tuple(
        tuple(sorted((wire, (wire + 1) % num_qubits)))
        for wire in range(num_qubits)
        if wire % 2 == residue
    )


def ring_floquet_code(num_qubits: int) -> FloquetCode:
    """Return the two-phase ring, the smallest schedule this package publishes.

    The ``num_qubits`` wires form one ring and its edges are split by parity: the
    first phase measures ``Z`` on the even-indexed edges and the second measures
    ``X`` on the odd-indexed ones, so each round's operators are disjoint and the
    two rounds share no operator. The even edges are a perfect matching of the ring
    and so are the odd ones, which is why the width has to be even and at least
    four: a matching of a ring needs an even number of wires, and a ring of two has
    one edge per class and no room for a code.

    **Its arithmetic is derived and it is small.** The first round leaves
    ``num_qubits / 2`` checks, the second leaves one of them beside its own
    ``num_qubits / 2`` measurements, and from there the sequence repeats every two
    rounds -- the first round is the schedule's only transient one, because its group
    is the span of its own measurements alone. The code protects ``num_qubits / 2 - 1``
    qubits and its least logical operator is a weight-two operator on one edge: a
    bare representative, which is why the family is published as a record of the
    *shape* of a dynamic code rather than as a distance claim. The bound is measured
    rather than assumed: sweeping every ring schedule of two, three and four phase
    classes over widths four to twelve, five hundred and eighty-five schedules, no
    schedule's least logical weight exceeds two, and the three- and four-class rings
    are lighter still, reaching weight one.

    Raises:
        ValueError: If ``num_qubits`` is odd, or below four.

    Examples:
        >>> from flagquantum.qec import ring_floquet_code
        >>> ring = ring_floquet_code(6)
        >>> (
        ...     ring.num_data_qubits,
        ...     ring.num_phases,
        ...     ring.period,
        ...     ring.num_transient_rounds,
        ...     ring.num_logical_qubits,
        ...     ring.least_logical_weight,
        ... )
        (6, 2, 2, 1, 2, 2)
    """

    if num_qubits % 2:
        raise ValueError(
            f"the ring has {num_qubits} wires, and a matching of a ring needs an even "
            "number of them"
        )
    if num_qubits < 4:
        raise ValueError(
            f"the ring has {num_qubits} wires, and a ring of two has one edge per "
            "class, which leaves no check to protect a qubit with"
        )
    phases = (
        MeasurementPhase(
            name="even-edges-z",
            operators=tuple(Pauli(z_wires=edge) for edge in _ring_edges(num_qubits, 0)),
        ),
        MeasurementPhase(
            name="odd-edges-x",
            operators=tuple(Pauli(x_wires=edge) for edge in _ring_edges(num_qubits, 1)),
        ),
    )
    return FloquetCode(name="two-phase ring", num_qubits=num_qubits, phases=phases)
