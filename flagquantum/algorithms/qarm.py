"""Frequent-itemset fractions by amplitude estimation over a classically iterated database.

Association rule mining asks which itemsets recur across the transactions of a database.
The quantum route this module follows is the amplitude-estimation one of Yu, Gao, Wang and
Wen, *Physical Review A* **94**(4), 042311 (2016), DOI 10.1103/PhysRevA.94.042311,
arXiv:1605.07444v3. **The improvement that paper claims is quadratic in the number of
database queries and it is stated conditionally**, for the case ``M_f^(k) << M_c^(k)``; it
is not exponential. The exponential wording appears only in the earlier arXiv listing of
the same work, `arXiv:1512.02420`, and is not repeated here.

**This unit asks a smaller question than the paper's.** The paper mines the frequent
itemsets. What is read out here is one number: the fraction of the **items** whose support
-- the count of transactions containing the item -- meets a threshold. The database is the
caller's binary incidence matrix, and no rule-mining stage is part of this module.

**The premise is coherent entry-wise database access, and this unit does not meet it.** The
paper's count is a count of oracle calls, and the oracle it counts returns one database
entry per call, held in a qRAM. Neither is exercised here. The transactions are iterated
over **classically**: one controlled increment of the support register per transaction-item
membership, emitted from the incidence matrix as Python reads it. The access cost is
therefore paid explicitly rather than assumed away, and **no end-to-end advantage follows**:
the classical loop is precisely the part of the paper's assumption that its speed-up is
measured against, and it is this unit's headline limitation.

**The construction.** The evaluation register carries a uniform superposition over the
items -- one half-turn ``ry`` per item wire, which is the state a Hadamard on every item
wire prepares and which has a native controlled form -- and a support register, which the
circuit fills by adding one to it once per transaction whose itemset holds the
superposition's item. The marking operator flips the phase of the support values at or
above the threshold, so the marked subspace is exactly the frequent items, and its
amplitude is the frequent fraction. :func:`frequent_itemset_operator` builds the operator
and :func:`run_frequent_itemset` hands it to
:func:`~flagquantum.algorithms.amplitude_estimation.run_amplitude_estimation`, whose
estimate is that fraction.

**The item register is one wire per item-index bit**, so the item count is a power of two.
A uniform state over a count that is not one has no controlled preparation in this package,
and the item register's preparation has to be controllable for the Grover operator.

**The support register has to be wide enough, and a narrow one is refused rather than
wrapped.** The increment is a permutation of the register's own values, so a support the
register cannot hold comes back as another value, the mark then sees a support that can be
on the wrong side of the threshold, and the readout can come back wrong with nothing
raised. :func:`frequent_itemset_operator` therefore refuses a register that cannot hold the
largest support the database can produce, which is the number of transactions; the fewest
wires that can hold it is the default.

**The threshold is inclusive.** An item whose support equals the threshold is frequent, and
the marking operator is built over the support values at or above it.

This unit is demonstration scale. It makes no performance, capacity, or hardware claim, and
it does not select a runtime.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ..circuit import Circuit
from .amplitude_estimation import AmplitudeEstimationResult, run_amplitude_estimation
from .primitives import append_multi_controlled_x

__all__ = [
    "FrequentItemsetOperator",
    "frequent_itemset_operator",
    "run_frequent_itemset",
]

# The default sample size, matching the other sampling units in this package.
_DEFAULT_SHOTS = 4096

# The marking operator enumerates the support values at or above the threshold, and the
# classical transaction loop emits one increment per transaction-item pair. Both grow with
# the database, and this unit is written at demonstration scale: at most eight items and at
# most seven transactions.
_MAX_ITEM_WIRES = 3
_MAX_SUPPORT_WIRES = 3

# The item register's preparation: ``ry`` at half a turn takes ``|0>`` to the state a
# Hadamard on that wire takes it to, and ``ry`` has a controlled form where a Hadamard here
# does not.
_HALF_TURN = math.pi / 2.0


@dataclass(frozen=True, kw_only=True)
class FrequentItemsetOperator:
    """One transactional database, as the amplitude operator of its frequent fraction.

    The operator carries the database and the threshold, and it satisfies
    :class:`~flagquantum.algorithms.primitives.types.AmplitudeOperator`, so
    :func:`~flagquantum.algorithms.amplitude_estimation.run_amplitude_estimation` consumes
    it directly. Its evaluation register holds the uniform superposition over the items
    together with the support register that superposition is read against, so the amplitude
    of the marking operator's subspace is the fraction of the items whose support meets the
    threshold.

    **The supports are computed classically, here and in the circuit.** :attr:`support` is
    the count of transactions holding each item, read straight off the incidence matrix.
    The circuit reaches the same values by the same route -- one controlled increment per
    transaction whose itemset holds the item -- which is what makes the transaction loop
    classical rather than coherent. See the module docstring for what that costs.

    **The register layout, in wire order.** The item register comes first, one wire per
    item-index bit with the first wire the most significant; the support register follows
    it, first wire most significant as well; then the ancillas the multi-controlled X
    ladders borrow and restore; and last the flag wire the marking operator flips. The
    ancillas are ``|0>`` at the start of every ladder that borrows them, and each ladder
    restores what it borrows; the flag is ``|0>`` on entry to and on exit from each
    protocol method.

    ``kw_only`` is not optional here, and the reason is how a call site reads rather than
    what it would catch: the fields are a matrix, a threshold and a register width, so a
    positional spelling would read as a matrix followed by two bare numbers of unrelated
    kind. A transposition of those two numbers is not what ``kw_only`` protects against
    either: each is checked against its own bound and the two bounds overlap, so a pair
    written the wrong way round is refused by neither. Measured on a database of two
    two-item transactions, the pair the caller who means a threshold of 2 and a single
    support wire would write -- ``threshold=2`` with ``n_support_wires=1`` -- is refused as
    a register too narrow to hold every support, and the same pair written the other way
    round, a threshold of 1 with two support wires, constructs, with a threshold every item
    of that database meets.

    Attributes:
        incidence: The database, one row per transaction and one entry per item, each entry
            ``1`` where the transaction holds the item and ``0`` where it does not. Every
            row carries the same number of entries, and that count is a power of two of at
            least two.
        threshold: The least support a frequent item has, an integer in
            ``range(1, n_transactions + 1)``. The comparison is inclusive.
        n_support_wires: The width of the support register, at least the number of bits the
            transaction count needs and at most :data:`_MAX_SUPPORT_WIRES`.
    """

    incidence: tuple[tuple[int, ...], ...]
    threshold: int
    n_support_wires: int

    def __post_init__(self) -> None:
        """Reject an operator whose fields do not describe one database and threshold.

        Raises:
            ValueError: If ``incidence`` is not a tuple of rows, holds no transaction, holds
                a row whose width differs from the first's or is empty, or holds an entry
                that is neither ``0`` nor ``1``; if its item count is not a power of two of
                at least two, or exceeds the unit's bound; if ``threshold`` is not an
                integer in ``range(1, n_transactions + 1)``; or if ``n_support_wires`` is
                not an integer, cannot hold the largest support the database can produce, or
                exceeds the unit's bound.
        """
        rows = _validated_incidence(self.incidence)
        _validated_item_count(len(rows[0]))
        _validated_threshold(self.threshold, len(rows))
        _validated_support_width(self.n_support_wires, len(rows))

    @property
    def n_items(self) -> int:
        """The number of items the database has a column for."""
        return len(self.incidence[0])

    @property
    def n_transactions(self) -> int:
        """The number of transactions, one per row of the incidence matrix."""
        return len(self.incidence)

    @property
    def n_item_wires(self) -> int:
        """The width of the item register, one wire per bit of the item index."""
        return self.n_items.bit_length() - 1

    @property
    def support(self) -> tuple[int, ...]:
        """The support of each item, in item order.

        The support of an item is the number of transactions whose row holds it, which is
        the column's sum, read here by a classical pass over the matrix. The circuit's
        support register reaches the same values; see the class docstring.
        """
        return tuple(
            sum(transaction[item] for transaction in self.incidence)
            for item in range(self.n_items)
        )

    @property
    def n_wires(self) -> int:
        """The number of wires the operator's evaluation register carries."""
        return (
            self.n_item_wires
            + self.n_support_wires
            + _ladder_ancillas(self.n_item_wires, self.n_support_wires)
            + 1
        )

    def apply_plain(self, circuit: Circuit, wires: Sequence[int]) -> None:
        """Append the preparation ``A`` to ``circuit`` on ``wires``.

        ``A`` is the uniform superposition over the items followed by the transaction loop,
        so ``A|0...0>`` is the superposition of ``|item>|support>`` over every item.

        Args:
            circuit: The circuit to extend.
            wires: The evaluation register, this operator's ``n_wires`` wires.

        Raises:
            ValueError: If ``wires`` is not the evaluation register this operator describes.
        """
        item, support, pool, _ = self._layout(wires)
        for wire in item:
            circuit.gate("ry", wire, theta=_HALF_TURN)
        self._append_increments(circuit, None, item, support, pool, forward=True)

    def apply_a(self, circuit: Circuit, control: int, wires: Sequence[int]) -> None:
        """Append ``A`` controlled on ``control``.

        Args:
            circuit: The circuit to extend.
            control: The control wire, outside the evaluation register.
            wires: The evaluation register, this operator's ``n_wires`` wires.

        Raises:
            ValueError: If ``wires`` is not the evaluation register this operator
                describes, or if ``control`` is one of its wires.
        """
        item, support, pool, _ = self._layout(wires, control)
        for wire in item:
            circuit.gate("cry", (control, wire), theta=_HALF_TURN)
        self._append_increments(circuit, control, item, support, pool, forward=True)

    def apply_a_dagger(
        self, circuit: Circuit, control: int, wires: Sequence[int]
    ) -> None:
        """Append the adjoint of ``A``, controlled on ``control``.

        The adjoint is the exact gate-level reverse of :meth:`apply_a`: the transaction loop
        runs backwards, each increment becoming the decrement that undoes it, and the item
        register's rotations then come with the opposite turn.

        Args:
            circuit: The circuit to extend.
            control: The control wire, outside the evaluation register.
            wires: The evaluation register, this operator's ``n_wires`` wires.

        Raises:
            ValueError: If ``wires`` is not the evaluation register this operator
                describes, or if ``control`` is one of its wires.
        """
        item, support, pool, _ = self._layout(wires, control)
        self._append_increments(circuit, control, item, support, pool, forward=False)
        for wire in item:
            circuit.gate("cry", (control, wire), theta=-_HALF_TURN)

    def apply_mark(self, circuit: Circuit, control: int, wires: Sequence[int]) -> None:
        """Append the marking operator, controlled on ``control``.

        The marking operator flips the phase of the subspace whose support register holds a
        value at or above the threshold, and leaves every other subspace alone. It is built
        as the XOR of ``support >= threshold`` onto the flag wire, a phase flip on the
        control and the flag, and the same XOR again to take the flag back to ``|0>``.

        Args:
            circuit: The circuit to extend.
            control: The control wire, outside the evaluation register.
            wires: The evaluation register, this operator's ``n_wires`` wires.

        Raises:
            ValueError: If ``wires`` is not the evaluation register this operator
                describes, or if ``control`` is one of its wires.
        """
        _, support, pool, flag = self._layout(wires, control)
        _append_support_match(circuit, support, pool, flag, self.threshold)
        circuit.gate("cz", (control, flag))
        _append_support_match(circuit, support, pool, flag, self.threshold)

    def apply_zero_reflection(
        self, circuit: Circuit, control: int, wires: Sequence[int]
    ) -> None:
        """Append the reflection about the register's zero state, controlled on ``control``.

        The reflection is taken over the item and support registers, which are the registers
        ``A`` prepares: the ancillas and the flag are ``|0>`` wherever the state is
        reachable by ``A``, so reflecting over them as well would change nothing there.

        Args:
            circuit: The circuit to extend.
            control: The control wire, outside the evaluation register.
            wires: The evaluation register, this operator's ``n_wires`` wires.

        Raises:
            ValueError: If ``wires`` is not the evaluation register this operator
                describes, or if ``control`` is one of its wires.
        """
        item, support, pool, _ = self._layout(wires, control)
        logical = item + support
        for wire in logical:
            circuit.gate("x", wire)
        circuit.gate("h", logical[-1])
        controls = [control, *logical[:-1]]
        append_multi_controlled_x(
            circuit,
            controls,
            logical[-1],
            ancillas=pool[: max(len(controls) - 2, 0)],
        )
        circuit.gate("h", logical[-1])
        for wire in logical:
            circuit.gate("x", wire)

    def _layout(
        self, wires: Sequence[int], control: int | None = None
    ) -> tuple[list[int], list[int], list[int], int]:
        """Return the item register, the support register, the ancillas and the flag wire.

        Args:
            wires: The evaluation register as given.
            control: The control wire of the controlled form being appended, or ``None``.

        Returns:
            The four parts of the register, in the order the class docstring names them.

        Raises:
            ValueError: If ``wires`` is not the evaluation register this operator
                describes, or if ``control`` is one of its wires.
        """
        if len(wires) != self.n_wires:
            raise ValueError(
                f"this operator's evaluation register is {self.n_wires} wires wide, got "
                f"{len(wires)}; it carries the item register, the support register, the "
                "multi-controlled X ladder ancillas and the marking flag, in that order"
            )
        if control is not None and control in wires:
            raise ValueError(
                f"the control wire {control} is one of the evaluation register's; a "
                "controlled form is controlled on a wire outside the register it acts on"
            )
        n = self.n_item_wires
        b = self.n_support_wires
        ancillas = _ladder_ancillas(n, b)
        return (
            list(wires[:n]),
            list(wires[n : n + b]),
            list(wires[n + b : n + b + ancillas]),
            int(wires[n + b + ancillas]),
        )

    def _append_increments(
        self,
        circuit: Circuit,
        control: int | None,
        item: Sequence[int],
        support: Sequence[int],
        pool: Sequence[int],
        *,
        forward: bool,
    ) -> None:
        """Append one controlled increment or decrement per transaction-item pair.

        ``forward`` counts the transactions and their itemsets in order and adds one to the
        support register for each membership; ``not forward`` counts both backwards and
        takes one away, which is the gate-level reverse of the first and so its inverse.

        Args:
            circuit: The circuit to extend.
            control: The control wire, or ``None`` for the uncontrolled loop.
            item: The item register's wires.
            support: The support register's wires.
            pool: The ancilla wires the multi-controlled X ladders may borrow.
            forward: Which direction the database is counted in.
        """
        rows = self.incidence if forward else tuple(reversed(self.incidence))
        for row in rows:
            members = range(len(row)) if forward else reversed(range(len(row)))
            for member in members:
                if not row[member]:
                    continue
                _append_pattern_bracket(circuit, item, member)
                _append_support_step(circuit, control, item, support, pool, forward)
                _append_pattern_bracket(circuit, item, member)


def frequent_itemset_operator(
    incidence: torch.Tensor,
    *,
    threshold: int,
    n_support_wires: int | None = None,
) -> FrequentItemsetOperator:
    """Build the amplitude operator of one database's frequent fraction.

    The operator's evaluation register carries the uniform superposition over the items
    together with the support register, and its amplitude estimation estimate is the
    fraction of the items whose support meets ``threshold``. The construction is the module
    docstring's.

    The support register's width is derived from the transaction count when none is given:
    the largest support a database of ``n`` transactions can produce is ``n``, so the fewest
    wires that can hold every support are the bits ``n`` needs. A width passed explicitly is
    accepted when it is at least that, and **refused when it is not**, rather than left to
    wrap the register's values into a readout that can come back wrong.

    Args:
        incidence: The database, a real two-dimensional torch.Tensor of ``0`` and ``1``, one
            row per transaction and one column per item. The item count must be a power of
            two of at least two, and the unit is bounded at eight items and seven
            transactions.
        threshold: The least support a frequent item has, an integer in
            ``range(1, n_transactions + 1)``. The comparison is inclusive, so an item whose
            support is exactly the threshold is frequent.
        n_support_wires: The width of the support register, or ``None`` for the fewest wires
            that can hold the largest support the database can produce.

    Returns:
        The operator, carrying the database, the threshold and the register width.

    Raises:
        ValueError: If ``incidence`` is not a two-dimensional real tensor; if it holds no
            transaction, holds no item, or holds an entry that is neither ``0`` nor ``1``;
            if its item count is not a power of two of at least two, or exceeds the unit's
            bound; if ``threshold`` is not an integer in ``range(1, n_transactions + 1)``;
            or if ``n_support_wires`` cannot hold the largest support the database can
            produce or exceeds the unit's bound.
    """
    rows = _validated_matrix(incidence)
    n_transactions = len(rows)
    width = (
        _validated_support_width(n_transactions.bit_length(), n_transactions)
        if n_support_wires is None
        else n_support_wires
    )
    return FrequentItemsetOperator(
        incidence=rows, threshold=threshold, n_support_wires=width
    )


def run_frequent_itemset(
    incidence: torch.Tensor,
    *,
    threshold: int,
    n_counting_wires: int,
    shots: int = _DEFAULT_SHOTS,
    seed: int | None = None,
    n_support_wires: int | None = None,
) -> AmplitudeEstimationResult:
    """Estimate the fraction of the items whose support meets ``threshold``.

    The operator :func:`frequent_itemset_operator` builds is handed to
    :func:`~flagquantum.algorithms.amplitude_estimation.run_amplitude_estimation`, and that
    function's result is returned as it comes back: the estimate it carries is the frequent
    fraction, on the counting register's own amplitude grid. This unit adds no readout of
    its own to that result, and what the estimate's accuracy is and is not is stated by the
    module it comes from.

    Args:
        incidence: The database, validated as in :func:`frequent_itemset_operator`.
        threshold: The least support a frequent item has, validated as in
            :func:`frequent_itemset_operator`.
        n_counting_wires: The width of the counting register, at least one.
        shots: The number of samples to draw, at least one.
        seed: The sampler's seed, or ``None`` to draw from the ambient generator.
        n_support_wires: The width of the support register, or ``None`` for the fewest wires
            that can hold the largest support the database can produce.

    Returns:
        The amplitude estimation result, whose estimate is the fraction of the items whose
        support meets the threshold.

    Raises:
        ValueError: If any argument fails the validation
            :func:`frequent_itemset_operator` or
            :func:`~flagquantum.algorithms.amplitude_estimation.run_amplitude_estimation`
            applies to it.
    """
    operator = frequent_itemset_operator(
        incidence, threshold=threshold, n_support_wires=n_support_wires
    )
    return run_amplitude_estimation(
        operator, n_counting_wires=n_counting_wires, shots=shots, seed=seed
    )


def _validated_matrix(incidence: object) -> tuple[tuple[int, ...], ...]:
    """Validate an incidence matrix given as a tensor and return it as rows of ints.

    Args:
        incidence: The candidate database.

    Returns:
        One tuple of ``0`` and ``1`` per transaction, in row order.

    Raises:
        ValueError: If it is not a two-dimensional real tensor, holds no transaction, or
            holds an entry that is neither ``0`` nor ``1``.
    """
    if not isinstance(incidence, torch.Tensor):
        raise ValueError(
            f"the incidence matrix must be a torch.Tensor, got "
            f"{type(incidence).__name__}; the database is read entry by entry to build the "
            "gates, and a description of it is not read that way"
        )
    if incidence.dim() != 2:
        raise ValueError(
            "the incidence matrix must be a two-dimensional tensor, one row per "
            f"transaction and one column per item, got shape "
            f"{tuple(int(size) for size in incidence.shape)}"
        )
    if incidence.is_complex():
        raise ValueError(
            "the incidence matrix must be a real tensor, got dtype "
            f"{incidence.dtype}; an entry says whether one transaction holds one item, "
            "and a complex entry is not a membership"
        )
    values = incidence.detach().to(device="cpu", dtype=torch.float64)
    if not values.shape[0]:
        raise ValueError(
            "a database holds at least one transaction, got none; with no transaction "
            "every support is zero and there is no frequent fraction to estimate"
        )
    for row in values:
        for value in row:
            if float(value) not in (0.0, 1.0):
                raise ValueError(
                    f"every incidence entry must be 0 or 1, got {float(value)}; an entry "
                    "says whether one transaction holds one item, and a count in its place "
                    "would be read as a membership"
                )
    return tuple(tuple(int(value) for value in row) for row in values)


def _validated_incidence(rows: object) -> tuple[tuple[int, ...], ...]:
    """Validate an incidence matrix already carried as rows and return it.

    Args:
        rows: The candidate rows.

    Returns:
        The same rows.

    Raises:
        ValueError: If it is not a tuple, holds no transaction, holds a row that is not a
            tuple, holds rows whose widths differ or are empty, or holds an entry that is
            neither ``0`` nor ``1``.
    """
    if not isinstance(rows, tuple):
        raise ValueError(
            "the incidence matrix is carried as a tuple of rows, each a tuple of zeros "
            f"and ones, got {type(rows).__name__}; build the operator with "
            "frequent_itemset_operator, which reads a tensor"
        )
    if not rows:
        raise ValueError(
            "a database holds at least one transaction, got none; with no transaction "
            "every support is zero and there is no frequent fraction to estimate"
        )
    for row in rows:
        if not isinstance(row, tuple):
            raise ValueError(
                "every row of the incidence matrix is carried as a tuple of zeros and "
                f"ones, got {type(row).__name__} in its place; build the operator with "
                "frequent_itemset_operator, which reads a tensor"
            )
    widths = {len(row) for row in rows}
    if len(widths) != 1 or not next(iter(widths)):
        raise ValueError(
            "every transaction carries one entry per item, so the rows must share one "
            f"width and hold at least one entry each, got widths {sorted(widths)}"
        )
    for row in rows:
        for entry in row:
            if entry not in (0, 1):
                raise ValueError(
                    f"every incidence entry must be 0 or 1, got {entry!r}; an entry says "
                    "whether one transaction holds one item, and a count in its place "
                    "would be read as a membership"
                )
    return rows


def _validated_item_count(n_items: int) -> None:
    """Check the number of items the database has a column for.

    Args:
        n_items: The candidate item count.

    Raises:
        ValueError: If it is not a power of two of at least two, or if it needs more item
            wires than this unit is written for.
    """
    if n_items < 2 or n_items & (n_items - 1):
        raise ValueError(
            f"the item count must be a power of two of at least two, got {n_items}; the "
            "item register is addressed by one wire per item-index bit and its preparation "
            "is emitted under control, which a uniform state over another count has no "
            "form for in this package"
        )
    if n_items.bit_length() - 1 > _MAX_ITEM_WIRES:
        raise ValueError(
            f"this unit is bounded at {_MAX_ITEM_WIRES} item wires, so at most "
            f"{2**_MAX_ITEM_WIRES} items, got {n_items}: the marking operator enumerates "
            "the support values at or above the threshold and the transaction loop emits "
            "one increment per transaction-item pair, so this is the demonstration scale "
            "the unit is written at"
        )


def _validated_threshold(threshold: object, n_transactions: int) -> int:
    """Validate the least support a frequent item has.

    Args:
        threshold: The candidate threshold.
        n_transactions: The number of transactions it is a support of.

    Returns:
        The threshold.

    Raises:
        ValueError: If it is not an integer in ``range(1, n_transactions + 1)``.
    """
    if isinstance(threshold, bool) or not isinstance(threshold, int):
        raise ValueError(
            f"the threshold must be an integer, got {threshold!r}; a support is a count of "
            "transactions, and a float or a flag is not one"
        )
    if not 1 <= threshold <= n_transactions:
        raise ValueError(
            f"the threshold must be a support in range(1, {n_transactions + 1}), got "
            f"{threshold}; no support exceeds the {n_transactions} transactions the "
            "database has, so a threshold above that would mark no item, and one below "
            "one would mark every item, and neither needs a circuit to answer"
        )
    return threshold


def _validated_support_width(n_support_wires: object, n_transactions: int) -> int:
    """Validate the width of the support register.

    Args:
        n_support_wires: The candidate width.
        n_transactions: The number of transactions the register counts over.

    Returns:
        The width.

    Raises:
        ValueError: If it is not an integer, if it cannot hold the largest support the
            database can produce, or if it exceeds this unit's bound.
    """
    if isinstance(n_support_wires, bool) or not isinstance(n_support_wires, int):
        raise ValueError(
            f"the support register's width must be an integer, got "
            f"{n_support_wires!r}; a width is a count of wires"
        )
    largest = n_transactions
    fewest = n_transactions.bit_length()
    if n_support_wires < fewest:
        raise ValueError(
            f"the support register needs at least {fewest} wires to hold the largest "
            f"support {largest} that {n_transactions} transactions can produce, got "
            f"{n_support_wires}; the increment is a permutation of the register's values, "
            "so a narrower register wraps a support into another value and the readout can "
            "come back wrong with nothing raised"
        )
    if n_support_wires > _MAX_SUPPORT_WIRES:
        raise ValueError(
            f"this unit is bounded at {_MAX_SUPPORT_WIRES} support wires, which is at most "
            f"{2**_MAX_SUPPORT_WIRES - 1} transactions, got {n_support_wires} support "
            "wire(s): the evaluation register carries the item register, the support "
            "register, the multi-controlled X ladder ancillas and the marking flag, and "
            "this is the demonstration scale the unit is written at"
        )
    return n_support_wires


def _ladder_ancillas(n_item_wires: int, n_support_wires: int) -> int:
    """Return the ancillas the widest multi-controlled X of this register layout needs.

    The widest is the one the controlled increment and the zero reflection emit, whose
    control list is the control wire, the item register and all but one wire of the support
    register. A multi-controlled X above two controls borrows one ancilla per control beyond
    the two a native gate takes, and the same pool serves every ladder because each restores
    what it borrows.

    Args:
        n_item_wires: The width of the item register.
        n_support_wires: The width of the support register.

    Returns:
        The number of ancilla wires the layout reserves.
    """
    return max(n_item_wires + n_support_wires - 2, 0)


def _append_pattern_bracket(circuit: Circuit, wires: Sequence[int], value: int) -> None:
    """Flip the wires whose bit of ``value`` is zero, so the ones are the pattern.

    The same call takes the bracket back off, because ``x`` is its own inverse. Both
    registers it is used on are read most significant bit first, so ``wires[0]`` is the
    most significant bit of ``value``.

    Args:
        circuit: The circuit to extend.
        wires: The register's wires, most significant first.
        value: The register value whose zero bits are to be flipped.
    """
    for index, wire in enumerate(wires):
        if not (value >> (len(wires) - 1 - index)) & 1:
            circuit.gate("x", wire)


def _append_support_step(
    circuit: Circuit,
    control: int | None,
    item: Sequence[int],
    support: Sequence[int],
    pool: Sequence[int],
    forward: bool,
) -> None:
    """Append the ladder that adds one to the support register, or takes one away.

    Besides the item register's wires, which control every gate in it, the ladder is a
    ripple carry: one multi-controlled X per support wire, each controlled by that wire's
    less significant neighbours, and emitted from the most significant wire down so that
    every carry reads the value the register held on entry. The reverse order is the same
    ladder run backwards, which is the decrement.

    Args:
        circuit: The circuit to extend.
        control: The control wire, or ``None`` for the uncontrolled ladder.
        item: The item register's wires, part of every control list.
        support: The support register's wires, most significant first.
        pool: The ancilla wires the ladders may borrow.
        forward: Whether to add one or to take one away.
    """
    order = range(len(support)) if forward else reversed(range(len(support)))
    for index in order:
        controls = (
            ([] if control is None else [control])
            + list(item)
            + list(support[index + 1 :])
        )
        append_multi_controlled_x(
            circuit,
            controls,
            support[index],
            ancillas=pool[: max(len(controls) - 2, 0)],
        )


def _append_support_match(
    circuit: Circuit,
    support: Sequence[int],
    pool: Sequence[int],
    flag: int,
    threshold: int,
) -> None:
    """XOR onto ``flag`` the truth of ``support >= threshold``.

    One multi-controlled X per support value at or above the threshold, each bracketed so
    that its controls are that value's ones. The values are distinct, so at most one of them
    matches and the flag ends holding the truth of the comparison rather than a parity.

    Args:
        circuit: The circuit to extend.
        support: The support register's wires, most significant first.
        pool: The ancilla wires the ladders may borrow.
        flag: The wire the comparison is XORed onto, in ``|0>`` on entry.
        threshold: The least support the comparison accepts.
    """
    for value in range(threshold, 2 ** len(support)):
        _append_pattern_bracket(circuit, support, value)
        append_multi_controlled_x(
            circuit,
            list(support),
            flag,
            ancillas=pool[: max(len(support) - 2, 0)],
        )
        _append_pattern_bracket(circuit, support, value)
