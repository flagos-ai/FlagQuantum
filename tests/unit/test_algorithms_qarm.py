"""What the frequent-itemset unit builds, checked against an enumerated reference.

Every database here is small enough for its supports and its frequent fraction to be
computed in plain Python from the incidence matrix: one column sum per item for the
supports, and one count of the columns at or above the threshold for the fraction. Those
values are the reference the circuit is read against, in two places -- the preparation's
joint distribution of item and support over the circuit's own state, and the
amplitude-estimation readout. The operator's own ``support`` property is that same column
sum written a second time, so the test on it checks the length and the item order it
reports rather than the arithmetic it performs.

The marking operator is checked against the same reference a third time, on the sign it
gives each item's amplitude: an item is marked exactly when its support meets the
threshold, which is where the inclusive comparison is pinned rather than described.

The measured instance of the brief -- two transactions over two items, threshold two --
is carried below with the values this repository's runs produce for it, and every expected
value in this file was measured by running the construction it checks. One measurement --
the one-wire support register -- was taken with the builder's width refusal lifted, which
the test that carries it says.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from flagquantum.algorithms.amplitude_estimation import (
    amplitude_resolution,
    run_amplitude_estimation,
)
from flagquantum.algorithms.primitives.types import AmplitudeOperator
from flagquantum.algorithms.qarm import (
    FrequentItemsetOperator,
    frequent_itemset_operator,
    run_frequent_itemset,
)
from flagquantum.circuit import Circuit

pytestmark = pytest.mark.unit

# The instance of the brief: two transactions over two items. Item 0 is held by both
# transactions, so its support is 2 and it is frequent at a threshold of 2; item 1 is held
# by one, so its support is 1 and it is not. The frequent fraction is then 1 / 2.
_MEASURED = [[1, 0], [1, 1]]
_MEASURED_THRESHOLD = 2

# Two transactions over four items, the first two held by both and the last two by neither.
# At a threshold of 2 the fraction is 2 / 4.
_HALF = [[1, 1, 0, 0], [1, 1, 0, 0]]

# Four transactions over four items whose supports fall from three to one. Its thresholds
# reach fractions the two-transaction databases cannot: 3 / 4 and 1 / 4.
_MIXED = [[1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 1, 1]]

# The unit's bound on both registers at once: eight items held by between one and seven of
# seven transactions, so the item register takes three wires, the support register takes
# three, and the multi-controlled X ladders borrow from a pool instead of folding onto the
# controls alone.
_LADDER = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0],
]

# One database of every shape the unit accepts, for the comparisons that do not need a
# particular support pattern.
_DATABASES = [
    _MEASURED,
    [[1, 1, 0, 1], [1, 1, 1, 0], [0, 1, 1, 1]],
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    _HALF,
    _MIXED,
    _LADDER,
]


def _supports(rows: list[list[int]]) -> tuple[int, ...]:
    """Return the support of each item, as the column sums of the incidence matrix."""
    return tuple(sum(row[item] for row in rows) for item in range(len(rows[0])))


def _frequent_fraction(rows: list[list[int]], threshold: int) -> float:
    """Return the share of the items whose support meets ``threshold``."""
    counts = _supports(rows)
    return sum(1 for count in counts if count >= threshold) / len(counts)


def _operator(rows: list[list[int]], threshold: int) -> FrequentItemsetOperator:
    """Build the operator of one database at one threshold."""
    return frequent_itemset_operator(
        torch.tensor(rows, dtype=torch.int64), threshold=threshold
    )


def _prepared_states(
    operator: FrequentItemsetOperator,
) -> dict[tuple[int, int, str], float]:
    """Return the item, support and auxiliary value of each state ``A|0>`` reaches.

    The keys are the three parts of the evaluation register -- the item index, the support
    register's value and the ancillas-and-flag suffix -- and the values are the state's
    probabilities, so a state the preparation does not reach is absent.
    """
    circuit = Circuit(operator.n_wires)
    operator.apply_plain(circuit, list(range(operator.n_wires)))
    probabilities = circuit.probabilities().reshape(-1)
    n_item = operator.n_item_wires
    n_support = operator.n_support_wires
    return {
        (
            int(bits[:n_item], 2),
            int(bits[n_item : n_item + n_support], 2),
            bits[n_item + n_support :],
        ): float(probabilities[index])
        for index, bits in _reached(probabilities, operator.n_wires)
    }


def _marked_signs(operator: FrequentItemsetOperator) -> dict[int, float]:
    """Return the sign each item's amplitude carries after the marking operator.

    The marking operator is applied with its control wire set, so the phase flip is the
    marking operator's own and not the control's; a marked item's amplitude is negated and
    an unmarked one's is not.
    """
    wires = list(range(operator.n_wires))
    control = operator.n_wires
    circuit = Circuit(operator.n_wires + 1)
    operator.apply_plain(circuit, wires)
    circuit.gate("x", control)
    operator.apply_mark(circuit, control, wires)
    state = circuit.state().reshape(-1)
    return {
        int(bits[: operator.n_item_wires], 2): float(state[index].real)
        for index, bits in _reached(state.abs(), operator.n_wires + 1)
    }


def _reached(weights: torch.Tensor, n_wires: int) -> list[tuple[int, str]]:
    """Return the index and big-endian bit string of every state carrying weight."""
    return [
        (int(index), format(int(index), f"0{n_wires}b"))
        for index in torch.nonzero(weights > 1e-6).reshape(-1).tolist()
    ]


def test_the_operator_satisfies_the_protocol() -> None:
    """The built operator is a valid AmplitudeOperator."""
    assert isinstance(_operator(_MEASURED, 2), AmplitudeOperator)


def test_the_operator_is_a_frozen_keyword_only_dataclass() -> None:
    """The three fields are keyword-only and frozen, so a call site names each one.

    This is what the class docstring's ``kw_only`` paragraph is about: with a positional
    spelling available, a matrix and two bare numbers of unrelated kind would read alike.
    """
    operator = _operator(_MEASURED, 2)
    with pytest.raises(TypeError):
        FrequentItemsetOperator(_MEASURED, 2, 2)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        operator.threshold = 1  # type: ignore[misc]


@pytest.mark.parametrize("rows", _DATABASES)
def test_the_supports_are_the_column_sums(rows: list[list[int]]) -> None:
    """The support column has one entry per item, in item order.

    What this compares is the operator's own column sum against the same column sum
    written in this file, so the arithmetic is the same expression on both sides and the
    test can fail only on the length or the item order. The circuit's supports are read
    against the independent reference in
    :func:`test_the_preparation_pairs_every_item_with_its_own_support`.
    """
    assert _operator(rows, 1).support == _supports(rows)


@pytest.mark.parametrize("rows", _DATABASES)
def test_the_preparation_pairs_every_item_with_its_own_support(
    rows: list[list[int]],
) -> None:
    """``A|0>`` reaches one state per item, carrying that item's support and a clean suffix.

    This is the transaction loop checked on the circuit's own state rather than on the
    operator's arithmetic: the support register's value for each item is read off the
    amplitude the preparation gives that item, so a loop that missed a transaction, ran a
    branch that never fired, or left an ancilla set would change the key set here.
    """
    operator = _operator(rows, 1)
    clean_suffix = "0" * (
        operator.n_wires - operator.n_item_wires - operator.n_support_wires
    )
    expected = {
        (item, support, clean_suffix): 1.0 / operator.n_items
        for item, support in enumerate(_supports(rows))
    }

    prepared = _prepared_states(operator)

    assert set(prepared) == set(expected)
    for key, probability in expected.items():
        assert prepared[key] == pytest.approx(probability, abs=1e-6)


@pytest.mark.parametrize(
    ("rows", "threshold"),
    [
        (_MEASURED, 1),
        (_MEASURED, 2),
        ([[1, 1, 0, 1], [1, 1, 1, 0], [0, 1, 1, 1]], 2),
        ([[1, 1, 0, 1], [1, 1, 1, 0], [0, 1, 1, 1]], 3),
        (_LADDER, 4),
        (_LADDER, 6),
    ],
)
def test_the_marking_operator_negates_exactly_the_frequent_items(
    rows: list[list[int]], threshold: int
) -> None:
    """An item's amplitude is negated exactly when its support meets the threshold."""
    operator = _operator(rows, threshold)
    signs = _marked_signs(operator)

    assert set(signs) == set(range(operator.n_items))
    for item, support in enumerate(_supports(rows)):
        if support >= threshold:
            assert signs[item] < 0.0, (item, support, threshold)
        else:
            assert signs[item] > 0.0, (item, support, threshold)


def test_the_threshold_is_inclusive_at_the_boundary() -> None:
    """An item whose support equals the threshold is frequent, and the readout counts it.

    The boundary is reachable and it decides the answer: item 0 has a support of exactly 2,
    so a threshold of 2 counts it and the exact fraction is one half, where a strict
    comparison would mark nothing and the readout would be 0.0.
    """
    supports = _supports(_MEASURED)
    assert supports == (2, 1)

    assert _marked_signs(_operator(_MEASURED, 2))[0] < 0.0
    assert _frequent_fraction(_MEASURED, 2) == 0.5
    assert run_frequent_itemset(
        torch.tensor(_MEASURED), threshold=2, n_counting_wires=4, shots=8000, seed=17
    ).estimate == pytest.approx(0.5, abs=1e-9)


@pytest.mark.parametrize(
    ("rows", "threshold"),
    [
        (_MEASURED, 1),
        (_MEASURED, 2),
        (_HALF, 2),
        ([[1, 1, 0, 1], [1, 1, 1, 0], [0, 1, 1, 1]], 3),
        (_MIXED, 2),
        (_MIXED, 3),
    ],
)
def test_the_readout_is_the_frequent_fraction(
    rows: list[list[int]], threshold: int
) -> None:
    """The estimate is the enumerated frequent fraction, to within one grid step."""
    result = run_frequent_itemset(
        torch.tensor(rows, dtype=torch.int64),
        threshold=threshold,
        n_counting_wires=4,
        shots=8000,
        seed=17,
    )

    assert result.within(_frequent_fraction(rows, threshold))


def test_the_measured_instance_reads_the_measured_fraction() -> None:
    """The brief's instance, at the counting width, sample size and seed it was measured at."""
    operator = _operator(_MEASURED, _MEASURED_THRESHOLD)
    result = run_frequent_itemset(
        torch.tensor(_MEASURED),
        threshold=_MEASURED_THRESHOLD,
        n_counting_wires=4,
        shots=8000,
        seed=17,
    )

    assert result.estimate == pytest.approx(0.5, abs=1e-9)
    assert result.resolution == pytest.approx(amplitude_resolution(4), abs=1e-12)
    assert result.resolution == pytest.approx(0.097545, abs=1e-6)
    assert result.n_counting_wires == 4
    assert result.n_evaluation_wires == operator.n_wires


def test_the_run_is_the_amplitude_estimation_result_of_the_operator() -> None:
    """The runner adds no readout of its own: it is Phase 1's function on the operator."""
    operator = _operator(_MEASURED, _MEASURED_THRESHOLD)
    assert run_frequent_itemset(
        torch.tensor(_MEASURED),
        threshold=_MEASURED_THRESHOLD,
        n_counting_wires=4,
        shots=8000,
        seed=17,
    ) == run_amplitude_estimation(operator, n_counting_wires=4, shots=8000, seed=17)


def test_the_default_width_is_the_fewest_that_holds_every_support() -> None:
    """The default width is the bits the transaction count needs, and it is not one more."""
    for rows in (_MEASURED, [[1, 1, 0, 1]] * 3, [[1, 1, 0, 1]] * 7):
        operator = _operator(rows, 1)
        assert operator.n_support_wires == operator.n_transactions.bit_length()


def test_a_support_register_that_cannot_hold_the_largest_support_is_refused() -> None:
    """A register too narrow to hold every support is refused, not left to wrap.

    Two transactions can produce a support of 2, and one wire holds 0 and 1 only. Measured
    on that instance with a register of one wire, and with the width refusal lifted for the
    measurement: the increment wraps the support of 2 back to 0, the mark then sees no
    support at the threshold, the estimate reads 0.0 against an exact fraction of 0.5, and
    nothing is raised -- which is why the width is checked here instead.
    """
    with pytest.raises(ValueError, match="at least 2 wires"):
        frequent_itemset_operator(
            torch.tensor(_MEASURED), threshold=2, n_support_wires=1
        )

    assert _operator(_MEASURED, 2).n_support_wires == 2


def test_the_evaluation_register_carries_no_transaction_index() -> None:
    """The register's width is set by the item count and the support width, never by ``N``.

    A transaction index held in a register would widen the evaluation register as soon as
    the transaction count crossed a power of two. Two databases of two items with a support
    register of three wires, one of two transactions and one of seven, have the same width,
    and that width is the item register, the support register, the ladder ancillas the
    widest multi-controlled X borrows and the marking flag.
    """
    two = frequent_itemset_operator(
        torch.tensor([[1, 0], [1, 1]]), threshold=2, n_support_wires=3
    )
    seven = frequent_itemset_operator(
        torch.tensor([[1, 0]] * 7), threshold=2, n_support_wires=3
    )

    assert two.n_wires == seven.n_wires == 7
    assert two.support == (2, 1)
    assert seven.support == (7, 0)


@pytest.mark.parametrize("dtype", [torch.int64, torch.bool, torch.float64])
def test_a_database_may_be_handed_over_in_any_real_dtype(dtype: torch.dtype) -> None:
    """A database may be handed over as an integer tensor, a boolean one or a real one."""
    incidence = torch.tensor(_MEASURED).to(dtype)
    assert frequent_itemset_operator(incidence, threshold=1).support == _supports(
        _MEASURED
    )


@pytest.mark.parametrize(
    "incidence",
    [
        "not a tensor",
        torch.tensor([1, 0, 1, 1]),
        torch.tensor([[[1, 0], [1, 1]]]),
        torch.empty((0, 2), dtype=torch.int64),
        torch.tensor([[1, 0], [1, 2]]),
        torch.tensor([[1.0, 0.0], [0.5, 1.0]]),
        torch.tensor([[float("inf"), 0.0], [1.0, 1.0]]),
        torch.tensor([[float("nan"), 0.0], [1.0, 1.0]]),
        torch.tensor([[1 + 0j, 0 + 0j]]),
        torch.ones((2, 3), dtype=torch.int64),
        torch.ones((2, 1), dtype=torch.int64),
        torch.ones((2, 16), dtype=torch.int64),
    ],
)
def test_an_incidence_matrix_the_unit_cannot_read_is_refused(incidence: object) -> None:
    """Every form of database this unit does not read raises rather than being coerced."""
    with pytest.raises(ValueError):
        frequent_itemset_operator(incidence, threshold=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("threshold", [0, 3, -1, 1.5, True, "2"])
def test_a_threshold_outside_the_supports_the_database_can_produce_is_refused(
    threshold: object,
) -> None:
    """A threshold no support can meet, or every support can, is not a question to run."""
    with pytest.raises(ValueError):
        frequent_itemset_operator(
            torch.tensor(_MEASURED), threshold=threshold  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("width", [0, 1, -2, 2.5, True, 4, 5])
def test_a_support_register_width_the_unit_cannot_use_is_refused(width: object) -> None:
    """A width that cannot hold every support, or that the unit is not written for, raises."""
    with pytest.raises(ValueError):
        frequent_itemset_operator(
            torch.tensor(_MEASURED),
            threshold=2,
            n_support_wires=width,  # type: ignore[arg-type]
        )


def test_the_row_diagnostic_names_the_row() -> None:
    """A row that is not a tuple is refused by that row's type, not the container's.

    The container here is a tuple either way, so a message that names the container says
    nothing about what is wrong with it.
    """
    with pytest.raises(ValueError, match="got int in its place"):
        FrequentItemsetOperator(
            incidence=((1, 0), 1),  # type: ignore[arg-type]
            threshold=1,
            n_support_wires=3,
        )


@pytest.mark.parametrize(
    "rows",
    [
        (),
        ((1, 0), (1,)),
        ((1, 0), (1, 2)),
        [(1, 0), (1, 1)],
        ((1, 0), 1),
        ((1, 0, 0), (1, 1, 1)),
    ],
)
def test_a_directly_constructed_operator_with_unreadable_fields_is_refused(
    rows: object,
) -> None:
    """The class validates its own fields, so the builder is not the only guard."""
    with pytest.raises(ValueError):
        FrequentItemsetOperator(
            incidence=rows,  # type: ignore[arg-type]
            threshold=1,
            n_support_wires=3,
        )
