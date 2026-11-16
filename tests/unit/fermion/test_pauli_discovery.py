"""Commuting discovery against exhaustive, independent small Pauli matrices."""

import itertools
import math
import random

import pytest
import torch

from flagquantum.algorithms.fermion.stabilizer import discover_pauli_symmetries
from flagquantum.core.algebra.pauli import PauliOperator, pauli

pytestmark = pytest.mark.unit


def _word(axes: str, coefficient: complex = 1) -> PauliOperator:
    return pauli(
        coefficient, tuple((q, axis) for q, axis in enumerate(axes) if axis != "i")
    )


def _matrix(axes: str) -> torch.Tensor:
    matrices = {
        "i": [[1, 0], [0, 1]],
        "x": [[0, 1], [1, 0]],
        "y": [[0, -1j], [1j, 0]],
        "z": [[1, 0], [0, -1]],
    }
    result = torch.ones((1, 1), dtype=torch.complex128)
    for axis in axes:
        result = torch.kron(
            result, torch.tensor(matrices[axis], dtype=torch.complex128)
        )
    return result


def _operator_matrix(operator: PauliOperator, width: int) -> torch.Tensor:
    result = torch.zeros((2**width, 2**width), dtype=torch.complex128)
    for term in operator.terms:
        factors = dict(term.factors)
        result += term.coefficient * _matrix(
            "".join(factors.get(q, "i") for q in range(width))
        )
    return result


def _same_label(left: torch.Tensor, right: torch.Tensor) -> bool:
    # Ignore scalar phases using the Hilbert-Schmidt overlap, independently of
    # the production binary representation and canonical Pauli multiplication.
    return bool(
        torch.isclose(
            torch.trace(left.mH @ right).abs(),
            torch.tensor(float(left.shape[0]), dtype=torch.float64),
        )
    )


def _span(generators: list[torch.Tensor], width: int) -> list[torch.Tensor]:
    result = [torch.eye(2**width, dtype=torch.complex128)]
    for generator in generators:
        result += [word @ generator for word in result]
    return result


def _assert_complete_commuting_choice(
    operator: PauliOperator,
    width: int,
    candidates: tuple[PauliOperator, ...] | None = None,
) -> None:
    found = discover_pauli_symmetries(operator, n_qubits=width, candidates=candidates)
    matrices = [_operator_matrix(word, width) for word in found]
    span = _span(matrices, width)
    assert all(
        len(word.terms) == 1
        and word.terms[0].coefficient == 1
        and word.terms[0].factors
        for word in found
    )
    assert all(
        not _same_label(left, right)
        for i, left in enumerate(span)
        for right in span[:i]
    )
    full = _operator_matrix(operator, width)
    assert all(torch.equal(word @ full, full @ word) for word in matrices)
    assert all(
        torch.equal(left @ right, right @ left)
        for left in matrices
        for right in matrices
    )
    allowed = (
        None
        if candidates is None
        else _span([_operator_matrix(word, width) for word in candidates], width)
    )
    centralizer = []
    for labels in itertools.product("ixyz", repeat=width):
        word = _matrix("".join(labels))
        if allowed is not None and not any(
            _same_label(word, other) for other in allowed
        ):
            continue
        if torch.equal(word @ full, full @ word):
            centralizer.append(word)
            if all(torch.equal(word @ other, other @ word) for other in matrices):
                assert any(_same_label(word, other) for other in span)
    radical = [
        word
        for word in centralizer
        if all(torch.equal(word @ other, other @ word) for other in centralizer)
    ]
    # Count dimensions from exhaustive matrix sets rather than reimplementing
    # binary elimination. This proves maximum rank as well as maximality.
    assert 2 * len(found) == round(math.log2(len(centralizer))) + round(
        math.log2(len(radical))
    )


@pytest.mark.parametrize(
    "operator,width",
    [
        (PauliOperator(), 0),
        (PauliOperator(), 1),
        (pauli(2j), 2),
        (_word("yi"), 2),
        (_word("xx") + _word("yy") + _word("zz"), 2),
        (_word("xi") + 2j * _word("zi"), 2),
        (_word("xi") + _word("zi") + _word("ix") + _word("iz"), 2),
        (_word("yzi") + 0.5 * _word("ixx") + _word("zzz"), 3),
    ],
)
def test_unrestricted_discovery_is_a_maximum_commuting_choice(
    operator: PauliOperator, width: int
) -> None:
    _assert_complete_commuting_choice(operator, width)


@pytest.mark.parametrize(
    "operator,candidates",
    [
        (_word("zz"), (_word("xi"), _word("ix"))),
        (_word("xx"), (_word("zi"), _word("iz"))),
        (_word("zz"), (_word("xi"), _word("yi"), _word("ix"))),
        (PauliOperator(), (_word("xi"), _word("zi"), _word("iz", -1))),
        (_word("xy") + _word("yz"), (_word("xx"), _word("yi"), _word("iz"))),
        (_word("xy"), ()),
    ],
)
def test_restricted_spans_include_commuting_products(
    operator: PauliOperator, candidates: tuple[PauliOperator, ...]
) -> None:
    _assert_complete_commuting_choice(operator, 2, candidates)


def test_seeded_three_qubit_operators_against_exhaustive_centralizers() -> None:
    rng = random.Random(20261112)
    labels = ["".join(axes) for axes in itertools.product("ixyz", repeat=3)][1:]
    for count in (1, 2, 3, 4, 5, 6):
        operator = sum(
            (
                (i + 1) / 8 * _word(axes)
                for i, axes in enumerate(rng.sample(labels, count))
            ),
            PauliOperator(),
        )
        _assert_complete_commuting_choice(operator, 3)


def test_noncommuting_candidates_can_produce_a_valid_product() -> None:
    # Neither supplied factor commutes with X+Y. Their binary product is the
    # positive Z representative, which also fails: there is no Pauli symmetry.
    assert (
        discover_pauli_symmetries(
            _word("x") + _word("y"), n_qubits=1, candidates=[_word("x"), _word("y")]
        )
        == ()
    )
    assert discover_pauli_symmetries(
        _word("zz"), n_qubits=2, candidates=[_word("xi"), _word("ix")]
    ) == (_word("xx"),)
    assert discover_pauli_symmetries(
        _word("z"), n_qubits=1, candidates=[_word("x"), _word("y")]
    ) == (_word("z"),)


def test_equivalent_candidate_bases_signs_and_term_orders_are_canonical() -> None:
    operator = _word("yiz") + 2 * _word("yix") + 0.3 * _word("ixi")
    bases = [
        [_word("yiy"), _word("ixi"), _word("iiz")],
        [_word("yiy", -1), _word("yxy"), _word("iiz", -1)],
    ]
    expected = discover_pauli_symmetries(operator, n_qubits=3, candidates=bases[0])
    for basis in bases:
        for order in itertools.permutations(basis):
            assert (
                discover_pauli_symmetries(operator, n_qubits=3, candidates=order)
                == expected
            )
            reversed_terms = PauliOperator(tuple(reversed(operator.terms)))
            assert (
                discover_pauli_symmetries(reversed_terms, n_qubits=3, candidates=order)
                == expected
            )
    assert discover_pauli_symmetries(
        _word("y"), n_qubits=1, candidates=[_word("y", -1)]
    ) == (_word("y"),)


def test_exact_cancellation_and_tiny_terms_are_not_thresholded() -> None:
    x = _word("x")
    assert discover_pauli_symmetries(x, n_qubits=1) == (x,)
    assert discover_pauli_symmetries(x + 1e-200 * _word("z"), n_qubits=1) == ()
    assert discover_pauli_symmetries(x + _word("z") - _word("z"), n_qubits=1) == (x,)


def test_wide_symbolic_search_preserves_idle_qubits_without_dense_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("symbolic discovery must not allocate dense tensors")

    for name in ("zeros", "ones", "eye", "empty", "tensor"):
        monkeypatch.setattr(torch, name, refuse)
    operator = pauli(factors=((127, "y"),))
    found = discover_pauli_symmetries(operator, n_qubits=128)
    assert len(found) == 128
    assert found[-1] == operator
    assert discover_pauli_symmetries(operator, n_qubits=128, candidates=[]) == ()


@pytest.mark.parametrize("width", [True, 1.5, "2"])
def test_invalid_width_types_fail_before_search(width: int) -> None:
    with pytest.raises(TypeError):
        discover_pauli_symmetries(PauliOperator(), n_qubits=width)


@pytest.mark.parametrize(
    "candidates,error",
    [
        (["x"], TypeError),
        ([PauliOperator()], ValueError),
        ([pauli()], ValueError),
        ([_word("x", 0.5)], ValueError),
        ([_word("x", 1j)], ValueError),
        ([_word("x") + _word("z")], ValueError),
        ([_word("ix")], ValueError),
        ([_word("x"), _word("x", -1)], ValueError),
        ([_word("x"), _word("y"), _word("z")], ValueError),
    ],
)
def test_invalid_candidates_fail_closed(
    candidates: list[PauliOperator], error: type[Exception]
) -> None:
    with pytest.raises(error):
        discover_pauli_symmetries(_word("x"), n_qubits=1, candidates=candidates)


@pytest.mark.parametrize("operator", ["x", object()])
def test_invalid_operator_type_fails_closed(operator: PauliOperator) -> None:
    with pytest.raises(TypeError, match="native"):
        discover_pauli_symmetries(operator, n_qubits=1)


def test_invalid_register_fails_closed() -> None:
    with pytest.raises(ValueError, match="at least"):
        discover_pauli_symmetries(PauliOperator(), n_qubits=-1)
    with pytest.raises(ValueError, match="outside"):
        discover_pauli_symmetries(_word("x"), n_qubits=0)
