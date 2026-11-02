"""Independent oracle specifications, finite domains and admission boundaries."""

from itertools import product

import pytest

from benchmarks.synthesis.expressions import EDGES, EXPRESSIONS, Case, corpus
from benchmarks.synthesis.oracle import evaluate, expected_amplitudes, output_width
from benchmarks.synthesis.verification import classical_apply
from flagquantum.core.ir import CircuitIR, Instruction

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("expression", EXPRESSIONS + EDGES, ids=lambda row: row.name)
def test_complete_small_domain_matches_source_and_corner_bound(expression) -> None:
    case = Case(expression, (2,) * len(expression.arguments))
    values = list(product(range(4), repeat=len(expression.arguments)))
    actual = [evaluate(expression.name, value) for value in values]
    assert actual == [expression.function(*value) for value in values]
    assert output_width(case) == max(1, max(actual).bit_length())
    assert min(actual) >= 0


def test_corpus_is_unique_and_has_all_fixed_domain_classes() -> None:
    cases = corpus()
    assert len(cases) == len({row.identifier for row in cases}) == 90
    for expression in EXPRESSIONS:
        for width in (1, 2, 3, 4, 8, 16):
            assert Case(expression, (width,) * len(expression.arguments)) in cases
    assert any(row.widths == (1, 3, 2) for row in cases)
    assert any(row.widths == (2, 4) for row in cases)


def test_oracle_refuses_unknown_specification_truncation_and_exponential_size() -> None:
    with pytest.raises(ValueError, match="unknown"):
        evaluate("not_an_expression", (1,))
    case = Case(EXPRESSIONS[4], (2, 2))
    with pytest.raises(ValueError, match="truncates"):
        expected_amplitudes(case, {"x": (0, 1), "y": (2, 3), "result": (4,)}, 5)
    with pytest.raises(ValueError, match="six input bits"):
        expected_amplitudes(Case(EXPRESSIONS[4], (4, 4)), {}, 18)


def test_exact_permutation_interpreter_and_unknown_gate_refusal() -> None:
    ir = CircuitIR(
        3,
        (
            Instruction("x", (0,)),
            Instruction("cx", (0, 1)),
            Instruction("ccx", (0, 1, 2)),
        ),
    )
    assert classical_apply(ir, 0) == 7
    assert classical_apply(ir, 1) == 0
    with pytest.raises(ValueError, match="X/CX/CCX"):
        classical_apply(CircuitIR(1, (Instruction("h", (0,)),)), 0)
