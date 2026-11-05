"""Both pinned greater-than paths with complete coherent XOR semantics."""

from collections import Counter

import pytest

from flagquantum.compiler.registers.less_equal import less_equal_requirements
from flagquantum.compiler.registers.ordering import greater_ir
from tests.unit.greater_checks import check_greater_amplitudes
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "m,n",
    [
        (1, 1),
        (2, 2),
        (3, 3),
        (5, 5),
        (8, 8),
        (1, 2),
        (2, 1),
        (1, 4),
        (4, 1),
        (3, 8),
        (8, 3),
    ],
)
def test_tree_greater_original_wrapper_and_model(m: int, n: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.arithmetic.comparison import GreaterThan
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    q, c = less_equal_requirements(m, n)
    target = m + n
    a = tuple(range(m))
    b = tuple(range(m, m + n))
    w = tuple(range(target + 1, target + q + 1))
    bits = tuple(range(40, 40 + c))
    actual = greater_ir(a, b, target, w, bits)
    source = GreaterThan(m, n)
    measured = {
        g.wires[0]: g.metadata["classical_bit"] for g in actual if g.name == "measure"
    }
    expected = reference_arithmetic_gates(
        source, {"a": a, "b": b, "target": (target,)}, w, measured, cirq_style=True
    )
    assert_source(actual, expected)
    counts = Counter(g.name for g in actual)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 4 * c
    assert counts["measure"] == model.measurement == c


@pytest.mark.integration
@pytest.mark.parametrize("m,n", [(1, 1), (2, 2), (3, 3), (1, 3), (3, 1)])
def test_tree_all_complex_amplitudes_and_clean_work(m: int, n: int) -> None:
    q, c = less_equal_requirements(m, n)
    target = m + n
    gates = greater_ir(
        tuple(range(m)),
        tuple(range(m, m + n)),
        target,
        tuple(range(target + 1, target + q + 1)),
        tuple(range(40, 40 + c)),
    )
    check_greater_amplitudes(m, n, q, c, gates)
