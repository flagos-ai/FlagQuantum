"""Both pinned greater-than paths with complete coherent XOR semantics."""

from collections import Counter

import pytest

from flagquantum.compiler.registers.linear_comparison import (
    linear_greater_ir,
    linear_greater_requirements,
)
from tests.unit.greater_checks import check_greater_amplitudes
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("width", [1, 2, 3, 4, 5, 8])
def test_linear_greater_source_stream_and_model(width: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.arithmetic.comparison import LinearDepthGreaterThan
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    q, c = linear_greater_requirements(width)
    target = 2 * width
    a = tuple(range(width))
    b = tuple(range(width, 2 * width))
    w = tuple(range(target + 1, target + q + 1))
    bits = tuple(range(40, 40 + c))
    actual = linear_greater_ir(a, b, target, w, bits)
    source = LinearDepthGreaterThan(width, signed=False)
    measured = {
        g.wires[0]: g.metadata["classical_bit"] for g in actual if g.name == "measure"
    }
    expected = reference_arithmetic_gates(
        source, {"a": a, "b": b, "target": (target,)}, w, measured
    )
    assert_source(actual, expected)
    counts = Counter(g.name for g in actual)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 4 * c
    assert counts["measure"] == model.measurement == c
    assert counts["reset"] == c


@pytest.mark.integration
@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_linear_all_complex_amplitudes_and_clean_work(width: int) -> None:
    q, c = linear_greater_requirements(width)
    target = 2 * width
    gates = linear_greater_ir(
        tuple(range(width)),
        tuple(range(width, 2 * width)),
        target,
        tuple(range(target + 1, target + q + 1)),
        tuple(range(40, 40 + c)),
    )
    check_greater_amplitudes(width, width, q, c, gates)


def test_linear_greater_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        linear_greater_requirements(0)
    with pytest.raises(ValueError, match="equal"):
        linear_greater_ir((0,), (1, 2), 3, (4,), (40,))
    with pytest.raises(ValueError, match="distinct"):
        linear_greater_ir((0,), (1,), 0, (2,), (40,))
    with pytest.raises(ValueError, match="clean work"):
        linear_greater_ir((0,), (1,), 2, (), (40,))
    with pytest.raises(ValueError, match="classical"):
        linear_greater_ir((0,), (1,), 2, (3,), (-1,))
