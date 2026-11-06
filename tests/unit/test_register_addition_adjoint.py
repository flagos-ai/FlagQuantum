"""Literal Add inverse gates, modular subtraction phases and clean borrow work."""

from collections import Counter
from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.arithmetic import add_adjoint_ir, add_ir
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit
WIDTHS = [(1, 1), (2, 2), (3, 3), (5, 5), (8, 8), (1, 3), (2, 5), (3, 8)]


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("m,n", WIDTHS)
@pytest.mark.parametrize("montgomery", [False, True])
def test_literal_add_adjoint_source_gates_and_model(
    m: int, n: int, montgomery: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QMontgomeryUInt, QUInt
    from qualtran.bloqs.arithmetic.addition import Add
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    cls = QMontgomeryUInt if montgomery else QUInt
    a, b = tuple(range(m)), tuple(range(m, m + n))
    carry, bits = tuple(range(m + n, m + 2 * n - 1)), tuple(range(40, 40 + n - 1))
    source = Add(cls(m), cls(n)).adjoint()
    gates = add_adjoint_ir(a, b, carry, bits)
    expected = reference_arithmetic_gates(
        source, {"a": a, "b": b}, carry, dict(zip(carry, bits, strict=True))
    )
    assert_source(gates, expected)
    counts = Counter(g.name for g in gates)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 4 * (n - 1)
    assert counts["measure"] == model.measurement == n - 1
    assert counts["cx"] == Counter(g.name for g in add_ir(a, b, carry, bits))["cx"]
    if n == 1:
        assert counts == {"cx": 1}
        assert model.clifford == -3  # Preserve the original source-model exception.


@pytest.mark.parametrize("m,n", [(1, 1), (1, 2), (2, 2), (1, 3), (2, 3), (3, 3)])
def test_every_complex_subtraction_branch_and_forward_restoration(
    m: int, n: int
) -> None:
    a, b = tuple(range(m)), tuple(range(m, m + n))
    carry, bits = tuple(range(m + n, m + 2 * n - 1)), tuple(range(40, 40 + n - 1))
    width = m + 2 * n
    data = torch.arange(1 << (m + n + 1), dtype=torch.int64)
    left, right, spectator = data >> (n + 1), (data >> 1) & ((1 << n) - 1), data & 1
    coeff = data.to(torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    state = torch.zeros(1 << width, dtype=torch.complex128)
    state[((data >> 1) << n) | spectator] = amps
    expected = torch.zeros_like(state)
    mapped = ((left << n) | ((right - left) % (1 << n))) << n | spectator
    expected[mapped] = amps
    inverse = add_adjoint_ir(a, b, carry, bits)
    forward = add_ir(a, b, carry, bits)
    for outcomes in product((0, 1), repeat=n - 1):
        torch.testing.assert_close(
            forced_branch(state, inverse, outcomes),
            expected / 2 ** ((n - 1) / 2),
            rtol=0,
            atol=1e-12,
        )
        torch.testing.assert_close(
            forced_branch(expected, forward, outcomes),
            state / 2 ** ((n - 1) / 2),
            rtol=0,
            atol=1e-12,
        )


def test_add_adjoint_validation() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        add_adjoint_ir((), (), (), ())
    with pytest.raises(ValueError, match="clean carry"):
        add_adjoint_ir((0,), (1, 2), (), ())
    with pytest.raises(ValueError, match="distinct"):
        add_adjoint_ir((0,), (0,), (), ())
    with pytest.raises(ValueError, match="nonnegative"):
        add_adjoint_ir((True,), (1,), (), ())
    with pytest.raises(ValueError, match="classical"):
        add_adjoint_ir((0,), (1, 2), (3,), ())
