"""Original GreaterThan adjoint before sharing it with compare/exchange."""

from collections import Counter
from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.comparison.ordering import (
    greater_adjoint_ir,
    greater_ir,
)
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_less_equal import CASES, layout
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("m,n", CASES)
def test_original_inverse_gates_feedback_and_models(m: int, n: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.arithmetic.comparison import GreaterThan, LessThanEqual
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    a, b, t, w, bits = layout(m, n)
    gates = greater_adjoint_ir(a, b, t, w, bits)
    measured = {
        g.wires[0]: g.metadata["classical_bit"] for g in gates if g.name == "measure"
    }
    source = GreaterThan(m, n).adjoint()
    assert LessThanEqual(m, n).adjoint() == LessThanEqual(m, n)
    expected = reference_arithmetic_gates(
        source.decompose_bloq(),
        {"a": a, "b": b, "target": (t,)},
        w,
        measured,
        cirq_style=True,
        preserve_components=True,
    )
    assert_source(gates, expected)
    model = get_cost_value(source, QECGatesCost())
    counts = Counter(g.name for g in gates)
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 4 * len(bits)
    assert counts["measure"] == model.measurement == len(bits)


@pytest.mark.parametrize("m,n", [(1, 1), (2, 2), (3, 3), (1, 3), (3, 1)])
def test_full_complex_columns_all_branches_and_forward_restore(m: int, n: int) -> None:
    a, b, t, w, bits = layout(m, n)
    width = m + n + 1 + len(w)
    before, after = [], []
    for x, y, z in product(range(1 << m), range(1 << n), (0, 1)):
        av = (((x << n) | y) << 1 | z) << len(w)
        bv = (((x << n) | y) << 1 | (z ^ int(x > y))) << len(w)
        before.append(tuple((av >> (width - 1 - i)) & 1 for i in range(width)))
        after.append(tuple((bv >> (width - 1 - i)) & 1 for i in range(width)))
    gates = greater_adjoint_ir(a, b, t, w, bits)
    actual, events = certified_and_channel_map(gates, tuple(before))
    assert actual == tuple(after) and events == len(bits)
    restored, _ = certified_and_channel_map(greater_ir(a, b, t, w, bits), actual)
    assert restored == tuple(before)
    state = torch.zeros(1 << (width + 1), dtype=torch.complex128)
    expected = torch.zeros_like(state)
    coeff = torch.arange(2 * len(before), dtype=torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    for i, (x, y) in enumerate(zip(before, after, strict=True)):
        xv = sum(v << (width - 1 - j) for j, v in enumerate(x))
        yv = sum(v << (width - 1 - j) for j, v in enumerate(y))
        for s in (0, 1):
            state[(xv << 1) | s] = amps[2 * i + s]
            expected[(yv << 1) | s] = amps[2 * i + s]
    for branch in product((0, 1), repeat=events):
        torch.testing.assert_close(
            forced_branch(state, gates, branch),
            expected / 2 ** (events / 2),
            rtol=0,
            atol=1e-12,
        )


def test_original_inverse_order_is_retained() -> None:
    a, b, t, w, bits = layout(3, 3)
    inverse = greater_adjoint_ir(a, b, t, w, bits)
    forward = greater_ir(a, b, t, w, bits)
    assert inverse[0].name == "x" and inverse[0].wires == (t,)
    assert forward[-1] == inverse[0] and inverse[1:] == forward[:-1]
    with pytest.raises(ValueError, match="clean work"):
        greater_adjoint_ir(a, b, t, w[:-1], bits)
