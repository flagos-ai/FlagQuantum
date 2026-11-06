"""Pinned CAdd adjoints keep both control polarities and shared work lifetimes."""

from collections import Counter
from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.controlled_addition import (
    controlled_add_adjoint_ir,
    controlled_add_ir,
)
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_controlled_addition import layout
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit
CASES = [(1, 1), (2, 2), (3, 3), (5, 5), (8, 8), (1, 3), (3, 8)]


@pytest.mark.parametrize("m,n", CASES)
@pytest.mark.parametrize("dtype", ["unsigned", "signed", "montgomery"])
@pytest.mark.parametrize("cv", [0, 1])
def test_source_gates_feedback_and_models(m: int, n: int, dtype: str, cv: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QInt, QMontgomeryUInt, QUInt
    from qualtran.bloqs.arithmetic.controlled_addition import CAdd
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    cls = {"unsigned": QUInt, "signed": QInt, "montgomery": QMontgomeryUInt}[dtype]
    a, b, w, c, wb, cb = layout(m, n)
    source = CAdd(cls(m), cls(n), cv=cv).adjoint()
    expected = reference_arithmetic_gates(
        source,
        {"ctrl": (0,), "a": a, "b": b},
        (*w, *c),
        dict(zip((*w, *c), (*wb, *cb), strict=True)),
    )
    actual = controlled_add_adjoint_ir(0, a, b, w, c, wb, cb, control_value=cv)
    assert_source(actual, expected)
    counts = Counter(g.name for g in actual)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 4 * (m + n - 1)
    assert counts["measure"] == model.measurement == m + n - 1
    assert get_cost_value(source, QubitCount()) == 2 * (m + n)
    assert counts["x"] == (4 * m if cv == 0 else 0)


def basis(
    m: int, n: int, cv: int
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    width = 2 * (m + n)
    work = m + n - 1
    a = []
    b = []
    for control in (0, 1):
        for x in range(1 << m):
            for y in range(1 << n):
                z = (y - x) % (1 << n) if control == cv else y
                before = ((control << (m + n)) | (x << n) | y) << work
                after = ((control << (m + n)) | (x << n) | z) << work
                a.append(tuple((before >> (width - 1 - i)) & 1 for i in range(width)))
                b.append(tuple((after >> (width - 1 - i)) & 1 for i in range(width)))
    return tuple(a), tuple(b)


@pytest.mark.parametrize("m,n", CASES)
@pytest.mark.parametrize("cv", [0, 1])
def test_every_input_every_branch_and_forward_restore(m: int, n: int, cv: int) -> None:
    a, b, w, c, wb, cb = layout(m, n)
    before, expected = basis(m, n, cv)
    actual, events = certified_and_channel_map(
        controlled_add_adjoint_ir(0, a, b, w, c, wb, cb, control_value=cv), before
    )
    assert actual == expected and events == m + n - 1
    restored, again = certified_and_channel_map(
        controlled_add_ir(0, a, b, w, c, wb, cb, control_value=cv), actual
    )
    assert restored == before and again == events


@pytest.mark.parametrize("m,n", [(1, 1), (2, 3)])
@pytest.mark.parametrize("cv", [0, 1])
def test_all_raw_measurement_branches_complex_spectator(
    m: int, n: int, cv: int
) -> None:
    a, b, w, c, wb, cb = layout(m, n)
    before, after = basis(m, n, cv)
    width = len(before[0]) + 1
    state = torch.zeros(1 << width, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    v = torch.arange(2 * len(before), dtype=torch.float64)
    amps = torch.complex(v + 1, 3 * v - 5)
    amps /= amps.norm()
    for i, (x, y) in enumerate(zip(before, after, strict=True)):
        xv = sum(bit << (len(x) - 1 - j) for j, bit in enumerate(x))
        yv = sum(bit << (len(y) - 1 - j) for j, bit in enumerate(y))
        for s in (0, 1):
            state[(xv << 1) | s] = amps[2 * i + s]
            expected[(yv << 1) | s] = amps[2 * i + s]
    gates = controlled_add_adjoint_ir(0, a, b, w, c, wb, cb, control_value=cv)
    for branch in product((0, 1), repeat=m + n - 1):
        torch.testing.assert_close(
            forced_branch(state, gates, branch),
            expected / 2 ** ((m + n - 1) / 2),
            rtol=0,
            atol=1e-12,
        )


def test_shared_validation() -> None:
    a, b, w, c, wb, cb = layout(2, 3)
    for cv in (True, 2):
        with pytest.raises(ValueError, match="zero or one"):
            controlled_add_adjoint_ir(0, a, b, w, c, wb, cb, control_value=cv)
    with pytest.raises(ValueError, match="distinct"):
        controlled_add_adjoint_ir(a[0], a, b, w, c, wb, cb)
    with pytest.raises(ValueError, match="loaded"):
        controlled_add_adjoint_ir(0, a, b, w[:-1], c, wb, cb)
    with pytest.raises(ValueError, match="classical"):
        controlled_add_adjoint_ir(0, a, b, w, c, wb, wb)


def test_exact_adjoint_is_not_complemented_forward_resynthesis() -> None:
    from flagquantum.core.ir import Instruction

    a, b, w, c, wb, cb = layout(3, 3)
    exact = controlled_add_adjoint_ir(0, a, b, w, c, wb, cb)
    flips = tuple(Instruction("x", (wire,)) for wire in b)
    equivalent = (*flips, *controlled_add_ir(0, a, b, w, c, wb, cb), *flips)
    assert exact != equivalent
    assert sum(g.name == "x" for g in exact) == 0
    assert sum(g.name == "x" for g in equivalent) == 6
