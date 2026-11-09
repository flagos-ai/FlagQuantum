"""Original Comparator with a logical history and separately clean comparison work."""

from collections import Counter
from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.comparison.compare_exchange import (
    compare_exchange_ir,
)
from flagquantum.compiler.registers.comparison.less_equal import less_equal_requirements
from flagquantum.core.ir import Instruction
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


def layout(
    n: int,
) -> tuple[tuple[int, ...], tuple[int, ...], int, tuple[int, ...], tuple[int, ...]]:
    q, c = less_equal_requirements(n, n)
    return (
        tuple(range(n)),
        tuple(range(n, 2 * n)),
        2 * n,
        tuple(range(2 * n + 1, 2 * n + 1 + q)),
        tuple(range(40, 40 + c)),
    )


def original_gates(n: int, *, adjoint: bool = False) -> tuple[Instruction, ...]:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.arithmetic.sorting import Comparator

    a, b, h, w, bits = layout(n)
    source = Comparator(n).adjoint() if adjoint else Comparator(n)
    # Retain original RIGHT/LEFT history ports and whole comparator lifetimes.
    composite = source.decompose_bloq()
    gates = compare_exchange_ir(a, b, h, w, bits, adjoint=adjoint)
    measured = {
        g.wires[0]: g.metadata["classical_bit"] for g in gates if g.name == "measure"
    }
    inputs = {"a": a, "b": b} | ({"out": (h,)} if adjoint else {})
    outputs = {"a": a, "b": b} | ({} if adjoint else {"out": (h,)})
    return reference_arithmetic_gates(
        composite,
        inputs,
        w if adjoint else (h, *w),
        measured,
        reuse_allocations=False,
        preserve_components=True,
        expected_outputs=outputs,
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("adjoint", [False, True])
def test_original_source_gates_feedback_and_model(n: int, adjoint: bool) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.arithmetic.sorting import Comparator
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    a, b, h, w, bits = layout(n)
    gates = compare_exchange_ir(a, b, h, w, bits, adjoint=adjoint)
    assert_source(gates, original_gates(n, adjoint=adjoint))
    source = Comparator(n).adjoint() if adjoint else Comparator(n)
    counts = Counter(g.name for g in gates)
    model = get_cost_value(source, QECGatesCost())
    assert (
        counts["t"] + counts["tdg"] == model.total_t_count(ts_per_cswap=7) == 15 * n - 4
    )
    assert model.total_t_count() == 12 * n - 4
    assert counts["measure"] == model.measurement == 2 * n - 1
    assert get_cost_value(source, QubitCount()) == 2 * n + 1 + len(w)


def basis_inputs(
    n: int,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    a, b, h, w, bits = layout(n)
    width = 2 * n + 1 + len(w)
    before, after = [], []
    for x, y in product(range(1 << n), repeat=2):
        av = ((x << n) | y) << (len(w) + 1)
        bv = (((min(x, y) << n) | max(x, y)) << 1 | int(x > y)) << len(w)
        before.append(tuple((av >> (width - 1 - i)) & 1 for i in range(width)))
        after.append(tuple((bv >> (width - 1 - i)) & 1 for i in range(width)))
    return tuple(before), tuple(after)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_every_input_history_work_zero_and_adjoint_restore(n: int) -> None:
    a, b, h, w, bits = layout(n)
    before, expected = basis_inputs(n)
    actual, m = certified_and_channel_map(compare_exchange_ir(a, b, h, w, bits), before)
    assert actual == expected and m == 2 * n - 1
    restored, m2 = certified_and_channel_map(
        compare_exchange_ir(a, b, h, w, bits, adjoint=True), actual
    )
    assert restored == before and m2 == m


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("adjoint", [False, True])
def test_all_small_raw_complex_histories_and_clean_work(n: int, adjoint: bool) -> None:
    a, b, h, w, bits = layout(n)
    before, after = basis_inputs(n)
    if adjoint:
        before, after = after, before
    width = len(before[0])
    state = torch.zeros(1 << (width + 1), dtype=torch.complex128)
    expected = torch.zeros_like(state)
    v = torch.arange(2 * len(before), dtype=torch.float64)
    amps = torch.complex(v + 1, 3 * v - 5)
    amps /= amps.norm()
    for i, (x, y) in enumerate(zip(before, after, strict=True)):
        xv = sum(bit << (width - 1 - j) for j, bit in enumerate(x))
        yv = sum(bit << (width - 1 - j) for j, bit in enumerate(y))
        for s in (0, 1):
            state[(xv << 1) | s] = amps[2 * i + s]
            expected[(yv << 1) | s] = amps[2 * i + s]
    gates = compare_exchange_ir(a, b, h, w, bits, adjoint=adjoint)
    m = 2 * n - 1
    for branch in product((0, 1), repeat=m):
        torch.testing.assert_close(
            forced_branch(state, gates, branch),
            expected / 2 ** (m / 2),
            rtol=0,
            atol=1e-12,
        )


def test_history_is_logical_output_and_invalid_inverse_is_not_reset() -> None:
    a, b, h, w, bits = layout(2)
    before, expected = basis_inputs(2)
    assert any(word[h] for word in expected)
    # Equal words with history=1 are not a valid forward output. Retain the
    # nonzero bit instead of silently resetting or claiming inverse-domain cleanup.
    invalid = (0, 0, 0, 0, 1, *(0,) * len(w))
    actual, _ = certified_and_channel_map(
        compare_exchange_ir(a, b, h, w, bits, adjoint=True), (invalid,)
    )
    assert actual == (invalid,)
    with pytest.raises(ValueError, match="boolean"):
        compare_exchange_ir(a, b, h, w, bits, adjoint=1)
    with pytest.raises(ValueError, match="equal"):
        compare_exchange_ir(a, b[:1], h, w, bits)
    with pytest.raises(ValueError, match="distinct"):
        compare_exchange_ir(a, b, a[0], w, bits)
