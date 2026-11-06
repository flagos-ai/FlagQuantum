"""Original offset comparator traversal, history ownership and clean shared work."""

from collections import Counter
from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.less_equal import less_equal_requirements
from flagquantum.compiler.registers.parallel_comparison import (
    parallel_compare_ir,
    parallel_comparison_pairs,
)
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit
CASES = [(2, 1, 1), (3, 1, 2), (4, 1, 2), (5, 2, 1), (7, 2, 3), (8, 4, 2), (2, 1, 8)]


def layout(
    k: int, offset: int, n: int
) -> tuple[
    tuple[tuple[int, ...], ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]
]:
    pairs = parallel_comparison_pairs(k, offset)
    q, c = less_equal_requirements(n, n) if pairs else (0, 0)
    return (
        tuple(tuple(range(i * n, (i + 1) * n)) for i in range(k)),
        tuple(range(k * n, k * n + len(pairs))),
        tuple(range(k * n + len(pairs), k * n + len(pairs) + q)),
        tuple(range(40, 40 + c)),
    )


@pytest.mark.parametrize("k,offset,n", CASES)
@pytest.mark.parametrize("adjoint", [False, True])
def test_original_traversal_ports_full_source_and_models(
    k: int, offset: int, n: int, adjoint: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.arithmetic.sorting import ParallelComparators
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    words, history, w, bits = layout(k, offset, n)
    pairs = parallel_comparison_pairs(k, offset)
    parent = ParallelComparators(k, offset, n)
    source = parent.adjoint() if adjoint else parent
    cb = source.decompose_bloq()
    # Only expand original Comparator nodes once; keep GreaterThan and CSwap
    # as complete original components so their temporary lifetimes stay intact.
    cb = cb.flatten_once(lambda bi: type(bi.bloq).__name__ in {"Comparator", "Adjoint"})
    flat = tuple(wire for word in words for wire in word)
    actual = parallel_compare_ir(words, offset, history, w, bits, adjoint=adjoint)
    measured = {
        g.wires[0]: g.metadata["classical_bit"] for g in actual if g.name == "measure"
    }
    inputs = {"xs": flat} | ({"junk": history} if adjoint else {})
    outputs = {"xs": flat} | ({} if adjoint else {"junk": history})
    allocations = (
        tuple(w) * len(pairs) if adjoint else (*history, *(tuple(w) * len(pairs)))
    )
    expected = reference_arithmetic_gates(
        cb,
        inputs,
        allocations,
        measured,
        reuse_allocations=False,
        preserve_components=True,
        expected_outputs=outputs,
    )
    assert_source(actual, expected)
    counts = Counter(g.name for g in actual)
    model = get_cost_value(source, QECGatesCost())
    assert len(pairs) == parent.num_comparisons
    assert (
        counts["t"] + counts["tdg"]
        == model.total_t_count(ts_per_cswap=7)
        == len(pairs) * (15 * n - 4)
    )
    assert model.total_t_count() == len(pairs) * (12 * n - 4)
    assert counts["measure"] == model.measurement == len(pairs) * (2 * n - 1)
    assert get_cost_value(source, QubitCount()) == k * n + len(history) + len(w)


def basis_inputs(
    k: int, offset: int, n: int
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    words, h, w, bits = layout(k, offset, n)
    total = k * n + len(h) + len(w)
    labels = (
        list(product(range(1 << n), repeat=k))
        if k * n <= 8
        else [
            tuple([0] * k),
            tuple([(1 << n) - 1] * k),
            tuple(i % (1 << n) for i in range(k)),
            tuple((k - i) % (1 << n) for i in range(k)),
        ]
    )
    before, after = [], []
    for values in labels:
        sorted_values = list(values)
        markers = []
        for i, j in parallel_comparison_pairs(k, offset):
            markers.append(int(values[i] > values[j]))
            sorted_values[i], sorted_values[j] = min(values[i], values[j]), max(
                values[i], values[j]
            )
        x = y = 0
        for value, other in zip(values, sorted_values, strict=True):
            x = (x << n) | value
            y = (y << n) | other
        x <<= len(h) + len(w)
        for marker in markers:
            y = (y << 1) | marker
        y <<= len(w)
        before.append(tuple((x >> (total - 1 - i)) & 1 for i in range(total)))
        after.append(tuple((y >> (total - 1 - i)) & 1 for i in range(total)))
    return tuple(before), tuple(after)


@pytest.mark.parametrize("k,offset,n", CASES)
def test_basis_histories_work_and_original_inverse(k: int, offset: int, n: int) -> None:
    words, h, w, bits = layout(k, offset, n)
    before, expected = basis_inputs(k, offset, n)
    actual, m = certified_and_channel_map(
        parallel_compare_ir(words, offset, h, w, bits), before
    )
    assert actual == expected and m == len(h) * (2 * n - 1)
    restored, m2 = certified_and_channel_map(
        parallel_compare_ir(words, offset, h, w, bits, adjoint=True), actual
    )
    assert restored == before and m == m2


@pytest.mark.parametrize("k,offset,n", [(4, 1, 1), (5, 2, 1), (3, 1, 2)])
@pytest.mark.parametrize("adjoint", [False, True])
def test_full_raw_complex_branches_and_work(
    k: int, offset: int, n: int, adjoint: bool
) -> None:
    words, h, w, bits = layout(k, offset, n)
    before, after = basis_inputs(k, offset, n)
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
    gates = parallel_compare_ir(words, offset, h, w, bits, adjoint=adjoint)
    m = len(h) * (2 * n - 1)
    for branch in product((0, 1), repeat=m):
        torch.testing.assert_close(
            forced_branch(state, gates, branch),
            expected / 2 ** (m / 2),
            rtol=0,
            atol=1e-12,
        )


def test_zero_pairs_and_validation() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.arithmetic.sorting import ParallelComparators

    for k, offset in ((1, 1), (3, 5)):
        words, h, w, bits = layout(k, offset, 2)
        source = ParallelComparators(k, offset, 2)
        assert (
            source.num_comparisons == 0 and not source.decompose_bloq().bloq_instances
        )
        assert parallel_compare_ir(words, offset, h, w, bits) == ()
        assert parallel_compare_ir(words, offset, h, w, bits, adjoint=True) == ()
    words, h, w, bits = layout(4, 1, 2)
    for k, offset in ((0, 1), (3, 0), (3, True)):
        with pytest.raises(ValueError, match="positive"):
            parallel_comparison_pairs(k, offset)
    with pytest.raises(ValueError, match="boolean"):
        parallel_compare_ir(words, 1, h, w, bits, adjoint=1)
    with pytest.raises(ValueError, match="equal"):
        parallel_compare_ir((words[0], words[1][:1]), 1, h, w, bits)
    with pytest.raises(ValueError, match="history"):
        parallel_compare_ir(words, 1, h[:1], w, bits)
    with pytest.raises(ValueError, match="clean work"):
        parallel_compare_ir(words, 1, h, w, ())
    with pytest.raises(ValueError, match="distinct"):
        parallel_compare_ir(words, 1, (words[0][0], h[1]), w, bits)


def test_user_append_example() -> None:
    from examples.algorithms.register_compare_exchange import main

    main()
