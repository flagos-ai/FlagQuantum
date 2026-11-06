"""Original recursive bitonic sort and exact inverse through shared primitives."""

from collections import Counter

import pytest

from flagquantum.compiler.registers.bitonic_sort import (
    bitonic_sort_history_count,
    bitonic_sort_ir,
)
from tests.unit.measurement_checks import certified_and_channel_map
from tests.unit.test_bitonic_merge import (
    input_words,
    layout,
    pack,
    raw_channels,
    reference_network,
)
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit
CASES = [(1, 1), (2, 1), (2, 2), (2, 8), (4, 1), (4, 2), (8, 1), (8, 3)]


@pytest.mark.parametrize("k,n", CASES)
@pytest.mark.parametrize("adjoint", [False, True])
def test_original_recursion_ports_gates_and_models(
    k: int, n: int, adjoint: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.arithmetic.sorting import BitonicSort
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    count = bitonic_sort_history_count(k)
    words, h, w, bits = layout(k, n, count)
    gates = bitonic_sort_ir(words, h, w, bits, adjoint=adjoint)
    parent = BitonicSort(k, n)
    assert_source(
        gates,
        reference_network(parent, words, h, w, bits, merge=False, adjoint=adjoint),
    )
    source = parent.adjoint() if adjoint else parent
    model = get_cost_value(source, QECGatesCost())
    counts = Counter(g.name for g in gates)
    assert count == parent.num_comparisons
    assert (
        counts["t"] + counts["tdg"]
        == model.total_t_count(ts_per_cswap=7)
        == count * (15 * n - 4)
    )
    assert model.total_t_count() == count * (12 * n - 4)
    assert counts["measure"] == model.measurement == count * (2 * n - 1)
    assert get_cost_value(source, QubitCount()) == k * n + count + len(w)


@pytest.mark.parametrize("k,n", CASES)
def test_all_small_input_lists_history_work_and_restore(k: int, n: int) -> None:
    count = bitonic_sort_history_count(k)
    words, h, w, bits = layout(k, n, count)
    labels = input_words(k, n, merge=False)
    width = k * n + count + len(w)
    before = tuple(
        tuple(
            (pack(v, n, (0,) * count, len(w)) >> (width - 1 - i)) & 1
            for i in range(width)
        )
        for v in labels
    )
    actual, m = certified_and_channel_map(bitonic_sort_ir(words, h, w, bits), before)
    assert m == count * (2 * n - 1)
    for values, out in zip(labels, actual, strict=True):
        observed = tuple(
            sum(out[i * n + j] << (n - 1 - j) for j in range(n)) for i in range(k)
        )
        assert observed == tuple(sorted(values))
        if w:
            assert not any(out[-len(w) :])
    restored, m2 = certified_and_channel_map(
        bitonic_sort_ir(words, h, w, bits, adjoint=True), actual
    )
    assert restored == before and m2 == m


@pytest.mark.parametrize("k,n", [(2, 1), (4, 1), (2, 2)])
@pytest.mark.parametrize("adjoint", [False, True])
def test_all_small_raw_complex_histories_and_source_values(
    k: int, n: int, adjoint: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.arithmetic.sorting import BitonicSort

    words, h, w, bits = layout(k, n, bitonic_sort_history_count(k))
    raw_channels(BitonicSort(k, n), words, h, w, bits, merge=False, adjoint=adjoint)


def test_source_bounds_and_native_layout_refusal() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.arithmetic.sorting import BitonicSort

    for k in (0, 3, True):
        with pytest.raises(ValueError, match="power of two"):
            bitonic_sort_history_count(k)
    with pytest.raises(AssertionError):
        BitonicSort(3, 2)
    words, h, w, bits = layout(4, 2, 6)
    with pytest.raises(ValueError, match="history"):
        bitonic_sort_ir(words, h[:-1], w, bits)
    with pytest.raises(ValueError, match="boolean"):
        bitonic_sort_ir(words, h, w, bits, adjoint=1)


def test_native_append_example() -> None:
    from examples.algorithms.register_bitonic_sort import main

    main()
