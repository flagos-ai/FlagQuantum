"""Frozen sorting factories and complete original offset cases."""

from collections.abc import Hashable
from hashlib import sha1
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest

from examples.algorithms.sorting_catalog import sorting_example_graph, sorting_examples
from flagquantum.compiler.registers.comparison.parallel_comparison import (
    parallel_comparison_pairs,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
SOURCE = (
    Path(__file__).resolve().parents[3] / "ftqc-reference-env/qualtran/bloqs/arithmetic"
)


def test_locked_assets() -> None:
    import kickmix

    assert version("qualtran") == "0.7.0"
    assert kickmix.__version__ == "0.1" and hasattr(kickmix, "Simulator")
    for name, digest in (
        ("sorting.py", "5014d74bc122a0438c520df0442c59a82739e7f1"),
        ("sorting_test.py", "5251bd5ea7e3e6ddb34df986d40a19abe7e0fab6"),
        ("sorting.ipynb", "b9e6568413f1d7aaf6a439f36ccfb7386e3ab864"),
    ):
        data = (SOURCE / name).read_bytes()
        assert sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == digest


def _key(child: Any) -> Hashable:
    from qualtran import QBit
    from qualtran.bloqs.arithmetic import GreaterThan
    from qualtran.bloqs.arithmetic.sorting import Comparator
    from qualtran.bloqs.basic_gates import CSwap
    from qualtran.bloqs.bookkeeping import Allocate

    if isinstance(child, Allocate):
        assert child.dtype == QBit() and not child.dirty
        return ("allocate_bit",)
    if isinstance(child, GreaterThan):
        return ("greater_than", child.a_bitsize, child.b_bitsize)
    if isinstance(child, CSwap):
        return ("controlled_swap", child.bitsize)
    if isinstance(child, Comparator):
        return ("comparator", child.bitsize)
    raise AssertionError(f"Unexpected sorting child {child}")


@pytest.mark.parametrize("index", range(6))
def test_all_original_sorting_factories(index: int) -> None:
    from qualtran import QBit, QUInt
    from qualtran.bloqs.arithmetic import sorting

    source = (
        sorting._comparator,
        sorting._comparator_symb,
        sorting._bitonic_sort,
        sorting._bitonic_sort_symb,
        sorting._parallel_compare,
        sorting._bitonic_merge,
    )[index].make()
    example = sorting_examples()[index]
    name, signature, _ = example
    assert all(isinstance(r.dtype, (QBit, QUInt)) for r in source.signature)
    assert [(r.name, r.bitsize, r.shape_symbolic, r.side.value) for r in signature] == [
        (r.name, r.bitsize, r.shape_symbolic, r.side.value) for r in source.signature
    ]
    graph, leaves = source.call_graph(max_depth=1)
    actual = sorting_example_graph(example)
    assert actual.direct_calls == {
        (name, _key(child)): edge["n"] for _, child, edge in graph.edges(data=True)
    }
    assert actual.leaf_counts == {_key(child): count for child, count in leaves.items()}


@pytest.mark.parametrize("count", range(1, 10))
@pytest.mark.parametrize("offset", range(1, 10))
def test_all_original_parallel_cases_keep_pair_and_history_order(
    count: int, offset: int
) -> None:
    from qualtran.bloqs.arithmetic.sorting import Comparator, ParallelComparators

    source = ParallelComparators(count, offset, 3)
    decomposition = source.decompose_bloq()
    entries = sorted(decomposition.iter_bloqsoqs(), key=lambda entry: entry[0].i)
    actual = []
    histories = []
    for instance, inputs, outputs in entries:
        assert instance.bloq == Comparator(3)
        actual.append((inputs["a"].idx[0], inputs["b"].idx[0]))
        histories.append(outputs[-1])
    assert tuple(actual) == parallel_comparison_pairs(count, offset)
    assert tuple(decomposition.final_soqs()["junk"]) == tuple(histories)
    assert source.num_comparisons == len(actual)


@pytest.mark.parametrize(
    "merge,count",
    [(True, k) for k in (2, 4, 8, 16)] + [(False, k) for k in (1, 2, 4, 8, 16)],
)
def test_original_seeded_lists_and_history_bits(merge: bool, count: int) -> None:
    import numpy as np
    from qualtran.bloqs.arithmetic.sorting import BitonicMerge, BitonicSort

    from flagquantum.compiler.registers.sorting.bitonic_merge import (
        bitonic_merge_history_count,
        bitonic_merge_ir,
    )
    from flagquantum.compiler.registers.sorting.bitonic_sort import (
        bitonic_sort_history_count,
        bitonic_sort_ir,
    )
    from tests.unit.measurement_checks import certified_and_channel_map
    from tests.unit.test_bitonic_merge import layout, pack

    width = 3
    total = 2 * count if merge else count
    rng = np.random.default_rng(1024)
    if merge:
        source = BitonicMerge(count, width)
        samples = [
            tuple(sorted(a.tolist()) + sorted(b.tolist()))
            for a, b in (rng.integers(0, 8, size=(2, count)) for _ in range(5))
        ]
        histories = bitonic_merge_history_count(count)
    else:
        source = BitonicSort(count, width)
        samples = [tuple(rng.integers(0, 8, size=count).tolist()) for _ in range(5)]
        histories = bitonic_sort_history_count(count)
    words, history, work, bits = layout(total, width, histories)
    physical_width = total * width + histories + len(work)

    def encode(values: tuple[int, ...], junk: tuple[int, ...]) -> tuple[int, ...]:
        packed = pack(values, width, junk, len(work))
        return tuple((packed >> bit) & 1 for bit in reversed(range(physical_width)))

    if merge:
        gates = bitonic_merge_ir(words, history, work, bits)
    else:
        gates = bitonic_sort_ir(words, history, work, bits)
    before = tuple(encode(values, (0,) * histories) for values in samples)
    actual, events = certified_and_channel_map(gates, before)
    assert events == histories * (2 * width - 1)
    for values, observed in zip(samples, actual, strict=True):
        inputs = {"xs": np.array(values[:count])}
        if merge:
            inputs["ys"] = np.array(values[count:])
        result, junk = source.call_classically(**inputs)
        expected = encode(tuple(map(int, result)), tuple(map(int, junk)))
        assert observed == expected
