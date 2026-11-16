"""Verify complete partial injections without concealing frozen source defects."""

from collections import deque
from itertools import combinations, permutations

import pytest

from flagquantum.compiler.registers.sorting.permutation import (
    complete_partial_permutation,
    source_partial_permutation_cycles,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def _apply(cycles: tuple[tuple[int, ...], ...], size: int) -> tuple[int, ...]:
    result = list(range(size))
    for cycle in cycles:
        for source, target in zip(cycle, (*cycle[1:], cycle[0]), strict=True):
            result[source] = target
    return tuple(result)


def _swap_distances(size: int) -> dict[tuple[int, ...], int]:
    """Compute the minimum transposition distance by independent BFS."""
    identity = tuple(range(size))
    distances = {identity: 0}
    queue = deque((identity,))
    while queue:
        before = queue.popleft()
        for i, j in combinations(range(size), 2):
            values = list(before)
            values[i], values[j] = values[j], values[i]
            after = tuple(values)
            if after not in distances:
                distances[after] = distances[before] + 1
                queue.append(after)
    return distances


def test_exhaustive_four_vertex_completions_and_working_source_parity() -> None:
    from qualtran.linalg.permutation import decompose_permutation_map_into_cycles

    size = 4
    distances = _swap_distances(size)
    source_cases, repaired_cases = 0, 0
    for width in range(size + 1):
        for keys in combinations(range(size), width):
            for values in permutations(range(size), width):
                edges = tuple(zip(keys, values, strict=True))
                minimum = min(
                    distance
                    for candidate, distance in distances.items()
                    if all(candidate[a] == b for a, b in edges)
                )
                for ordering in permutations(edges):
                    mapping = dict(ordering)
                    cycles = complete_partial_permutation(size, mapping)
                    assert len({v for cycle in cycles for v in cycle}) == sum(
                        len(cycle) for cycle in cycles
                    )
                    actual = _apply(cycles, size)
                    assert all(actual[a] == b for a, b in edges)
                    assert distances[actual] == minimum
                    source = tuple(decompose_permutation_map_into_cycles(mapping))
                    assert source_partial_permutation_cycles(size, mapping) == source
                    source_map = _apply(source, size)
                    if all(source_map[a] == b for a, b in edges):
                        assert cycles == source
                        source_cases += 1
                    else:
                        repaired_cases += 1
    assert (source_cases, repaired_cases) == (1121, 192)


def test_prefix_loss_witness_and_component_order() -> None:
    mapping = {0: 1, 5: 6, 6: 5, 1: 3, 2: 0}
    assert source_partial_permutation_cycles(7, mapping) == ((0, 1, 3), (5, 6))
    assert complete_partial_permutation(7, mapping) == ((2, 0, 1, 3), (5, 6))


@pytest.mark.parametrize("size,mapping", [(0, {}), (4, {0: 3, 1: 3}), (4, {0: 4})])
def test_invalid_injections_share_validation(
    size: int, mapping: dict[int, int]
) -> None:
    with pytest.raises(ValueError) as original:
        source_partial_permutation_cycles(size, mapping)
    with pytest.raises(ValueError) as actual:
        complete_partial_permutation(size, mapping)
    assert str(actual.value) == str(original.value)
