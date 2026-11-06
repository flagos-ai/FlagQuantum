"""Original ordered cycle construction for concrete partial permutations."""

from itertools import combinations, permutations

import pytest

from flagquantum.compiler.registers.permutation import source_partial_permutation_cycles

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
def test_original_ordered_map_cycles() -> None:
    pytest.importorskip("qualtran")
    from qualtran.linalg.permutation import decompose_permutation_map_into_cycles

    for count in range(5):
        for keys in combinations(range(4), count):
            for ordered in (keys, keys[::-1]):
                for values in permutations(range(4), count):
                    mapping = dict(zip(ordered, values, strict=True))
                    assert source_partial_permutation_cycles(4, mapping) == tuple(
                        decompose_permutation_map_into_cycles(mapping)
                    )


@pytest.mark.parametrize("mapping", [{-1: 0}, {0: 4}, {1: 0, 2: 0}, {0: True}])
def test_invalid_partial_map(mapping: dict[int, int]) -> None:
    with pytest.raises(ValueError, match="injective and in range"):
        source_partial_permutation_cycles(4, mapping)
