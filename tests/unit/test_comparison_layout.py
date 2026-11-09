"""One physical comparison pool shared by real offset and bitonic consumers."""

import pytest

from flagquantum.compiler.registers.comparison.comparison_layout import (
    comparison_network_requirements,
    validate_comparison_layout,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("width", [1, 2, 8])
@pytest.mark.parametrize("count", [0, 1, 12])
def test_shared_budgets_exclude_logical_history(width: int, count: int) -> None:
    q, c = comparison_network_requirements(width, count)
    source_budgets = {1: (2, 1), 2: (5, 3), 8: (23, 15)}
    assert (q, c) == (source_budgets[width] if count else (0, 0))
    words = (tuple(range(width)),)
    histories = tuple(range(width, width + count))
    work = tuple(range(width + count, width + count + q))
    bits = tuple(range(c))
    for inverse in (False, True):
        validate_comparison_layout(
            words, histories, work, bits, history_count=count, adjoint=inverse
        )


def test_invalid_physical_roles_and_budgets() -> None:
    for count in (-1, True, 1.5):
        with pytest.raises(ValueError, match="nonnegative"):
            comparison_network_requirements(2, count)
    with pytest.raises(ValueError, match="positive"):
        comparison_network_requirements(0, 0)
    valid = (((0, 1), (2, 3)), (4,), tuple(range(5, 10)), (0, 1, 2))
    with pytest.raises(ValueError, match="boolean"):
        validate_comparison_layout(*valid, history_count=1, adjoint=1)
    for words in ((), ((),), ((0, 1), (2,))):
        with pytest.raises(ValueError, match="equal"):
            validate_comparison_layout(
                words, valid[1], valid[2], valid[3], history_count=1, adjoint=False
            )
    with pytest.raises(ValueError, match="history"):
        validate_comparison_layout(
            valid[0], (), valid[2], valid[3], history_count=1, adjoint=False
        )
    for bits in ((), (0, 0, 2), (-1, 1, 2), (True, 1, 2)):
        with pytest.raises(ValueError, match="clean work"):
            validate_comparison_layout(*valid[:3], bits, history_count=1, adjoint=False)
    with pytest.raises(ValueError, match="distinct"):
        validate_comparison_layout(
            valid[0], (0,), valid[2], valid[3], history_count=1, adjoint=False
        )
