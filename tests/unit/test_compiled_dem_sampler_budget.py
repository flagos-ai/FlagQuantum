"""Principal-data limits include all three native tables, even without errors."""

import pytest

from flagquantum.errors import CapabilityError
from flagquantum.simulation.stabilizer._dem_sampling import (
    _check_budget,
    _principal_bytes,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "packed,replay,maximum",
    [
        (False, None, 7010),
        (True, None, 56084),
        (False, False, 4215),
        (True, True, 33720),
    ],
)
def test_exact_payload_boundaries(
    packed: bool, replay: bool | None, maximum: int
) -> None:
    _check_budget(maximum, (1024, 16, 2048), packed, True, replay)
    with pytest.raises(CapabilityError, match="64 MiB"):
        _check_budget(maximum + 1, (1024, 16, 2048), packed, True, replay)


def test_error_native_table_is_counted_without_returning_errors() -> None:
    assert (
        _principal_bytes(0, (1, 0, 2049), True, False, None)
        == 4 * (1024 + 3072) * 128 + 65536
    )
    with pytest.raises(CapabilityError):
        _check_budget(0, (65536, 65536, 65536), True, False, None)


def test_zero_widths_do_not_reserve_a_phantom_column() -> None:
    assert (
        _principal_bytes(100, (1, 0, 0), True, True, True)
        == 4 * 1024 * 128 + 300 + 65536
    )
