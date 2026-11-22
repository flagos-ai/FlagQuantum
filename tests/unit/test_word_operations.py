"""Complete immutable-source correspondence for the shared scalar limb owner."""

import json
from pathlib import Path

import pytest

from flagquantum.core._word_operations import (
    WORD_MASK,
    add_carry,
    extract_bits,
    full_multiply_add,
    subtract_borrow,
)

pytestmark = pytest.mark.unit


def test_complete_source_word_operations() -> None:
    fixture = Path(__file__).with_name("data") / "word_operations_reference.jsonl"
    count = 0
    with fixture.open() as stream:
        for line in stream:
            a, b, c, d, low_sum, carry, low_difference, borrow, extracted, low, high = (
                json.loads(line)
            )
            assert add_carry(bool(c), a, b) == (low_sum, bool(carry))
            assert subtract_borrow(bool(c), a, b) == (low_difference, bool(borrow))
            assert extract_bits(a, b) == extracted
            assert full_multiply_add(a, b, c, d) == (low, high)
            count += 1
    assert count == 3472


@pytest.mark.parametrize("bad", [-1, WORD_MASK + 1, True, 1.5])
def test_word_domain_refuses_invalid_values(bad: int) -> None:
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        add_carry(False, bad, 0)
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        subtract_borrow(False, 0, bad)
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        extract_bits(0, bad)
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        full_multiply_add(0, 0, bad, 0)
