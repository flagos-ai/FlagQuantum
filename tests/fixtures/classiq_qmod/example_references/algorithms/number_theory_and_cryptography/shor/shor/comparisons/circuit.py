"""Test-only unchanged source/native/independent acceptance."""

from tests.integration.test_synthesis_native_period_feedback import (
    test_ten_bit_period_finding_full_dirty_residue_and_clean_work,
)


def run_example() -> None:
    test_ten_bit_period_finding_full_dirty_residue_and_clean_work(21, 11)


if __name__ == "__main__":
    run_example()
