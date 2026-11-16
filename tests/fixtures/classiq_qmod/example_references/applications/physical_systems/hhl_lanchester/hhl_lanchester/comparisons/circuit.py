"""Test-only original/native Lanchester full-state acceptance."""

from tests.integration.test_synthesis_native_lanchester import (
    test_complete_lanchester_state_global_phase_field_order_and_clean_work,
)


def run_example() -> None:
    test_complete_lanchester_state_global_phase_field_order_and_clean_work(None)


if __name__ == "__main__":
    run_example()
