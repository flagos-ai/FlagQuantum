"""Test-only unchanged source/native/independent full-state acceptance."""

from tests.integration.test_synthesis_native_controlled_preparation import (
    test_frozen_original_native_and_independent_absolute_full_state,
)


def run_example() -> None:
    test_frozen_original_native_and_independent_absolute_full_state()


if __name__ == "__main__":
    run_example()
