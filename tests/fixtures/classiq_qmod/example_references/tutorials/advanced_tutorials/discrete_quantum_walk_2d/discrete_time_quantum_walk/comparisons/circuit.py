"""Test-only unchanged source/native/Circuit grid-walk acceptance."""

from tests.integration.test_synthesis_native_grid_walk import (
    test_original_native_and_independent_full_state,
)


def run_example() -> None:
    for steps in (0, 1, 2, 3):
        test_original_native_and_independent_full_state(steps)


if __name__ == "__main__":
    run_example()
