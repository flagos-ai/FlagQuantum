"""Test-only frozen original/native/independent Circuit acceptance."""

from tests.integration.test_synthesis_native_rainbow_direct import (
    test_original_native_and_independent_complete_environment,
)


def run_example() -> None:
    for k in (0, 1, 2):
        test_original_native_and_independent_complete_environment(k)


if __name__ == "__main__":
    run_example()
