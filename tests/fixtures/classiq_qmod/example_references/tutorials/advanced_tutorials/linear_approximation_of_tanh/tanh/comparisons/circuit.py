"""Test-only source, native and independent absolute full-state acceptance."""

from tests.integration.test_synthesis_native_tanh import (
    test_original_native_and_independent_full_state,
)


def run_example() -> None:
    test_original_native_and_independent_full_state()


if __name__ == "__main__":
    run_example()
