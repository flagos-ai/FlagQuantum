"""Test-only unchanged original and native Fourier LCU full-state acceptance."""

from tests.integration.test_synthesis_native_fourier_lcu import (
    test_unchanged_source_native_and_independent_full_states,
)


def run_example() -> None:
    test_unchanged_source_native_and_independent_full_states()


if __name__ == "__main__":
    run_example()
