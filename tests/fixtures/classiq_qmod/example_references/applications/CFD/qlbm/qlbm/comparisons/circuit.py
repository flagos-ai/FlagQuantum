"""Test-only unchanged source/native/Circuit QLBM acceptance."""

from tests.integration.test_synthesis_native_qlbm import (
    test_approved_source_native_and_independent_complete_state,
)


def run_example() -> None:
    test_approved_source_native_and_independent_complete_state()


if __name__ == "__main__":
    run_example()
