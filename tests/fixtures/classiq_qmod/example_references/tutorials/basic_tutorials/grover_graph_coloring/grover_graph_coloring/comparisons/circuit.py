"""Test-only source/native/independent full-state acceptance."""

from tests.integration.test_synthesis_native_sparse_grover import (
    test_original_full_iteration_count_matches_native_and_independent,
)


def run_example() -> None:
    test_original_full_iteration_count_matches_native_and_independent("graph")


if __name__ == "__main__":
    run_example()
