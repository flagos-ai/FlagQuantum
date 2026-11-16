"""Test-only original matrix HHL acceptance entrypoint."""

from tests.integration.test_synthesis_native_matrix_hhl import (
    test_original_native_and_independent_complete_hhl_state,
)


def run_example() -> None:
    test_original_native_and_independent_complete_hhl_state(
        "applications/finance/hybrid_hhl_for_portfolio_optimization/portfolio_optimization_with_hhl"
    )


if __name__ == "__main__":
    run_example()
