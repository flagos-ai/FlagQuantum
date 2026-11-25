"""Test-only original matrix HHL acceptance entrypoint."""

from tests.integration.test_synthesis_native_matrix_hhl import (
    test_original_native_and_independent_complete_hhl_state,
)


def run_example() -> None:
    test_original_native_and_independent_complete_hhl_state(
        "applications/finance/portfolio_optimization_hhl/HHL_portfolio"
    )


if __name__ == "__main__":
    run_example()
