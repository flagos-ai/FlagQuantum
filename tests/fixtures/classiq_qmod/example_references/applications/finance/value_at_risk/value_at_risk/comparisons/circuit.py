"""Test-only unchanged original, native and independent Circuit acceptance."""

from tests.integration.test_synthesis_native_amplification import (
    CASES,
    test_original_native_independent_state_and_named_marginal,
)


def run_example() -> None:
    case = next(row for row in CASES if row[0] == "applications.finance.value_at_risk")
    for k in (0, 1, 2):
        test_original_native_independent_state_and_named_marginal(*case, k)


if __name__ == "__main__":
    run_example()
