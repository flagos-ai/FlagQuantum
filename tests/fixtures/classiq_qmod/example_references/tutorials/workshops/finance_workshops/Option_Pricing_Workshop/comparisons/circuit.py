"""Test-only unchanged original, native and independent Circuit acceptance."""

from tests.integration.test_synthesis_native_option_pricing import (
    CASES,
    test_unchanged_source_native_and_independent_full_states,
)


def run_example() -> None:
    case = next(
        row
        for row in CASES
        if row[0] == "tutorials.workshops.finance_workshops.option_pricing_workshop"
    )
    for k in (0, 1, 2):
        test_unchanged_source_native_and_independent_full_states(*case, k)


if __name__ == "__main__":
    run_example()
