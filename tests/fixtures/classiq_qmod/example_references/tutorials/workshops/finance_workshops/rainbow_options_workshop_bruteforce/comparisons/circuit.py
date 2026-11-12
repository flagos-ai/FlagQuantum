"""Test-only unchanged original, native and independent Circuit acceptance."""

from tests.integration.test_synthesis_native_rainbow import (
    CASES,
    test_original_native_and_independent_full_states,
)


def run_example() -> None:
    case = next(
        row
        for row in CASES
        if row[0]
        == "tutorials.workshops.finance_workshops.rainbow_options_workshop_bruteforce"
    )
    for k in (0, 1, 2):
        test_original_native_and_independent_full_states(*case, k)


if __name__ == "__main__":
    run_example()
