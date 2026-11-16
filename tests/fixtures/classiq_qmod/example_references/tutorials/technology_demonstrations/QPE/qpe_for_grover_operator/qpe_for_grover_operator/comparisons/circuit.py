"""Test-only original/native/Circuit full-state validation."""

from tests.integration.test_synthesis_native_qpe_examples import (
    CASES,
    test_complete_state_absolute_phase_and_retained_data,
)


def run_example() -> None:
    case = next(
        row
        for row in CASES
        if row[0] == "tutorials.technology_demonstrations.QPE.qpe_for_grover_operator"
    )
    test_complete_state_absolute_phase_and_retained_data(*case)


if __name__ == "__main__":
    run_example()
