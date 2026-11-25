"""Test-only original/native/Circuit full-state validation."""

from tests.integration.test_synthesis_native_qpe_examples import (
    CASES,
    test_complete_state_absolute_phase_and_retained_data,
)


def run_example() -> None:
    case = next(
        row
        for row in CASES
        if row[0]
        == "algorithms.amplitude_amplification_and_estimation.quantum_counting.quantum_counting_qpe"
    )
    test_complete_state_absolute_phase_and_retained_data(*case)


if __name__ == "__main__":
    run_example()
