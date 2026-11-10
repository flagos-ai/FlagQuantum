"""Test-only untouched matrix-QPE reference validation."""

from tests.integration.test_synthesis_native_matrix_qpe import test_original_native_and_independent_complete_environment


def run_example() -> None:
    test_original_native_and_independent_complete_environment('algorithms/quantum_phase_estimation/qpe_for_matrix')


if __name__ == "__main__":
    run_example()
