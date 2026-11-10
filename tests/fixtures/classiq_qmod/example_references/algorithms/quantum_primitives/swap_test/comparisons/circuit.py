"""Test-only swap source precision refusal and approved reference validation."""

from tests.integration.test_synthesis_native_swap_environment import test_raw_original_refusal_and_approved_full_environment


def run_example() -> None:
    test_raw_original_refusal_and_approved_full_environment()


if __name__ == "__main__":
    run_example()
