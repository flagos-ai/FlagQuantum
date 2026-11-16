"""Test-only native/source-reference Vlasov-Ampere acceptance."""

from tests.integration.test_synthesis_native_vlasov import (
    test_native_vlasov_full_state_absolute_phase_and_clean_workspace,
)


def run_example() -> None:
    test_native_vlasov_full_state_absolute_phase_and_clean_workspace()


if __name__ == "__main__":
    run_example()
