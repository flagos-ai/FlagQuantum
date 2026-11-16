"""Test-only unchanged source/native/independent acceptance."""

from math import pi

from tests.integration.test_synthesis_native_period_feedback import (
    test_private_feedback_full_purification_and_actual_measurement_density,
)


def run_example() -> None:
    test_private_feedback_full_purification_and_actual_measurement_density(pi / 2, 0.0)


if __name__ == "__main__":
    run_example()
