"""Test-only unchanged source, native and independent walk acceptance."""

from tests.integration.test_synthesis_native_predicate_walks import (
    test_hypercube_unchanged_source_native_and_independent_state,
)


def run_example() -> None:
    for steps in (0, 1, 2, 3):
        test_hypercube_unchanged_source_native_and_independent_state(steps)


if __name__ == "__main__":
    run_example()
