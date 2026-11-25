"""Run the Workshop's original width28 and final tolerance factor10."""

from examples.synthesis.applications.finance.option_pricing.overview.main import (
    run_example as run_pricing_example,
)


def run_example() -> None:
    run_pricing_example(max_width=28, seed=6101, source_tolerance_factor=10)


if __name__ == "__main__":
    run_example()
