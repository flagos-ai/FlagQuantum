"""Run the complete original-size Shor phase estimation and factor recovery."""

from .workflow import run_estimate


def main() -> None:
    """Report actual observations, verified period, factors and resources."""
    result = run_estimate(seed=1700)
    print("Shots:", result.shots)
    print("Retained fractions:", result.fractions)
    print("Recovered order:", result.order)
    print("Factors:", result.factors)
    print("Full-state error:", result.maximum_state_error)
    print("Dirty clean-work probability:", result.dirty_workspace_probability)
    print(result.resources)


if __name__ == "__main__":
    main()
