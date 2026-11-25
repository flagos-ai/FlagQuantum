"""Recover the source hidden shift in all three original-size scenarios."""

from .workflow import Route, run_estimate


def main() -> None:
    """Run actual samples, full-state references and exact GF(2) reconstruction."""
    routes: tuple[Route, ...] = ("simple", "dual", "no_dual")
    for index, route in enumerate(routes):
        result = run_estimate(route, seed=1500 + index)
        print(route, result.recovered)
        print(result.resources)
        print("Full-state maximum absolute error:", result.maximum_state_error)
        print("Workspace nonzero probability:", result.dirty_workspace_probability)
        print("Shots:", result.shots, "Distinct observations:", len(result.counts))


if __name__ == "__main__":
    main()
