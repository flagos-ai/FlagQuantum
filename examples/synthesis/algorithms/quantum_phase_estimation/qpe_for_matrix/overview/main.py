"""Run both original-size matrix eigenvalue estimation workflows locally."""

from .workflow import EIGENVALUES, ENERGY_RESOLUTION, run_estimate


def main() -> None:
    """Print sampled estimates, intervals, full-state errors and actual resources."""
    for route in ("exact", "approximate"):
        estimate = run_estimate(route, seed=1400 if route == "exact" else 1401)
        print(route, estimate.resources)
        print("Maximum absolute state error:", estimate.maximum_state_error)
        print("Top two sampled energies:", estimate.solutions)
        print("Intervals:", estimate.intervals)
    print("Energy resolution:", ENERGY_RESOLUTION)
    print("All classical eigenvalues:", EIGENVALUES)


if __name__ == "__main__":
    main()
