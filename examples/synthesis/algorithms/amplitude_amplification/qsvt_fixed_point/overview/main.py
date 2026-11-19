"""Fit fresh phases and execute the native six-bit arithmetic search."""

from pathlib import Path

from .fitting import fit_source
from .plots import save_polynomial
from .workflow import run_search, show_fit, show_outcome

OUTPUT = Path("artifacts/development/fixed-point-qsvt/polynomial.png")


def main() -> None:
    """Run the full bounded source workflow."""
    fit = fit_source()
    show_fit(fit)
    print(save_polynomial(fit, OUTPUT))
    outcome = run_search(fit)
    show_outcome(outcome)
    if abs(outcome.good - outcome.expected) > 5 * outcome.sigma:
        raise ValueError(
            "The measured count lies outside the source five-sigma acceptance"
        )


if __name__ == "__main__":
    main()
