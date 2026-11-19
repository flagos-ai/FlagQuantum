"""Run the native Fourier and Gibbs examples with the recorded shot budgets."""

from pathlib import Path

from .model import BETA, DELTA, ROUNDS
from .plots import save_gibbs, save_spectra, save_window
from .workflow import run_workflow

FOLDER = Path("artifacts/development/thermal-state")


def main() -> None:
    result = run_workflow()
    print("Beta, delta and rounds:", BETA, DELTA, ROUNDS)
    print(
        "Actual Gaussian, uniform and thermal shots:",
        sum(result.gaussian.counts),
        sum(result.uniform.counts),
        sum(result.thermal.counts),
    )
    print("Thermal counts:", result.thermal.counts)
    print("Gibbs reference:", result.exact_gibbs)
    print("Thermal resources:", result.thermal_program.report["resources"])
    print(save_window(FOLDER / "window.png"))
    print(save_spectra(result, FOLDER / "spectra.png"))
    print(save_gibbs(result, FOLDER / "gibbs.png"))


if __name__ == "__main__":
    main()
