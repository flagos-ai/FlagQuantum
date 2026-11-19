"""Run both source models and all three original evolution schedules."""

from pathlib import Path

from .plots import save_comparison, save_spectrum
from .workflow import SCENARIOS, evolve, source_models

ARTIFACTS = Path("artifacts/development/adiabatic-linear-solver")


def main() -> None:
    original, perturbed = source_models()
    print("Condition numbers:", original.condition_number, perturbed.condition_number)
    print(save_spectrum(original, ARTIFACTS / "original-spectrum.png"))
    print(save_spectrum(original, ARTIFACTS / "original-zoom.png", zoom=True))
    print(save_spectrum(perturbed, ARTIFACTS / "perturbed-zoom.png", zoom=True))
    for (name, duration, steps), model in zip(
        SCENARIOS, (original, perturbed, perturbed), strict=True
    ):
        result = evolve(model, duration, steps)
        print(f"{name} schedule:", duration, steps)
        print(f"{name} fidelity:", result.fidelity)
        print(f"{name} full state:", result.state.tolist())
        print(f"{name} native resources:", result.program.report["resources"])
        print(save_comparison(model, result, ARTIFACTS / f"{name}-comparison.png"))


if __name__ == "__main__":
    main()
