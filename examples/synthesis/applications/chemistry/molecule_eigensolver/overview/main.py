"""Report the two ansatz profiles and the measured energy."""

from pathlib import Path

from .workflow import DISTANCE, ITERATIONS, SHOTS, Result, run_workflow

OUTPUT = Path("artifacts/development/molecular-eigensolver")


def report(result: Result) -> None:
    print("Hydrogen distance and basis:", DISTANCE, "sto-3g")
    print("Shots per Pauli basis and iteration ceiling:", SHOTS, ITERATIONS)
    print("Hardware-efficient resources:", result.hardware)
    print("Tapered UCCSD resources:", result.tapered)
    print(
        "Optimizer evaluations, iterations, converged:",
        result.evaluations,
        result.iterations,
        result.converged,
    )
    print("Optimized angle:", result.angle)
    print("Reported finite-shot energy:", result.sampled)
    print("Exact expectation at that angle:", result.exact_expectation)
    print("Full CI energy:", result.fci)


def save_figure(result: Result, output: Path = OUTPUT) -> Path:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4), layout="constrained")
    errors = (
        1_000 * (result.sampled - result.fci),
        1_000 * (result.exact_expectation - result.fci),
    )
    axis.bar(
        ("Finite-shot objective", "Exact expectation"),
        errors,
        color=("#946046", "#49758b"),
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(
        title="Tapered H2 energy relative to full CI",
        ylabel="Energy difference (millihartree)",
    )
    output.mkdir(parents=True, exist_ok=True)
    plot = output / "energy-estimates.png"
    figure.savefig(plot, dpi=135)
    plt.close(figure)
    return plot


def main() -> None:
    result = run_workflow()
    report(result)
    save_figure(result)


if __name__ == "__main__":
    main()
