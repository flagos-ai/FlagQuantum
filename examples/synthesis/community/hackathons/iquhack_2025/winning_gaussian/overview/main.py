"""Execute both original Quantum Tree stages and save native circuit artifacts."""

import argparse
from pathlib import Path

from .workflow import (
    SIMULATED_WIDTH,
    SYNTHESIS_WIDTH,
    Compiled,
    Simulation,
    build,
    save_program,
    simulate,
)


def report(compiled: Compiled, stage: int) -> None:
    cost = compiled.resources
    print(
        f"Stage {stage} compiled width, operations, depth and CX:",
        (cost.n_qubits, cost.n_operations, cost.depth, cost.operation_counts["cx"]),
    )


def report_simulation(result: Simulation) -> None:
    print("Actual signed Gaussian sample:", result.sample)
    print("Mean squared error:", result.mse)
    print("Absolute state error:", result.source_state_error)
    print(
        "Converted phase and aligned error:",
        (result.global_phase, result.converted_state_error),
    )


def plot(result: Simulation, directory: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
    axis.plot(
        result.grid,
        result.target,
        color="#bf562d",
        linewidth=1.6,
        label="Normalized Gaussian target",
    )
    axis.scatter(
        result.grid,
        result.probabilities,
        color="#225980",
        s=8,
        alpha=0.6,
        label="Actual native state",
    )
    axis.set(
        xlabel="Signed fixed-point coordinate",
        ylabel="Probability",
        title="Quantum Tree Gaussian preparation",
    )
    axis.legend(frameon=False)
    axis.text(0.02, 0.96, f"MSE {result.mse:.3e}", transform=axis.transAxes, va="top")
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / "gaussian-comparison.png", dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=Path("gaussian-figures"))
    parser.add_argument("--programs", type=Path, default=Path("gaussian-programs"))
    args = parser.parse_args()
    stage_1 = build(SIMULATED_WIDTH, ceiling=18)
    observed = simulate(stage_1)
    report_simulation(observed)
    report(stage_1, 1)
    plot(observed, args.figures)
    stage_2 = build(SYNTHESIS_WIDTH, ceiling=127)
    report(stage_2, 2)
    saved = (
        save_program(stage_1, args.programs, 1),
        save_program(stage_2, args.programs, 2),
    )
    print("Saved native program artifacts:", tuple(path.name for path in saved))


if __name__ == "__main__":
    main()
