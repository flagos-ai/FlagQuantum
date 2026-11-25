"""Run both native H2 molecular walks and plot postselected QPE energies."""

import argparse
from pathlib import Path

from .workflow import SHOTS, TARGET_BASIS, Method, Result, run_workflow


def print_method(label: str, method: Method) -> None:
    print(f"{label} block-zero draws:", method.selected_count)
    print(f"{label} block-zero fraction:", method.selected_count / SHOTS)
    print(f"{label} sampled energy:", method.estimate)
    print(f"{label} target depth:", method.resources.depth)
    print(f"{label} target operations:", method.resources.n_operations)
    print(f"{label} target width:", method.resources.n_qubits)


def print_report(result: Result) -> None:
    print("H2 full CI:", result.model.full_ci)
    print("LCU scale:", result.model.normalization)
    print("Fresh Pauli terms:", len(result.model.coefficients))
    print("Reference coefficient delta:", result.model.max_reference_delta)
    print("Selector and phase widths:", 4, 5)
    print("Target basis:", TARGET_BASIS)
    print("Draws per method:", SHOTS)
    print_method("Naive", result.naive)
    print_method("Optimized", result.optimized)
    print("Original error bound:", result.error_tolerance)


def save_figure(method: Method, full_ci: float, name: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(8, 4.5), layout="constrained")
    observed = [
        (energy, count / SHOTS)
        for energy, count in zip(method.energies, method.block_zero_counts, strict=True)
        if count
    ]
    energies, probabilities = zip(*observed, strict=True)
    axes.scatter(energies, probabilities, label="Block-zero observations")
    axes.axvline(full_ci, color="tab:red", linestyle="--", label="Full CI energy")
    axes.set(
        xlabel="Energy (Ha)",
        ylabel="Joint observed probability",
        title=f"H₂ molecular QPE, {name}",
    )
    axes.legend()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=Path("h2-qubitization-figures"))
    args = parser.parse_args()
    result = run_workflow()
    print_report(result)
    save_figure(
        result.naive,
        result.model.full_ci,
        "controlled powers",
        args.figures / "naive-histogram.png",
    )
    save_figure(
        result.optimized,
        result.model.full_ci,
        "conjugated powers",
        args.figures / "optimized-histogram.png",
    )


if __name__ == "__main__":
    main()
