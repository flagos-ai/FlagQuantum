"""Solve both original nozzle systems with native Chebyshev-LCU circuits."""

import argparse
from pathlib import Path

from .workflow import Result, solve


def print_report(label: str, result: Result) -> None:
    print(
        f"{label} degree, data, matrix and inverse widths:",
        result.degree,
        result.encoding.data_width,
        result.encoding.block_width,
        result.selector_width,
    )
    print(f"{label} block scale:", result.encoding.alpha)
    print(f"{label} inverse cutoff:", result.relative_cutoff)
    print(f"{label} direct solution L2 error:", result.solution_error)
    print(f"{label} projected reference max error:", result.projected_reference_error)
    print(f"{label} full state max error:", result.full_state_error)
    print(f"{label} preparation L2 error:", result.preparation_error)
    print(f"{label} projection probability:", result.projection_probability)
    print(
        f"{label} physical bits and represented operations:",
        result.resources.n_qubits,
        result.resources.n_operations,
    )
    print(
        f"{label} build and execution seconds:",
        result.build_seconds,
        result.execution_seconds,
    )


def plot_small(result: Result, directory: Path) -> None:
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(figsize=(8, 4.5), layout="constrained")
    indices = range(result.solution.numel())
    axes.plot(indices, result.expected, "o", label="Exact dense solve")
    axes.plot(indices, result.solution, ".", label="Pauli encoding, degree 31")
    axes.set(xlabel="Data index", ylabel="Solution", title="Eight-row nozzle")
    axes.legend()
    fig.savefig(directory / "nozzle-small.png", dpi=150)
    plt.close(fig)


def plot_large(exact: Result, approximate: Result, directory: Path) -> None:
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(figsize=(8, 4.5), layout="constrained")
    indices = range(exact.solution.numel())
    axes.plot(indices, exact.expected, "o", label="Exact dense solve")
    axes.plot(indices, exact.solution, ".", label="Exact coefficient preparation")
    axes.plot(
        indices, approximate.solution, "x", label="Bounded approximate preparation"
    )
    axes.set(
        xlabel="Data index", ylabel="Solution", title="Sixteen-row nozzle, degree 255"
    )
    axes.legend()
    fig.savefig(directory / "nozzle-008.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=Path("nozzle-lcu-figures"))
    args = parser.parse_args()
    small = solve(False)
    exact = solve(True)
    approximate = solve(True, 0.03)
    print_report("Small", small)
    print_report("Large exact", exact)
    print_report("Large approximate", approximate)
    plot_small(small, args.figures)
    plot_large(exact, approximate, args.figures)


if __name__ == "__main__":
    main()
