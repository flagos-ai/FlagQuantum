"""Fit and execute the original degree-101 and degree-255 QSVT inverses."""

import argparse
from pathlib import Path

from .workflow import Result, solve


def print_report(label: str, result: Result) -> None:
    print(
        f"{label} degree, data and block widths:",
        result.request_degree,
        result.data_width,
        result.block_width,
    )
    print(
        f"{label} block scale and raw singular cutoff:",
        result.alpha,
        result.minimum_sigma,
    )
    print(
        f"{label} fitted phases and Newton steps:",
        result.phases.phases.numel(),
        result.phases.iterations,
    )
    print(f"{label} dense QSP response error:", result.phases.residual)
    print(
        f"{label} spectral and full-state max errors:",
        result.spectral_error,
        result.full_state_error,
    )
    print(f"{label} direct solution L2 error:", result.solution_l2)
    print(f"{label} projection probability:", result.projection_probability)
    print(
        f"{label} physical bits and represented operations:",
        result.resources.n_qubits,
        result.resources.n_operations,
    )
    print(
        f"{label} build and execution seconds:",
        result.build_seconds,
        result.execute_seconds,
    )


def save_figure(result: Result, title: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(8, 4.5), layout="constrained")
    x = range(result.expected.numel())
    axes.plot(x, result.expected, "o", label="Exact dense solve")
    axes.plot(
        x, result.solution, ".", label=f"Native QSVT, degree {result.request_degree}"
    )
    axes.set(xlabel="Data index", ylabel="Solution", title=title)
    axes.legend()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=Path("nozzle-qsvt-figures"))
    args = parser.parse_args()
    for large, name in ((False, "Small"), (True, "Large")):
        result = solve(large)
        print_report(name, result)
        save_figure(
            result,
            f"{name} nozzle inverse",
            args.figures / f"nozzle-{name.lower()}.png",
        )


if __name__ == "__main__":
    main()
