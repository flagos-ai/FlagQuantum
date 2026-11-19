"""Execute the fresh thermal inversion and native one-phase-bit design example."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_qsvt.overview.phases import (
    response,
)

from . import model
from .workflow import BINDINGS, Result, run


def print_report(result: Result) -> None:
    print("Physical nodes and configurations:", 4, model.CONFIGURATIONS)
    print("Thermal normalization:", model.alpha())
    print(
        "Encoded and conditional block errors:",
        result.matrix_error,
        result.conditional_error,
    )
    print("Signed classical temperatures:", result.reference.tolist())
    print("Signed QSVT temperatures:", result.computed.tolist())
    print(
        "Spectral amplitude error and relative thermal error:",
        result.inverse_error,
        result.relative_error,
    )
    print("Inverse phase count:", result.reciprocal.fit.phases.numel())
    print("Mock bindings:", BINDINGS)
    print("Mock full-state errors:", result.mock_errors)
    print(
        "Mock probability masses:", tuple(result.mock_probabilities.sum(dim=1).tolist())
    )
    print(
        "Inverse, conditional and mock circuit widths:",
        result.inverse_resources.n_qubits,
        result.conditional_resources.n_qubits,
        result.mock_resources.n_qubits,
    )


def save_figures(result: Result, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    x = torch.linspace(result.demo_reciprocal.minimum, 1, 1000, dtype=model.REAL)
    approximation = response(x, result.demo_reciprocal.fit.phases)
    axes[0].plot(x.tolist(), (0.3 / x).tolist(), "--", label="Scaled reciprocal")
    axes[0].plot(x.tolist(), approximation.tolist(), label="Fresh degree-11 response")
    axes[0].set(
        xlabel="Encoded value", ylabel="Response", title="Reciprocal demonstration"
    )
    axes[0].legend()
    nodes = torch.arange(1, 5, dtype=model.REAL)
    axes[1].bar(
        (nodes - 0.18).tolist(),
        result.reference.tolist(),
        width=0.35,
        label="Classical solve",
    )
    axes[1].bar(
        (nodes + 0.18).tolist(), result.computed.tolist(), width=0.35, label="QSVT"
    )
    axes[1].set(
        xlabel="Thermal node",
        ylabel="Temperature offset (K)",
        title="Four-node thermal solution",
    )
    axes[1].set_xticks(range(1, 5))
    axes[1].legend()
    figure.savefig(directory / "thermal-inverse.png", dpi=140)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=Path("cooling-figures"))
    args = parser.parse_args()
    result = run()
    print_report(result)
    save_figures(result, args.figures)


if __name__ == "__main__":
    main()
