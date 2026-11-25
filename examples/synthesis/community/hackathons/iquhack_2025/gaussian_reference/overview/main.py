"""Run the three original Gaussian reference stages with checked cleanup."""

import argparse
from pathlib import Path

import torch

from flagquantum.core.ir import CircuitIR

from .workflow import (
    Amplification,
    exact_grid,
    grid,
    resource_row,
    run_amplification,
    target,
)


def print_stage(archived: Amplification, checked: Amplification) -> None:
    assert archived.width == checked.width
    print(
        f"Width {archived.width} source and lookup overlap:",
        (archived.overlap, archived.actual_overlap),
    )
    print(
        f"Width {archived.width} source leaked probability:",
        archived.workspace_probability,
    )
    print(
        f"Width {archived.width} predicted source leaked probability:",
        archived.predicted_workspace_probability,
    )
    print(
        f"Width {checked.width} checked work probability:",
        checked.workspace_probability,
    )
    print(f"Width {checked.width} checked absolute state error:", checked.state_error)
    print(f"Width {checked.width} checked resources:", resource_row(checked.program))


def save_checked_program(
    result: Amplification, folder: Path, *, team: str | None
) -> Path:
    """Save an optional artifact only after checking amplitude and workspace."""
    if team is None:
        raise ValueError("set a team name to enable native export")
    if result.workspace_probability >= 1e-22:
        raise ValueError("a Gaussian artifact requires verified clean work qubits")
    folder.mkdir(parents=True, exist_ok=True)
    prefix = team.strip().replace(" ", "_")
    if not prefix or not prefix.replace("_", "").isalnum():
        raise ValueError(
            "team name must contain letters, numbers, spaces or underscores"
        )
    path = folder / f"{prefix}__resolution_{result.width}.json"
    path.write_text(result.program.circuit.to_json() + "\n")
    verified = CircuitIR.from_json(path.read_text())
    assert verified.content_hash == result.program.circuit.content_hash
    return path


def plot_eight(result: Amplification, output: Path) -> None:
    import matplotlib.pyplot as plt

    assert result.width == 8
    input_grid = grid(result.width)
    order = input_grid.argsort()
    prepared = result.state.abs().square()[order].tolist()
    theoretical = target(result.width)[order].tolist()
    coordinates = input_grid[order].tolist()
    fig, axis = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
    axis.plot(
        coordinates, theoretical, color="#bf562d", label="Exact target probability"
    )
    axis.scatter(
        coordinates,
        prepared,
        color="#225980",
        s=10,
        alpha=0.65,
        label="Native checked state",
    )
    axis.set(
        xlabel="Signed fixed-point coordinate",
        ylabel="Probability",
        title="Exact Gaussian amplitude amplification",
    )
    axis.legend(frameon=False)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "checked-gaussian.png", dpi=140)
    plt.close(fig)


def stage_one_summary(checked: Amplification) -> None:
    probabilities = checked.state.abs().square()
    probabilities /= probabilities.sum()
    observed = int(
        torch.multinomial(
            probabilities, 1, generator=torch.Generator().manual_seed(17)
        ).item()
    )
    coordinate = float(grid(checked.width)[observed])
    mse = float((probabilities - target(checked.width)).square().mean())
    print("Checked stage-one probability MSE:", mse)
    print("Checked stage-one actual signed sample:", coordinate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures", type=Path, default=Path("gaussian-reference-figures")
    )
    args = parser.parse_args()
    direct, direct_error = exact_grid()
    print("Six-bit direct preparation resources:", resource_row(direct))
    print("Six-bit direct absolute error:", direct_error)
    raw = run_amplification(8, preserve_source_overlap=True)
    checked = run_amplification(8, preserve_source_overlap=False)
    print_stage(raw, checked)
    stage_one_summary(checked)
    plot_eight(checked, args.figures)
    print_stage(
        run_amplification(4, preserve_source_overlap=True),
        run_amplification(4, preserve_source_overlap=False),
    )
    print("Optional team export:", "disabled")


if __name__ == "__main__":
    main()
