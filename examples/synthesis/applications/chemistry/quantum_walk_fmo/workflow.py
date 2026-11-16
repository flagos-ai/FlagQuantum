"""Run the full local FMO time sweep and compare exact site populations."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

import torch

import flagquantum as fq

from .main import build_program
from .physics import SITE_HAMILTONIAN


def quantum_sweep(times: Sequence[float]) -> tuple[torch.Tensor, tuple[float, ...]]:
    """Execute named site readout; timings include synthesis and execution."""
    populations = []
    durations = []
    for time_point in times:
        start = perf_counter()
        program = build_program(time_point)
        probabilities = dict(program.probabilities(fq.run(program), registers=("qn",)))
        populations.append([probabilities.get((site,), 0.0) for site in range(8)])
        durations.append(perf_counter() - start)
    return torch.tensor(populations, dtype=torch.float64), tuple(durations)


def exact_sweep(times: Sequence[float], initial_node: int = 0) -> torch.Tensor:
    """Use the seven-site dense matrix exponential, independently of synthesis."""
    if type(initial_node) is not int or not 0 <= initial_node < 7:
        raise ValueError("initial_node must be an integer in 0..6")
    hamiltonian = torch.tensor(SITE_HAMILTONIAN, dtype=torch.complex128)
    initial = torch.zeros(7, dtype=torch.complex128)
    initial[initial_node] = 1
    return torch.stack(
        [
            (torch.linalg.matrix_exp(-1j * time * hamiltonian) @ initial).abs().square()
            for time in times
        ]
    )


def save_plot(
    quantum_times: Sequence[float],
    quantum: torch.Tensor,
    exact_times: Sequence[float],
    exact: torch.Tensor,
    output: Path,
) -> None:
    """Plot the same first three sites as the source Notebook."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 5))
    for site, color, marker in zip(
        range(3), ("b", "g", "r"), ("s", "^", "x"), strict=True
    ):
        axis.plot(
            quantum_times,
            quantum[:, site].numpy(),
            color=color,
            marker=marker,
            linestyle="--",
            label=f"Site {site} (FlagQuantum product formula)",
        )
        axis.plot(
            exact_times,
            exact[:, site].numpy(),
            color=color,
            label=f"Site {site} (exact matrix exponential)",
        )
    axis.set(
        title="Continuous-time quantum walk with FMO Hamiltonian",
        xlabel="Time",
        ylabel="Population",
    )
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    axis.grid(True, linestyle="--", alpha=0.7)
    figure.tight_layout()
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)


def run_workflow(output: Path) -> None:
    """Run 20 quantum time points and 100 exact-reference time points."""
    quantum_times = tuple(index * 0.1 / 19 for index in range(20))
    exact_times = tuple(index * 0.1 / 99 for index in range(100))
    quantum, durations = quantum_sweep(quantum_times)
    exact = exact_sweep(exact_times)
    save_plot(quantum_times, quantum, exact_times, exact, output)
    reference = torch.nn.functional.pad(exact_sweep(quantum_times), (0, 1))
    errors = torch.linalg.vector_norm(quantum - reference, dim=1)
    print(f"Saved population comparison to {output}")
    print(
        f"Maximum probability L2 error against exact evolution: {float(errors.max()):.8g}"
    )
    print(f"Maximum padding-state population: {float(quantum[:, 7].max()):.8g}")
    print(f"Synthesis/execution/readout seconds at each quantum time: {durations}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("fmo-populations.png"))
    run_workflow(parser.parse_args().output)
