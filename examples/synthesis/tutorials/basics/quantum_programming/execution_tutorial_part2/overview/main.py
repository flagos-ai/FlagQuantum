"""Run the local parameterized execution tutorial and its full sample sweep."""

import argparse
import math
from pathlib import Path

import flagquantum as fq

from . import jobs, workflow


def report_single_counts(circuit: fq.Circuit) -> None:
    print("Single binding counts:", workflow.sample(circuit, math.pi / 2, seed=70))


def report_single_expectation(circuit: fq.Circuit) -> None:
    value, _ = workflow.observe(circuit, math.pi / 5)
    print("Single projector expectation:", value)
    estimate, count = workflow.sample_projector(circuit, math.pi / 5, seed=208)
    print("Single finite projector estimate and shots:", (estimate, count))


def report_sweep(rows: tuple[workflow.Observation, ...]) -> None:
    print(
        "Actual finite projector sweep shots:", sum(row.projector_shots for row in rows)
    )
    print("Actual angle bindings:", len(rows))
    print("Actual finite sweep shots:", sum(sum(row.counts.values()) for row in rows))
    print("Maximum sweep absolute state error:", max(row.state_error for row in rows))
    print(
        "Sweep endpoint projector expectations:",
        (rows[0].expectation, rows[-1].expectation),
    )


def report_jobs(sample_id: str, observe_id: str, directory: Path) -> None:
    print("Local sample and observation job IDs:", (sample_id, observe_id))
    sample = jobs.lookup(directory, sample_id)
    observed = jobs.lookup(directory, observe_id)
    print("Reloaded actual sample counts:", sample.counts)
    print("Reloaded actual projector result:", observed.expectation)
    print("Reloaded exact projector comparison:", observed.exact_expectation)


def report_optimizer(result: workflow.Optimization) -> None:
    print("Actual variational queries:", len(result.trials))
    print("Converged and budget exhausted:", (result.converged, result.budget_stop))
    print("Maximum variational absolute state error:", result.state_error)
    print(
        "Final evaluated fidelity and parameters:",
        (result.trials[-1].fidelity, result.trials[-1].angles),
    )
    best = max(result.trials, key=lambda row: row.fidelity)
    print("Best observed fidelity and parameters:", (best.fidelity, best.angles))


def plot(rows: tuple[workflow.Observation, ...], output: Path) -> None:
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    angles = [row.angle / math.pi for row in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.4), layout="constrained")
    for word, color in (("00", "#225980"), ("11", "#bf562d")):
        ax.plot(
            angles,
            [row.counts.get(word, 0) for row in rows],
            label=f"Observed |{word}⟩",
            color=color,
        )
    ax.set(
        xlabel="Angle (π radians)",
        ylabel="Counts per binding",
        title="Fifty bound-circuit samples",
    )
    ax.legend(frameon=False)
    fig.savefig(output / "angle-counts.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 4.4), layout="constrained")
    ax.plot(
        angles,
        [row.expectation for row in rows],
        color="#225980",
        label="Exact expectation",
    )
    ax.scatter(
        angles,
        [row.sampled_expectation for row in rows],
        color="#bf562d",
        s=11,
        alpha=0.7,
        label="Finite estimate",
    )
    ax.legend(frameon=False)
    ax.set(
        xlabel="Angle (π radians)",
        ylabel="Bell projector expectation",
        title="Bound-circuit Bell-state fidelity",
    )
    fig.savefig(output / "projector-sweep.png", dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=Path, default=Path("execution-figures"))
    parser.add_argument("--jobs", type=Path, default=Path("execution-jobs"))
    args = parser.parse_args()
    template = workflow.template()
    report_single_counts(template)
    report_single_expectation(template)
    rows = workflow.sweep(template)
    report_sweep(rows)
    plot(rows, args.figures)
    ids = jobs.submit_both(template, math.pi / 5, args.jobs)
    report_jobs(*ids, args.jobs)
    report_optimizer(workflow.optimize())


if __name__ == "__main__":
    main()
