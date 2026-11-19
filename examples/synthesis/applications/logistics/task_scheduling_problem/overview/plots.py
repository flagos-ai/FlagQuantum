"""Show each original task graph, all measured costs, and actual schedules."""

from pathlib import Path

import matplotlib.pyplot as plt

from .workflow import Result

LABELS = (
    "small-input",
    "small-trace",
    "small-costs",
    "small-quantum",
    "small-classical",
    "large-input",
    "large-costs",
    "large-quantum",
    "large-classical",
)


def plan(
    result: Result, matrix: tuple[tuple[int, ...], ...] | None, title: str, path: Path
) -> None:
    model = result.problem
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), layout="constrained")
    times = tuple(range(len(model.capacity)))
    used = (
        tuple(
            sum(a * b for a, b in zip(row, model.workload, strict=True))
            for row in matrix
        )
        if matrix is not None
        else (0,) * len(times)
    )
    axes[0, 0].bar(
        times, model.capacity, label="Available", color="tab:green", alpha=0.55
    )
    if matrix is not None:
        axes[0, 0].bar(times, used, width=0.45, label="Used", color="tab:red")
    axes[0, 0].set(
        title="Capacity", xlabel="Timeslot", ylabel="Resources", xticks=times
    )
    axes[0, 0].legend()
    axes[0, 1].bar(range(model.tasks), model.workload)
    axes[0, 1].set(
        title="Task workloads",
        xlabel="Task",
        ylabel="Resources",
        xticks=range(model.tasks),
    )
    positions = {}
    for task in range(model.tasks):
        assigned = [i for i, row in enumerate(matrix or ()) if row[task]]
        positions[task] = (
            float(assigned[0]) if assigned else float(task),
            float(task % 3),
        )
    for first, second in model.edges:
        axes[1, 0].annotate(
            "",
            positions[second],
            positions[first],
            arrowprops={
                "arrowstyle": "->",
                "color": "0.4",
                "shrinkA": 12,
                "shrinkB": 12,
            },
        )
    for task, (x, y) in positions.items():
        axes[1, 0].scatter(
            x, y, s=320, color="lightgreen", edgecolors="black", zorder=3
        )
        axes[1, 0].text(x, y, str(task), ha="center", va="center", zorder=4)
    axes[1, 0].set(
        title="Original task dependencies", xlabel="Assigned slot / source order"
    )
    axes[1, 0].margins(0.3)
    if matrix is not None:
        axes[1, 1].imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="OrRd")
        axes[1, 1].set(
            title="Actual task assignments",
            xlabel="Task",
            ylabel="Timeslot",
            xticks=range(model.tasks),
            yticks=times,
        )
    else:
        axes[1, 1].axis("off")
    fig.suptitle(title)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save(results: tuple[Result, Result], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for result in results:
        name = result.problem.name
        plan(
            result,
            None,
            f"{name.title()} original scheduling instance",
            directory / f"{name}-input.png",
        )
        plan(
            result,
            result.selected_matrix,
            f"{name.title()} lowest-cost observed schedule • feasible={result.selected_valid}",
            directory / f"{name}-quantum.png",
        )
        plan(
            result,
            result.classical_matrix,
            f"{name.title()} independent classical schedule • finish={result.classical_cost}",
            directory / f"{name}-classical.png",
        )
        fig, ax = plt.subplots(figsize=(7.5, 4.6), layout="constrained")
        # Plot every actually sampled cost with its observed multiplicity.
        from .problem import original_cost

        costs = [original_cost(result.problem, word) for word, _ in result.counts]
        weights = [count for _, count in result.counts]
        ax.hist(costs, bins=30, weights=weights)
        ax.set(
            title=f"{name.title()} observed cost distribution",
            xlabel="Original penalized cost",
            ylabel="Actual shots",
        )
        fig.savefig(directory / f"{name}-costs.png", dpi=140)
        plt.close(fig)
        if name == "small":
            fig, ax = plt.subplots(figsize=(7.5, 4.6), layout="constrained")
            ax.plot(
                range(1, len(result.trace) + 1),
                [row.quantile_cost for row in result.trace],
            )
            ax.set(
                title="Small-instance sampled lower-tail objective",
                xlabel="Actual objective query",
                ylabel="CVaR cost (alpha = 0.3)",
            )
            fig.savefig(directory / "small-trace.png", dpi=140)
            plt.close(fig)
