"""Execute the original portfolio HHL case on its bundled historical data."""

from pathlib import Path

from .model import ASSETS
from .workflow import Result, run_workflow

OUTPUT = Path("artifacts/development/hhl-portfolio")


def report(result: Result) -> None:
    print("Price rows and asset order:", len(result.model.dates), ASSETS)
    print("Condition and cutoff:", result.model.condition, result.model.cutoff)
    print("Absolute state error:", result.absolute_error)
    print("Postselection probability:", result.postselection_probability)
    print("Source fitted relative error:", result.relative_error)
    print("Direct and HHL weights:", result.weights.tolist())
    print("Normalized final series:", tuple(result.values[-1].tolist()))


def save_figures(result: Result) -> tuple[Path, ...]:
    import matplotlib.pyplot as plt

    OUTPUT.mkdir(parents=True, exist_ok=True)

    figure, axe = plt.subplots(figsize=(8, 3.9))
    for index, asset in enumerate(ASSETS):
        axe.plot(result.model.dates, result.model.prices[:, index], label=asset)
    axe.set(title="Source asset observations", ylabel="Price", xlabel="Date")
    axe.legend(ncols=4)
    figure.autofmt_xdate()
    prices = OUTPUT / "prices.png"
    figure.savefig(prices, dpi=120, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    panels = (
        (result.model.encoded_rhs.reshape(1, -1), "Encoded forcing vector"),
        (result.model.hermitian, "Hermitian embedding"),
        (result.model.phases, "Shifted phase operator"),
    )
    for ax, (matrix, label) in zip(axes, panels, strict=True):
        handle = ax.imshow(matrix, aspect="auto", cmap="coolwarm")
        ax.set(title=label, xlabel="Column", ylabel="Row")
        if matrix.shape[0] == 1:
            ax.set_yticks([0])
        figure.colorbar(handle, ax=ax, fraction=0.046)
    figure.tight_layout()
    system = OUTPUT / "system.png"
    figure.savefig(system, dpi=120, bbox_inches="tight")
    plt.close(figure)

    figure, ax = plt.subplots(figsize=(7, 4))
    positions = range(len(ASSETS))
    ax.bar(
        [index - 0.18 for index in positions],
        result.weights[:, 0],
        width=0.35,
        label="Direct",
    )
    ax.bar(
        [index + 0.18 for index in positions],
        result.weights[:, 1],
        width=0.35,
        label="Native HHL",
    )
    ax.set(
        title="Source portfolio weights",
        xticks=list(positions),
        xticklabels=ASSETS,
        ylabel="Weight",
    )
    ax.legend()
    weights = OUTPUT / "weights.png"
    figure.savefig(weights, dpi=120, bbox_inches="tight")
    plt.close(figure)

    figure, ax = plt.subplots(figsize=(8, 4))
    for index, label in enumerate(("Direct", "Native HHL")):
        ax.plot(result.model.dates, result.values[:, index], label=label)
    ax.set(
        title="Reconstructed portfolio series",
        ylabel="Value relative to first date",
        xlabel="Date",
    )
    ax.legend()
    figure.autofmt_xdate()
    values = OUTPUT / "values.png"
    figure.savefig(values, dpi=120, bbox_inches="tight")
    plt.close(figure)
    return prices, system, weights, values


def main() -> None:
    result = run_workflow()
    report(result)
    for figure in save_figures(result):
        print(figure)


if __name__ == "__main__":
    main()
