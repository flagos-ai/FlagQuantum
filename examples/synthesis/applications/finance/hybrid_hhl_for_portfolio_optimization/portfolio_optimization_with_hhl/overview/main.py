"""Run the hybrid HHL experiment, write the two spectrum figures and print metrics."""

from pathlib import Path

from .model import PHASE_BITS, REDUCED_BITS, SHOTS, Model
from .workflow import Profile, Spectrum, run_experiment


def save_figures(
    model: Model,
    spectrum: Spectrum,
    output: Path = Path("artifacts/development/hybrid-hhl-portfolio"),
) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    first, second = output / "spectrum.png", output / "reduced-spectrum.png"
    probabilities = [spectrum.counts[word] / SHOTS for word in spectrum.words]
    points = spectrum.eigenvalues

    fig, (top, bottom) = plt.subplots(2, figsize=(8, 6), layout="constrained")
    fig.suptitle("Eigenvalues estimated from 2,048 QPE samples")
    top.scatter(points, probabilities, color="firebrick")
    top.set_ylabel("Measured probability")
    bottom.scatter(points, tuple(map(abs, spectrum.projections)), color="navy")
    bottom.set(xlabel="Estimated eigenvalue", ylabel="Inverse-weighted amplitude")
    fig.savefig(first, dpi=140)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4), layout="constrained")
    axis.scatter(points, probabilities, marker="s", label="Four-bit samples")
    step = (1 << REDUCED_BITS) * model.scale
    reduced = [
        (word >> (PHASE_BITS - REDUCED_BITS)) / step - model.eigenvalue_bound
        for word in spectrum.selected_words
    ]
    axis.scatter(
        reduced,
        [spectrum.counts[word] / SHOTS for word in spectrum.selected_words],
        marker="o",
        s=90,
        facecolors="none",
        edgecolors="darkcyan",
        label="Selected at three-bit resolution",
    )
    axis.vlines(
        model.eigenvalues.tolist(),
        0,
        max(probabilities),
        colors="gray",
        linestyles="dotted",
        label="Direct eigensolve",
    )
    axis.set(
        title="Measured spectrum and the two feed-forward bins",
        xlabel="Eigenvalue",
        ylabel="Measured probability",
    )
    axis.legend(loc="upper right")
    fig.savefig(second, dpi=140)
    plt.close(fig)
    return first, second


def report(spectrum: Spectrum, profiles: tuple[Profile, ...]) -> None:
    print("Observed QPE counts by raw word:", spectrum.counts)
    print("Selected four-bit QPE words:", spectrum.selected_words)
    print("Resources in the decomposed CX/U3 circuit")
    for row in profiles:
        stats = row.resources
        print(
            f"{row.name}: {stats.n_qubits} qubits, depth {stats.depth}, "
            f"{stats.operation_counts.get('cx', 0)} CX, "
            f"SWAP overlap {row.overlap}, postselection {row.postselection_rate:.6f}"
        )
        print("  joint counts (00, 01, 10, 11):", row.counts)
        print(
            "  converted state error after global phase alignment:",
            row.projective_error,
        )


def main() -> None:
    model, spectrum, profiles = run_experiment()
    report(spectrum, profiles)
    save_figures(model, spectrum)


if __name__ == "__main__":
    main()
