"""Execute the original-size native bars-and-stripes GAN tutorial."""

import argparse
from pathlib import Path

import torch

from .workflow import (
    ClassicalRun,
    QuantumRun,
    Training,
    run_classical,
    run_quantum,
)


def plot_patterns(images: torch.Tensor, title: str, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig, axes = plt.subplots(3, 3, figsize=(5, 5), layout="constrained")
    palette = ListedColormap(["#f1eee5", "#145960"])
    for index, axis in enumerate(axes.flat):
        pixels = images[index].reshape(2, 2)
        axis.imshow(pixels.numpy(), cmap=palette, vmin=0, vmax=1)
        axis.set_title(str(index + 1), fontsize=9)
        axis.set_axis_off()
        for row in range(2):
            for column in range(2):
                value = int(pixels[row, column].item())
                axis.text(
                    column,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value else "#145960",
                )
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def report_training(name: str, training: Training) -> None:
    print(f"{name} minibatches:", len(training.losses))
    print(f"{name} first and final BCE:", (training.losses[0], training.losses[-1]))
    print(
        f"{name} changed tensors:",
        (training.changed_generator_tensors, training.changed_discriminator_tensors),
    )
    print(f"{name} maximum generator gradient:", training.maximum_generator_gradient)


def report_classical(classical: ClassicalRun) -> None:
    report_training("Classical", classical.training)
    print("Classical 100-sample scores:", classical.scores)


def report_quantum(quantum: QuantumRun) -> None:
    report_training("Hybrid", quantum.training)
    print("Hybrid 10-sample scores:", quantum.scores)
    print("Hybrid quantum query rows:", quantum.query_rows)
    print("Hybrid finite circuit shots:", quantum.finite_draws)
    print("Independent maximum state error:", quantum.maximum_state_error)
    print("Hybrid generated rows:", quantum.examples.to(torch.int64).tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=Path("resources"))
    parser.add_argument("--output", type=Path, default=Path("qgan-output"))
    parser.add_argument("--figures", type=Path)
    args = parser.parse_args()
    figures = args.figures if args.figures is not None else args.output
    classical = run_classical(args.assets, args.output)
    report_classical(classical)
    for label, images, title in (
        (
            "training-examples",
            classical.initial_examples,
            "Nine generated training patterns",
        ),
        (
            "classical-samples",
            classical.generated_examples,
            "Classical generator samples",
        ),
        (
            "fresh-comparison",
            classical.fresh_comparison_examples,
            "Fresh comparison patterns",
        ),
    ):
        plot_patterns(images, title, figures / f"{label}.png")
    report_quantum(run_quantum(args.assets, args.output))


if __name__ == "__main__":
    main()
