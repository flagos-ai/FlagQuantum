"""Plot the full image and the actual finite-shot edge results."""

from pathlib import Path

import matplotlib.pyplot as plt

from .workflow import THRESHOLD, Result


def image_panels(result: Result, path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(9, 8), layout="constrained")
    for axis, image, title in (
        (axes[0, 0], result.photo, "Original RGB image"),
        (axes[0, 1], result.preview, "Posterized preview"),
        (axes[1, 0], result.exact_edge.T, "Exact edge magnitudes"),
        (axes[1, 1], result.thresholded_edge.T, "Sampled edges after threshold"),
    ):
        axis.imshow(image, cmap="gray", interpolation="nearest")
        axis.set_title(title)
        axis.set_axis_off()
    figure.suptitle("Hadamard edge detection at 128 × 128")
    figure.savefig(path, dpi=140)
    plt.close(figure)


def histograms(result: Result, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for axis, image, title in (
        (axes[0], result.sampled_edge, "Sampled magnitudes"),
        (axes[1], result.thresholded_edge, "After threshold"),
    ):
        axis.hist(image.flatten().tolist(), bins=45, color="#256d85", log=True)
        axis.axvline(
            THRESHOLD, color="#ae4b41", linestyle="--", label="Threshold 0.003"
        )
        axis.set_title(title)
        axis.set_xlabel("Square root of joint pixel and edge probability")
        axis.set_ylabel("Pixel count (log scale)")
        axis.legend()
    figure.savefig(path, dpi=140)
    plt.close(figure)
