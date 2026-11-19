"""Plot the measured sign approximation and the allowed overlap regions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .fitting import DEGREE, MINIMUM, TARGET, PhaseFit, polynomial


def save_polynomial(fitted: PhaseFit, path: Path) -> Path:
    """Show the actual fitted polynomial over the full unitary domain."""
    x = np.linspace(-1, 1, 1001)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(x, polynomial(x, fitted.coefficients), label=f"Degree-{DEGREE} fit")
    axis.plot((MINIMUM, 1), (TARGET, TARGET), "--", color="#666666", label="Target")
    axis.plot((-1, -MINIMUM), (-TARGET, -TARGET), "--", color="#666666")
    axis.axvspan(
        -MINIMUM, MINIMUM, alpha=0.08, color="#666666", label="Unconstrained interval"
    )
    axis.set(
        title="Fixed-point amplification approximation",
        xlabel="Initial overlap",
        ylabel="Signed amplitude",
    )
    axis.legend(loc="upper left")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    return path
