"""Plot actual noisy objective measurements without inventing a robot rendering."""

from pathlib import Path

import matplotlib.pyplot as plt

from .workflow import Workflow


def plot_trace(workflow: Workflow, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots()
    axes.plot([value.estimate.cost for value in workflow.trace])
    axes.set_xlabel("Actual objective query")
    axes.set_ylabel("Squared position error (1,000 draws per basis)")
    axes.set_title("Native two-link posture optimization")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path
