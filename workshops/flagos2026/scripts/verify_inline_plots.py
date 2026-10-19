"""Verify inline PNG output with the separately installed workshop teaching tools."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def plotting_setup(path: Path) -> str:
    """Read the actual notebook initialization without running training or cloud jobs."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    plotting = [cell for cell in cells if "import matplotlib.pyplot as plt" in cell]
    if len(plotting) != 1:
        raise ValueError(f"{path.name}: expected exactly one pyplot initialization")
    lines = plotting[0].splitlines()
    magic = "%matplotlib inline"
    pyplot = "import matplotlib.pyplot as plt"
    if (
        magic not in lines
        or pyplot not in lines
        or lines.index(magic) > lines.index(pyplot)
    ):
        raise ValueError(f"{path.name}: initialize inline plotting before pyplot")
    return "\n".join(line for line in lines if line in {magic, pyplot})


def verify_inline_plots(notebooks: Path, kernel: str) -> None:
    """Launch bounded notebook probes and require rich PNG output for all six labs."""
    import nbformat
    from nbclient import NotebookClient

    paths = sorted(notebooks.glob("0[1-6]_*.ipynb"))
    if len(paths) != 6:
        raise ValueError("Expected six plotting showcase notebooks")
    with tempfile.TemporaryDirectory(prefix="flagquantum-inline-") as directory:
        for path in paths:
            notebook = nbformat.v4.new_notebook(
                cells=[
                    nbformat.v4.new_code_cell(plotting_setup(path)),
                    nbformat.v4.new_code_cell("plt.plot([0, 1], [0, 1])\nplt.show()"),
                ]
            )
            NotebookClient(
                notebook,
                kernel_name=kernel,
                timeout=60,
                resources={"metadata": {"path": directory}},
            ).execute(env={**os.environ, "MPLBACKEND": "Agg"})
            if not any(
                "image/png" in output.get("data", {})
                for output in notebook.cells[-1].outputs
            ):
                raise RuntimeError(f"{path.name}: no inline PNG output")
            print(f"PASS {path.parent.name}/{path.name}: inline PNG", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default="python3")
    parser.add_argument("--language", choices=("en", "cn", "both"), default="both")
    args = parser.parse_args()
    directories = {
        "en": ("notebooks",),
        "cn": ("notebooks_cn",),
        "both": ("notebooks", "notebooks_cn"),
    }[args.language]
    for directory in directories:
        verify_inline_plots(Path(__file__).resolve().parents[1] / directory, args.kernel)


if __name__ == "__main__":
    main()
