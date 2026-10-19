"""Keep the notebook plotting setup contract in the dependency-light unit lane."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
NOTEBOOKS = Path(__file__).resolve().parents[2] / "workshops/flagos2026/notebooks"
SCRIPT = NOTEBOOKS.parent / "scripts/verify_inline_plots.py"
SPEC = importlib.util.spec_from_file_location("verify_inline_plots", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


@pytest.mark.parametrize(
    "path",
    sorted(NOTEBOOKS.glob("0[1-6]_*.ipynb"))
    + sorted((NOTEBOOKS.parent / "notebooks_cn").glob("0[1-6]_*.ipynb")),
    ids=lambda p: f"{p.parent.name}-{p.stem}",
)
def test_plot_initialization_selects_inline_before_pyplot(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    plotting = [cell for cell in cells if "import matplotlib.pyplot as plt" in cell]
    assert len(plotting) == 1
    initialization = plotting[0]
    assert initialization.index("%matplotlib inline") < initialization.index(
        "import matplotlib.pyplot as plt"
    )
    assert all("plt.show()" in cell for cell in cells if "plt.subplots(" in cell)
    drawings = [cell for cell in cells if '.draw(format="mpl"' in cell]
    assert drawings
    assert all("plt.show()" in cell for cell in drawings)
    if path.name.startswith(("02_", "03_", "04_")):
        assert len(drawings) == 1
    plotting_setup = "\n".join(
        line
        for line in initialization.splitlines()
        if line in {"%matplotlib inline", "import matplotlib.pyplot as plt"}
    )
    assert VERIFIER.plotting_setup(path) == plotting_setup


@pytest.mark.parametrize(
    "source",
    (
        "import matplotlib.pyplot as plt",
        "%matplotlib inline",
        "import matplotlib.pyplot as plt\n%matplotlib inline",
    ),
)
def test_plot_probe_rejects_missing_or_late_initialization(
    tmp_path: Path, source: str
) -> None:
    path = tmp_path / "lesson.ipynb"
    path.write_text(json.dumps({"cells": [{"cell_type": "code", "source": source}]}))
    with pytest.raises(ValueError):
        VERIFIER.plotting_setup(path)
