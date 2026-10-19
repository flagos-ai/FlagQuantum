"""Launch the image's workshop in authenticated personal shared storage."""

from __future__ import annotations

import sys
from pathlib import Path

from jiuding_paths import personal_directory, prepare_workshop


def arguments(argv: list[str], workshop: Path) -> list[str]:
    """Replace landing settings without changing authentication or proxy settings."""
    landing = {
        "--ServerApp.root_dir",
        "--NotebookApp.notebook_dir",
        "--notebook-dir",
        "--ServerApp.default_url",
        "--LabApp.default_url",
        "--default-url",
    }
    result: list[str] = []
    skip_value = False
    for argument in argv:
        if skip_value:
            skip_value = False
            continue
        if argument.split("=", 1)[0] in landing:
            skip_value = "=" not in argument
        else:
            result.append(argument)
    result.extend(
        [
            f"--ServerApp.root_dir={workshop.parent.parent}",
            "--ServerApp.default_url=/lab/tree/workshop/flagos2026/notebooks/01_first_circuit.ipynb",
        ]
    )
    return result


def main() -> None:
    """Prepare a personal copy before Jupyter opens any editable notebooks."""
    from jupyterlab.labapp import main as launch

    from flagquantum.remote.compute.jiuding import JiudingClient

    template = Path(__file__).resolve().parents[1]
    directory = personal_directory(JiudingClient())
    workshop = prepare_workshop(template, directory)
    print(f"Personal workshop: {workshop}")
    sys.argv[1:] = arguments(sys.argv[1:], workshop)
    launch()


if __name__ == "__main__":
    main()
