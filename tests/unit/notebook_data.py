"""Locate template data owned by the installed notebook conversion distribution."""

import csv
from importlib.metadata import distribution
from io import StringIO
from pathlib import Path, PurePosixPath


def installed_jupyter_data() -> Path:
    """Resolve pip prefix and target layouts without a personal SDK directory."""
    package = distribution("nbconvert")
    record = package.read_text("RECORD")
    if record is None:
        raise ValueError("nbconvert must include its installation RECORD")
    # Distribution.files can filter missing files, hiding incomplete installs.
    roots = {
        Path(str(package.locate_file(row[0]))).parents[3]
        for row in csv.reader(StringIO(record))
        if row
        and PurePosixPath(row[0]).parts[-4:]
        == ("nbconvert", "templates", "markdown", "conf.json")
    }
    if len(roots) != 1:
        raise ValueError("nbconvert must declare exactly one Markdown template root")
    root = roots.pop()
    if not (root / "nbconvert/templates/markdown/conf.json").is_file():
        raise FileNotFoundError("Installed nbconvert Markdown template data is missing")
    return root
