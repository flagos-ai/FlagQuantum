"""Execute native Notebook cells with source caching for qfunc inspection."""

import json
import linecache
from pathlib import Path


def execute_code_cells(path: Path) -> tuple[dict[str, object], int]:
    """Run all code cells in a fresh namespace, preserving their source order."""
    notebook = json.loads(path.read_text())
    assert notebook["nbformat"] == 4
    namespace: dict[str, object] = {"__name__": __name__}
    executed = 0
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        filename = f"{path}:cell-{index}"
        assert filename not in linecache.cache
        linecache.cache[filename] = (
            len(source),
            None,
            source.splitlines(keepends=True),
            filename,
        )
        try:
            exec(compile(source, filename, "exec"), namespace)
        finally:
            linecache.cache.pop(filename)
        executed += 1
    return namespace, executed
