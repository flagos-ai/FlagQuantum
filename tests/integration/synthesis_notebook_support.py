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
    cached = []
    try:
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
            cached.append(filename)
            # Factories in earlier cells may capture qfunc source in later cells.
            exec(compile(source, filename, "exec"), namespace)
            executed += 1
    finally:
        for filename in cached:
            linecache.cache.pop(filename)
    return namespace, executed
