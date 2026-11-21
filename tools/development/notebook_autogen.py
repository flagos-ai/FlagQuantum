# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 dev_tools/qualtran_dev_tools/jupyter_autogen.py.

"""Generate and merge component documentation cells in ordinary notebooks."""

import inspect
import re
import textwrap
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from types import ModuleType

from attrs import field, frozen
from nbformat import NotebookNode

from flagquantum.testing.component_examples import (
    ComponentDocumentation,
    ComponentExample,
)
from tools.development.files import WriteIfDifferent

# nbformat exposes untyped optional I/O; validate nodes at this boundary.
nbformat = import_module("nbformat")

# Retain persisted source notebook identifiers to update existing cells.
_AUTOGEN_KEY = "cq.autogen"
_IMPORTS = "import flagquantum as fq\nimport numpy as np\nimport sympy"


@frozen(kw_only=True)
class NotebookSpec:
    """Describe a module notebook using the existing component metadata.

    Directory defaults to the module's source directory. An explicit path_stem
    overrides the module basename. Diagram code is supplied to the renderer by
    its owning domain because native components have no universal graph method.
    """

    title: str
    module: ModuleType
    component_specs: list[ComponentDocumentation[object]]
    directory: str = field()
    _path_stem: str | None = None

    @directory.default
    def _default_directory(self) -> str:
        path = self.module.__file__
        assert path is not None
        return str(Path(path).parent)

    @property
    def path_stem(self) -> str:
        if self._path_stem is None:
            return self.module.__name__.split(".")[-1]
        return self._path_stem

    @property
    def path(self) -> Path:
        return Path(self.directory) / f"{self.path_stem}.ipynb"


def example_source_lines(example: ComponentExample[object]) -> list[str]:
    """Extract a factory's indented body up to its first top-level return.

    Preserve the original regex/indentation rules and omission of the return
    expression. Examples should assign their named object before returning it.
    """
    source = textwrap.dedent(inspect.getsource(example.factory))
    match = re.match(r"^.*?def .*?\) -> .*?:\n", source, flags=re.MULTILINE | re.DOTALL)
    if match is None:
        raise ValueError(
            f"{example.name} function source was not in the form we expected."
        )
    definition_start, body_start = match.span()
    assert definition_start == 0, "The regex failed to trim off the function definition"
    trimmed = []
    for line in source[body_start:].splitlines():
        if not (line == "" or line.startswith(" " * 4)):
            raise ValueError(f"Bad indentation in {example}: {line}")
        line = line[4:]
        if line.startswith("return "):
            break
        trimmed.append(line)
    return trimmed


def _markdown(source: str, cell_id: str) -> NotebookNode:
    cell = nbformat.v4.new_markdown_cell(source, metadata={_AUTOGEN_KEY: cell_id})
    if not isinstance(cell, NotebookNode):
        raise TypeError("notebook markdown factory must return a NotebookNode")
    return cell


def _code(source: str, cell_id: str) -> NotebookNode:
    cell = nbformat.v4.new_code_cell(source, metadata={_AUTOGEN_KEY: cell_id})
    if not isinstance(cell, NotebookNode):
        raise TypeError("notebook code factory must return a NotebookNode")
    return cell


def get_documentation_cells(
    documentation: ComponentDocumentation[object], prefix: str
) -> list[NotebookNode]:
    """Return class documentation and its native import statement."""
    from tools.development.docstrings import get_markdown_docstring_lines

    return [
        _markdown(
            "\n".join(get_markdown_docstring_lines(documentation.component_type)),
            f"{prefix}.bloq_doc.md",
        ),
        _code(documentation.import_line, f"{prefix}.bloq_doc.py"),
    ]


def get_example_cells(
    documentation: ComponentDocumentation[object], prefix: str
) -> list[NotebookNode]:
    """Return the heading and source body for each lazy example factory."""
    if not documentation.examples:
        return []
    return [
        _markdown("### Example Instances", f"{prefix}.example_instances.md"),
        *[
            _code("\n".join(example_source_lines(example)), f"{prefix}.{example.name}")
            for example in documentation.examples
        ],
    ]


def get_graphical_signature_cells(
    documentation: ComponentDocumentation[object],
    prefix: str,
    code: Callable[[ComponentDocumentation[object]], str],
) -> list[NotebookNode]:
    """Render domain-owned signature code only when examples exist."""
    if not documentation.examples:
        return []
    return [
        _markdown("#### Graphical Signature", f"{prefix}.graphical_signature.md"),
        _code(code(documentation), f"{prefix}.graphical_signature.py"),
    ]


def get_call_graph_cells(
    documentation: ComponentDocumentation[object],
    prefix: str,
    code: Callable[[ComponentExample[object]], str],
) -> list[NotebookNode]:
    """Render domain-owned call graph code for the selected example."""
    if documentation.call_graph_example is None:
        return []
    return [
        _markdown("### Call Graph", f"{prefix}.call_graph.md"),
        _code(code(documentation.call_graph_example), f"{prefix}.call_graph.py"),
    ]


def get_cells(
    documentation: ComponentDocumentation[object],
    *,
    graphical_signature_code: Callable[[ComponentDocumentation[object]], str],
    call_graph_code: Callable[[ComponentExample[object]], str],
) -> list[NotebookNode]:
    """Assemble documentation, examples, signature and call graph in source order."""
    prefix = documentation.component_type.__name__
    return (
        get_documentation_cells(documentation, prefix)
        + get_example_cells(documentation, prefix)
        + get_graphical_signature_cells(documentation, prefix, graphical_signature_code)
        + get_call_graph_cells(documentation, prefix, call_graph_code)
    )


def _get_title_lines(title: str, module: ModuleType) -> list[str]:
    lines = [f"# {title}"]
    if module.__doc__ is not None:
        lines += ["", *inspect.cleandoc(module.__doc__).splitlines()]
    return lines


def _init_notebook(
    path_stem: str, overwrite: bool = False, directory: str = "."
) -> tuple[NotebookNode, Path]:
    path = Path(directory) / f"{path_stem}.ipynb"
    if overwrite:
        path.unlink(missing_ok=True)
    if path.exists():
        print(f"[{path_stem}] Loading existing: {path}")
        with path.open() as stream:
            notebook = nbformat.read(stream, as_version=4)
        if not isinstance(notebook, NotebookNode):
            raise TypeError("notebook reader must return a NotebookNode")
        return notebook, path
    print(f"[{path_stem}] Creating new: {path}")
    notebook = nbformat.v4.new_notebook()
    if not isinstance(notebook, NotebookNode):
        raise TypeError("notebook factory must return a NotebookNode")
    notebook["metadata"].update(
        {
            "kernelspec": {
                "language": "python",
                "name": "python3",
                "display_name": "Python 3",
            },
            "language_info": {"name": "python"},
        }
    )
    return notebook, path


def render_notebook(
    spec: NotebookSpec,
    *,
    graphical_signature_code: Callable[[ComponentDocumentation[object]], str],
    call_graph_code: Callable[[ComponentExample[object]], str],
    imports: str = _IMPORTS,
) -> None:
    """Merge generated cells by their persisted IDs and retain authored content.

    Existing generated cells keep their notebook IDs. Superfluous tagged cells
    are retained, matching the original. Duplicate existing generated tags raise
    ValueError before writing. The existing writer avoids rewriting equal data.
    """
    notebook, path = _init_notebook(spec.path_stem, directory=spec.directory)
    cells = {
        "title_cell": _markdown(
            "\n".join(_get_title_lines(spec.title, spec.module)), "title_cell"
        ),
        "top_imports": _code(imports, "top_imports"),
    }
    for documentation in spec.component_specs:
        cells.update(
            (cell.metadata[_AUTOGEN_KEY], cell)
            for cell in get_cells(
                documentation,
                graphical_signature_code=graphical_signature_code,
                call_graph_code=call_graph_code,
            )
        )
    remaining = list(cells)
    for index, old in enumerate(notebook.cells):
        if _AUTOGEN_KEY not in old.metadata:
            continue
        cell_id = old.metadata[_AUTOGEN_KEY]
        new = cells.get(cell_id)
        if new is None:
            print(f"[{spec.path_stem}] Superfluous {cell_id} cell.")
            continue
        print(f"[{spec.path_stem}] Replacing {cell_id} cell.")
        new.id = old.id
        notebook.cells[index] = new
        remaining.remove(cell_id)
    for cell_id in remaining:
        print(f"[{spec.path_stem}] Adding {cell_id}")
        notebook.cells.append(cells[cell_id])
    with WriteIfDifferent(path) as output:
        nbformat.write(notebook, output)
