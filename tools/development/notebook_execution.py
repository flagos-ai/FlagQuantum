# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 dev_tools/qualtran_dev_tools/notebook_execution.py.

"""Execute tracked documentation notebooks and export their accepted outputs."""

import multiprocessing
import os
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

from nbformat import NotebookNode

from flagquantum.testing.notebook_execution import execute_notebook_node
from tools.development import process
from tools.development.repository import get_git_root

nbformat = import_module("nbformat")


def get_nb_rel_paths(sourceroot: Path) -> list[tuple[Path, Path]]:
    """List tracked notebook paths paired with their actual source root."""
    output = process.output_of(
        ["git", "ls-files", "*.ipynb"], cwd=sourceroot, timeout=15
    )
    return [(Path(line), sourceroot) for line in output.splitlines()]


def is_out_of_date(source_path: Path, out_path: Path) -> bool:
    """Return whether an output is missing or older than its existing source."""
    assert source_path.exists()
    return (
        not out_path.exists() or source_path.stat().st_mtime > out_path.stat().st_mtime
    )


@dataclass(frozen=True)
class NotebookExportPaths:
    nb_in: Path
    md_out: Path | None
    nb_out: Path | None

    @classmethod
    def from_nb_rel_path(
        cls,
        nb_rel_path: Path,
        reporoot: Path,
        sourceroot: Path,
        output_md: bool,
        output_nbs: bool,
    ) -> "NotebookExportPaths":
        return cls(
            sourceroot / nb_rel_path,
            reporoot / "docs" / nb_rel_path.with_suffix(".md") if output_md else None,
            reporoot / "docs" / nb_rel_path if output_nbs else None,
        )

    def md_needs_reexport(self) -> bool:
        return self.md_out is not None and is_out_of_date(self.nb_in, self.md_out)

    def nb_needs_reexport(self) -> bool:
        return self.nb_out is not None and is_out_of_date(self.nb_in, self.nb_out)

    def needs_reexport(self) -> bool:
        return self.md_needs_reexport() or self.nb_needs_reexport()


def _make_link_replacements() -> list[tuple[str, str]]:
    names = {
        "Circuit": "/reference/API.md",
        "ProgramRecord": "/development/REGISTER_COMPONENT_GRAPH_GUIDE.md",
        "ComponentProgramBuilder": "/development/REGISTER_COMPONENT_GRAPH_GUIDE.md",
        "Register": "/development/REGISTER_COMPONENT_GRAPH_GUIDE.md",
        "Signature": "/development/REGISTER_COMPONENT_GRAPH_GUIDE.md",
        "RegisterSide": "/development/REGISTER_COMPONENT_GRAPH_GUIDE.md",
        "RegisterBinding": "/development/REGISTER_COMPONENT_GRAPH_GUIDE.md",
        "ValueRef": "/development/REGISTER_COMPONENT_GRAPH_GUIDE.md",
    }
    return [(f"`{name}`", f"[`{name}`]({path})") for name, path in names.items()]


_LINK_REPLACEMENTS = _make_link_replacements()


def linkify(
    notebook: NotebookNode,
    replacements: Sequence[tuple[str, str]] = _LINK_REPLACEMENTS,
) -> None:
    """Apply the original ordered substring replacements to Markdown cells only."""
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            for old, new in replacements:
                cell.source = cell.source.replace(old, new)


def _write_export(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def execute_and_export_notebook(
    paths: NotebookExportPaths,
    *,
    replacements: Sequence[tuple[str, str]] = _LINK_REPLACEMENTS,
    startup_lock: Path = Path("jupyter_kernel.lock"),
) -> Exception | None:
    """Execute one notebook, returning execution errors and exporting successes.

    Input/format and export errors propagate as in the original. Execution and
    kernel startup errors return as task outcomes. Successful exports replace
    each destination atomically. Markdown-linked resources retain the original
    exporter behavior, including its separately returned image resources.
    """
    with paths.nb_in.open() as stream:
        notebook = nbformat.read(stream, as_version=4)
    if not isinstance(notebook, NotebookNode):
        raise TypeError("notebook reader must return a NotebookNode")
    linkify(notebook, replacements)
    print(f"Executing {paths.nb_in.stem}")
    try:
        notebook, resources = execute_notebook_node(
            notebook, record_timing=False, startup_lock=startup_lock
        )
    except Exception as error:
        print(f"{paths.nb_in} failed!")
        print(error)
        return error
    # Render before touching either destination. A renderer failure cannot
    # publish a new notebook alongside a partially generated Markdown file.
    notebook_text = None
    if paths.nb_out:
        serialized = nbformat.writes(notebook, version=4)
        if not isinstance(serialized, str):
            raise TypeError("notebook serialization must return text")
        notebook_text = serialized if serialized.endswith("\n") else serialized + "\n"
    markdown = None
    if paths.md_out:
        nbconvert = import_module("nbconvert")
        markdown, _ = nbconvert.export(
            nbconvert.MarkdownExporter(), notebook, resources=resources
        )
        if not isinstance(markdown, str):
            raise TypeError("notebook Markdown export must return text")
    if paths.nb_out and notebook_text is not None:
        _write_export(paths.nb_out, notebook_text)
    if paths.md_out and markdown is not None:
        _write_export(paths.md_out, markdown)
    return None


@dataclass(frozen=True)
class NotebookRunResult:
    nb_in: Path
    err: Exception | None
    duration_s: float


@dataclass(frozen=True)
class _NotebookRunClosure:
    reporoot: Path
    output_nbs: bool
    output_md: bool
    only_out_of_date: bool

    def __call__(self, nb_rel_path: Path, sourceroot: Path) -> NotebookRunResult:
        paths = NotebookExportPaths.from_nb_rel_path(
            nb_rel_path, self.reporoot, sourceroot, self.output_md, self.output_nbs
        )
        if self.only_out_of_date and not paths.needs_reexport():
            print(f"{nb_rel_path} up to date")
            return NotebookRunResult(paths.nb_in, None, 0.0)
        start = time.time()
        error = execute_and_export_notebook(
            paths, startup_lock=self.reporoot / "jupyter_kernel.lock"
        )
        duration = time.time() - start
        print(f"Exported {nb_rel_path} in {duration:.2f} seconds.")
        return NotebookRunResult(paths.nb_in, error, duration)


def execute_and_export_notebooks(
    *,
    output_nbs: bool,
    output_md: bool,
    only_out_of_date: bool = True,
    n_workers: int | None = None,
    reporoot: Path | None = None,
    sourceroots: Sequence[Path] | None = None,
    start_method: Literal["spawn", "fork", "forkserver"] | None = None,
) -> None:
    """Execute tracked notebooks serially or in one-task-per-child processes.

    Native defaults scan the package and examples/notebooks. Explicit roots
    support focused documentation checkouts. Error outcomes produce exit status
    1 and successful runs print the ten longest durations, matching the source.
    Empty source sets succeed in either mode. Guard process-launching scripts
    with their ordinary __main__ entry when using platform-default spawn.
    """
    root = get_git_root() if reporoot is None else reporoot
    roots = (
        (root / "flagquantum", root / "examples" / "notebooks")
        if sourceroots is None
        else sourceroots
    )
    paths = [pair for source in roots for pair in get_nb_rel_paths(source)]
    random.shuffle(paths)
    print(f"Found {len(paths)} notebooks.")
    run = _NotebookRunClosure(root, output_nbs, output_md, only_out_of_date)
    if not paths:
        print("Slowest 10 notebooks:")
        return
    if n_workers == 1:
        print("(Not using multiprocessing, n_workers=1)")
        results = [run(path, source) for path, source in paths]
    else:
        print(f"Multiprocessing with n_workers={n_workers}")
        with multiprocessing.get_context(start_method).Pool(
            n_workers, maxtasksperchild=1
        ) as pool:
            results = pool.starmap(run, paths)
    bad = [result.nb_in for result in results if result.err is not None]
    if bad:
        print("\nErrors in notebooks:")
        for path in bad:
            print(" ", path)
        sys.exit(1)
    print("Slowest 10 notebooks:")
    for result in sorted(results, key=lambda result: result.duration_s, reverse=True)[
        :10
    ]:
        print(f"{result.duration_s:5.2f}s {result.nb_in}")
