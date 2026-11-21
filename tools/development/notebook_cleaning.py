# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 dev_tools/qualtran_dev_tools/clean_notebooks.py.

"""Clear tracked notebook outputs and metadata while retaining generated-cell IDs."""

import os
import subprocess
from importlib import import_module
from pathlib import Path
from tempfile import NamedTemporaryFile

from nbformat import NotebookNode

from tools.development import process

# nbformat exposes untyped optional I/O; validate nodes at this boundary.
nbformat = import_module("nbformat")


def get_nb_rel_paths(rootdir: Path) -> list[Path]:
    """List tracked notebooks relative to a checkout, with a bounded Git query."""
    output = process.output_of(["git", "ls-files", "*.ipynb"], cwd=rootdir, timeout=15)
    paths = [Path(line) for line in output.splitlines()]
    print(paths)
    return paths


def clean_notebook(nb_path: Path, do_clean: bool = True) -> bool:
    """Print a notebook cleanup diff and optionally replace the original file.

    Preserve the source cq.autogen cell identifier used to merge generated
    content. Temporary files live beside the notebook for same-filesystem
    replacement and are removed on every success or failure path.
    """
    preprocessors = import_module("nbconvert.preprocessors")
    with nb_path.open() as stream:
        notebook = nbformat.read(stream, as_version=4)
    if not isinstance(notebook, NotebookNode):
        raise TypeError("notebook reader must return a NotebookNode")
    notebook, resources = preprocessors.ClearOutputPreprocessor().preprocess(
        notebook, resources={}
    )
    notebook, _ = preprocessors.ClearMetadataPreprocessor(
        preserve_cell_metadata_mask={"cq.autogen"}
    ).preprocess(notebook, resources=resources)
    if not isinstance(notebook, NotebookNode):
        raise TypeError("notebook preprocessors must return a NotebookNode")
    with NamedTemporaryFile("w", dir=nb_path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open("w") as stream:
            nbformat.write(notebook, stream, version=4)
        result = subprocess.run(
            ["diff", nb_path, temporary], capture_output=True, timeout=15
        )
        if result.returncode not in (0, 1):
            result.check_returncode()
        dirty = bool(result.stdout)
        print(str(nb_path))
        if dirty:
            print(result.stdout.decode())
        if dirty and do_clean:
            os.replace(temporary, nb_path)
        return dirty
    finally:
        temporary.unlink(missing_ok=True)


def clean_notebooks(sourceroot: Path, do_clean: bool = True) -> int:
    """Clean all tracked notebooks and return the number that needed cleanup."""
    dirty = [
        path
        for path in get_nb_rel_paths(sourceroot)
        if clean_notebook(sourceroot / path, do_clean=do_clean)
    ]
    if dirty:
        print("Dirty notebooks: ")
        for path in dirty:
            print(" ", str(path))
    return len(dirty)
