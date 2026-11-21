# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 dev_tools/clean-stage-2-doc-outputs.py.

"""Remove generated documentation from an explicitly owned output directory."""

import argparse
import shutil
from collections.abc import Sequence
from pathlib import Path

from tools.development.repository import get_git_root


def clean_generated_outputs(output_root: Path) -> None:
    """Remove notebook checkpoints, Markdown, notebooks and generated TOCs.

    The caller owns this generated-only directory. Removal is sequential and
    partial results remain visible when an operation fails, as in the source.
    """
    for checkpoint in output_root.glob("**/.ipynb_checkpoints"):
        assert checkpoint.is_dir(), checkpoint
        print("Removing", checkpoint)
        shutil.rmtree(checkpoint)
    for pattern in ("**/*.md", "**/*.ipynb"):
        for path in output_root.glob(pattern):
            print("Removing", path)
            path.unlink()
    for relative in ("reference/autotoc.rst", "reference/autotoc.rst.inc"):
        toc = output_root / relative
        if toc.exists():
            print("Removing", toc)
            toc.unlink()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_root", type=Path, help="Generated-only output directory"
    )
    args = parser.parse_args(argv)
    root = get_git_root().resolve()
    output = args.output_root.resolve()
    owned = (root / "docs/components", root / "docs/reference/flagquantum")
    if output in root.parents or (output.is_relative_to(root) and output not in owned):
        parser.error(
            "Select a generated-only staging directory outside the checkout, "
            "docs/components, or docs/reference/flagquantum"
        )
    clean_generated_outputs(output)


if __name__ == "__main__":
    main()
