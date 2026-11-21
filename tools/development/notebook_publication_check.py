# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 check/autogenerate-notebooks.

"""Publish registered notebooks and report any uncommitted checkout changes."""

import subprocess
import sys
from pathlib import Path

from tools.development.repository import get_git_root


def check_notebooks(root: Path) -> int:
    """Run publication, then require the entire checkout to be clean.

    Generated documents must already be tracked or deliberately ignored by the
    checkout owner. This check reports new outputs without staging them.
    """
    generated = subprocess.run(
        [sys.executable, "-m", "tools.development.notebook_publication"],
        cwd=root,
        check=False,
        timeout=120,
    )
    if generated.returncode:
        return generated.returncode
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        stdout=subprocess.PIPE,
        text=True,
        check=False,
        timeout=15,
    )
    if status.returncode == 0 and not status.stdout:
        return 0
    print(" ".join(status.stdout.split()))
    return 1


def main() -> None:
    raise SystemExit(check_notebooks(get_git_root()))


if __name__ == "__main__":
    main()
