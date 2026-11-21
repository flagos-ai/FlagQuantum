# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 dev_tools/qualtran_dev_tools/git_tools.py.

"""Resolve the Git root used by a development script."""

import subprocess
from pathlib import Path


def get_git_root() -> Path:
    """Return the existing absolute root of the current Git checkout."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        universal_newlines=True,
        check=True,
        timeout=15,
    )
    path = Path(result.stdout.strip()).absolute()
    assert path.exists()
    return path
