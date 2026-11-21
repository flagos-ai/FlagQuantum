# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 check/mypy and check/mypy-dev-tools.

"""Run package or developer typing checks under native strict policy."""

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from tools.development import process
from tools.development.repository import get_git_root

Mode = Literal["package", "development"]


def run_typecheck(root: Path, mode: Mode, arguments: Sequence[str]) -> int:
    """Forward mypy arguments and return its exact exit status.

    The existing checkout configuration owns import policy. Strictness and Python
    version follow the package pre-commit gate. Developer modules additionally
    require explicit package bases because tools is a namespace package.
    """
    options = [
        "--config-file=pyproject.toml",
        "--strict",
        "--python-version=3.12",
        "--ignore-missing-imports",
    ]
    if mode == "development":
        options.append("--explicit-package-bases")
    target = "flagquantum/" if mode == "package" else "tools/development/"
    print("\033[31m", end="", flush=True)
    try:
        return int(
            process.run(
                [sys.executable, "-m", "mypy", *options, *arguments, target],
                cwd=root,
                check=False,
                timeout=1200,
                log_run_to_stderr=False,
            ).returncode
        )
    finally:
        print("\033[0m", end="", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] not in {"package", "development"}:
        raise SystemExit("Expected package or development followed by mypy arguments")
    mode = arguments.pop(0)
    if mode == "package":
        result = run_typecheck(get_git_root(), "package", arguments)
    else:
        result = run_typecheck(get_git_root(), "development", arguments)
    raise SystemExit(result)


if __name__ == "__main__":
    main()
