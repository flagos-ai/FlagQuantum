# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 check/mypy and check/mypy-dev-tools.

"""Run native strict typing or an explicit imported-source profile audit."""

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from tools.development import process
from tools.development.repository import get_git_root

Mode = Literal["package", "development", "retained-source"]


def run_typecheck(root: Path, mode: Mode, arguments: Sequence[str]) -> int:
    """Forward mypy arguments and return its exact exit status.

    The existing checkout configuration owns import policy. Strictness and Python
    version follow the package pre-commit gate. Developer modules additionally
    require explicit package bases because tools is a namespace package. The
    retained-source audit uses the complete frozen profile with explicit targets
    and its optional plugin dependencies in the caller environment.
    """
    if mode == "retained-source":
        profile = Path(__file__).with_name("retained_source_mypy.ini")
        options = [f"--config-file={profile}"]
        targets = []
    else:
        options = [
            "--config-file=pyproject.toml",
            "--strict",
            "--python-version=3.12",
            "--ignore-missing-imports",
        ]
        if mode == "development":
            options.append("--explicit-package-bases")
        targets = ["flagquantum/" if mode == "package" else "tools/development/"]
    print("\033[31m", end="", flush=True)
    try:
        return int(
            process.run(
                [sys.executable, "-m", "mypy", *options, *arguments, *targets],
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
    if not arguments or arguments[0] not in {
        "package",
        "development",
        "retained-source",
    }:
        raise SystemExit(
            "Expected package, development or retained-source followed by mypy arguments"
        )
    mode = arguments.pop(0)
    if mode == "package":
        result = run_typecheck(get_git_root(), "package", arguments)
    elif mode == "development":
        result = run_typecheck(get_git_root(), "development", arguments)
    else:
        result = run_typecheck(get_git_root(), "retained-source", arguments)
    raise SystemExit(result)


if __name__ == "__main__":
    main()
