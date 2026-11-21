# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 dev_tools/qualtran_dev_tools/shell_tools.py.

"""Logged subprocess calls for internal development commands."""

import subprocess
import sys
from typing import Any

BOLD = 1
DIM = 2
RED = 31
GREEN = 32
YELLOW = 33


def highlight(text: str, color_code: int, bold: bool = False) -> str:
    """Wrap text in the source tool's terminal color sequence."""
    prefix = "\033[1m" if bold else ""
    return f"{prefix}\033[{color_code}m{text}\033[0m"


def abbreviate_command_arguments_after_switches(
    cmd: tuple[str, ...],
) -> tuple[str, ...]:
    """Retain the executable and leading switches in a command log."""
    result = [cmd[0]]
    for argument in cmd[1:]:
        if not argument.startswith("-"):
            result.append("[...]")
            break
        result.append(argument)
    return tuple(result)


def run(
    args: str | list[str],
    *,
    log_run_to_stderr: bool = True,
    abbreviate_non_option_arguments: bool = False,
    check: bool = True,
    text: bool = True,
    **subprocess_run_kwargs: Any,
) -> subprocess.CompletedProcess[Any]:
    """Run a command with checked text output and optional command logging.

    Remaining keywords pass through to subprocess.run, including timeout,
    cwd and capture_output. Commands that can stall must supply a timeout.
    """
    subprocess_run_kwargs.update(check=check, text=text)
    if log_run_to_stderr:
        description = (args,) if isinstance(args, str) else tuple(args)
        if abbreviate_non_option_arguments:
            description = abbreviate_command_arguments_after_switches(description)
        print("run:", description, file=sys.stderr)
    return subprocess.run(args, **subprocess_run_kwargs)


def output_of(args: str | list[str], **kwargs: Any) -> str:
    """Return command stdout with exactly one trailing newline removed."""
    result: str = run(
        args, log_run_to_stderr=False, stdout=subprocess.PIPE, **kwargs
    ).stdout
    return result[:-1] if result.endswith("\n") else result
