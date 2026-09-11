"""Stable entry points for reproducible benchmark execution.

Runner modules own argument handling, environment capture, and JSON output.
Research/plotting scripts remain outside this namespace and are not imported
as library code.
"""

from .registry import RunnerSpec, describe, names, register, resolve, specs

__all__ = ["RunnerSpec", "describe", "names", "register", "resolve", "specs"]
