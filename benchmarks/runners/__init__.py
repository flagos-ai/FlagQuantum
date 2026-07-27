"""Stable entry points for reproducible benchmark execution.

Runner modules own argument handling, environment capture, and JSON output.
Research/plotting scripts remain outside this namespace and are not imported
as library code.
"""

from .registry import names, register, resolve

__all__ = ["names", "register", "resolve"]
