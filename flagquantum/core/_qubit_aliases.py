"""Compatibility for the staged public qubit-keyword migration."""

import warnings
from enum import Enum


class Omitted(Enum):
    VALUE = "omitted"

    def __repr__(self) -> str:
        return "OMITTED"


OMITTED = Omitted.VALUE


def warn_qubit_alias(old: str, new: str, *, stacklevel: int = 3) -> None:
    warnings.warn(
        f"{old} is deprecated; use {new}. Removal is planned for 0.4.0 "
        "after the 0.3.x migration window.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
