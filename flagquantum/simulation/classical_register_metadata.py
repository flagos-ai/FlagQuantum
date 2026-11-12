"""Shared register metadata for numerical values and concrete bit codecs."""

from __future__ import annotations

from typing import cast

from ..core.ir.program.registers import Register
from ..core.value_math import SymbolicInt


def _register_parameters(
    register: Register, *, symbolic: bool
) -> dict[str, SymbolicInt]:
    parameters: dict[str, SymbolicInt] = {}
    for name, value in register.dtype.parameters:
        if name == "storage" and value == "classical":
            continue
        allowed = isinstance(value, int)
        if symbolic and not allowed:
            from sympy import Expr

            allowed = isinstance(value, Expr)
        if (
            name not in {"bound", "modulus", "fractional_width", "signed"}
            or not allowed
        ):
            raise ValueError(
                "register codec parameters must be concrete native metadata"
                if not symbolic
                else "register value parameters must be integer or symbolic metadata"
            )
        parameters[name] = cast(SymbolicInt, value)
    return parameters


def concrete_register_metadata(register: Register) -> tuple[int, dict[str, int | bool]]:
    """Read finite codec metadata, retaining the existing refusal diagnostics."""
    width = register.bitsize
    if type(width) is not int or width < 0:
        raise ValueError("register codecs require a concrete nonnegative bit width")
    return width, cast(
        dict[str, int | bool], _register_parameters(register, symbolic=False)
    )


def value_register_metadata(
    register: Register,
) -> tuple[SymbolicInt, dict[str, SymbolicInt]]:
    """Read numerical metadata without evaluating symbolic widths or parameters.

    The dtype-specific numerical owner decides which symbolic operations work.
    Unlike a binary codec, value binding need not evaluate a width, and scalar
    collection returns the original object before choosing array storage.
    """
    width = register.bitsize
    if type(width) is int:
        if width < 0:
            raise ValueError(
                "register values require nonnegative or symbolic bit widths"
            )
    else:
        from sympy import Expr

        if not isinstance(width, Expr):
            raise ValueError(
                "register values require nonnegative or symbolic bit widths"
            )
    return width, _register_parameters(register, symbolic=True)
