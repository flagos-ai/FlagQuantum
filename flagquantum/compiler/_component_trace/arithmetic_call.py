# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Native identities adapted from the parameter fields of locked arithmetic bloqs.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic

"""Immutable arithmetic resource identities without SDK objects or execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from .symbolic_math import SymbolicInt

ArithmeticOperation: TypeAlias = Literal[
    "add",
    "controlled_add",
    "xor",
    "controlled_xor",
    "bitwise_not",
    "out_of_place_add",
    "constant_add",
    "xor_constant",
    "controlled_constant_add",
    "controlled_xor_constant",
    "multi_controlled_x",
    "controlled_subtraction",
    "less_than",
    "controlled_half",
    "hamming_weight",
]
ArithmeticEncoding: TypeAlias = Literal["unsigned", "twos_complement", "montgomery"]


@dataclass(frozen=True)
class ArithmeticCall:
    """A resource callee's operation and operands, independent of its recipe.

    Width objects remain unchanged, including Sympy integers. Word encodings
    distinguish identities even when cost categories agree. Default encodings
    are unsigned. Controlled arithmetic retains control_value; only the less_than
    placeholder uses signed. Constant operations retain their constant object;
    Add can retain the existing Always wrapper identity. Direction denotes a
    resource adjoint, not an IR.
    """

    operation: ArithmeticOperation
    widths: tuple[SymbolicInt, ...]
    encodings: tuple[ArithmeticEncoding, ...] = ()
    control_value: int = 1
    signed: bool = False
    adjoint: bool = False
    include_most_significant_bit: bool = True
    constant: SymbolicInt | None = None
    always: bool = False
    _python_constant: bool | None = field(init=False, repr=False)
    _python_widths: tuple[bool, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        operations = (
            "add",
            "controlled_add",
            "xor",
            "controlled_xor",
            "bitwise_not",
            "out_of_place_add",
            "constant_add",
            "xor_constant",
            "controlled_constant_add",
            "controlled_xor_constant",
            "multi_controlled_x",
            "controlled_subtraction",
            "less_than",
            "controlled_half",
            "hamming_weight",
        )
        if self.operation not in operations:
            raise ValueError("Unknown arithmetic resource operation")
        arity = 2 if self.operation in ("add", "controlled_add") else 1
        if not isinstance(self.widths, tuple) or len(self.widths) != arity:
            raise ValueError(
                "Arithmetic resource widths must be a tuple matching its arity"
            )
        for width in self.widths:
            if isinstance(width, (bool, float)) or isinstance(width, int) and width < 0:
                raise ValueError(
                    "Arithmetic resource widths require nonnegative integers"
                )
            if type(width) is not int:
                from sympy import Expr

                if not isinstance(width, Expr):
                    raise ValueError(
                        "Arithmetic resource widths require integers or symbolic expressions"
                    )
        # Numeric equality alone would merge Python 3 and Sympy Integer(3),
        # despite their different source decomposition and width-query dispatch.
        object.__setattr__(
            self, "_python_widths", tuple(type(w) is int for w in self.widths)
        )
        if not isinstance(self.encodings, tuple):
            raise ValueError("Arithmetic resource encodings must be an immutable tuple")
        if not self.encodings:
            object.__setattr__(self, "encodings", ("unsigned",) * arity)
        if len(self.encodings) != arity or any(
            encoding not in ("unsigned", "twos_complement", "montgomery")
            for encoding in self.encodings
        ):
            raise ValueError(
                "Arithmetic resource encodings must match the operand widths"
            )
        if self.operation not in (
            "add",
            "controlled_add",
            "xor",
            "controlled_xor",
            "bitwise_not",
            "constant_add",
            "xor_constant",
            "controlled_constant_add",
            "controlled_xor_constant",
        ) and any(encoding != "unsigned" for encoding in self.encodings):
            raise ValueError(
                "Original placeholder signatures use unsigned word encodings"
            )
        if type(self.control_value) is not int or self.control_value not in (0, 1):
            raise ValueError("Arithmetic resource control value must be zero or one")
        if (
            self.operation
            not in (
                "controlled_add",
                "controlled_xor",
                "controlled_constant_add",
                "controlled_xor_constant",
            )
            and self.control_value != 1
        ):
            raise ValueError(
                "Only controlled arithmetic identities accept a control value"
            )
        if type(self.signed) is not bool or (
            self.signed and self.operation != "less_than"
        ):
            raise ValueError(
                "Only less-than placeholder identities accept a signed flag"
            )
        if type(self.adjoint) is not bool:
            raise ValueError("Arithmetic resource direction must be boolean")

        if type(self.include_most_significant_bit) is not bool or (
            not self.include_most_significant_bit
            and self.operation != "out_of_place_add"
        ):
            raise ValueError("Only out-of-place addition accepts a truncated sum")

        has_constant = self.operation in (
            "constant_add",
            "xor_constant",
            "controlled_constant_add",
            "controlled_xor_constant",
        )
        if has_constant != (self.constant is not None):
            raise ValueError("Only constant arithmetic identities require a constant")
        if has_constant and type(self.constant) is not int:
            from sympy import Expr

            if not isinstance(self.constant, Expr):
                raise ValueError(
                    "Arithmetic constants require Python integers or symbolic expressions"
                )
        object.__setattr__(
            self,
            "_python_constant",
            None if not has_constant else type(self.constant) is int,
        )
        if type(self.always) is not bool or self.always and self.operation != "add":
            raise ValueError("The implemented Always resource identity wraps Add")
