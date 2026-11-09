# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 quantum/classical integer constructors.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/data_types.py

"""Construct native register metadata using the pinned type-specific rules.

Proposal 103 prerequisites 0.3 through 0.6 separate constructor validation from numerical
codecs, physical wire binding and executable component graph admission.
"""

from __future__ import annotations

from enum import Enum, auto

from ...value_math import SymbolicInt as RegisterWidth
from ...value_parameters import is_symbolic_value
from .registers import RegisterType, TypeParameter


class _DefaultBound(Enum):
    OMITTED = auto()


def quantum_bit_type() -> RegisterType:
    """Return the single-qubit encoding."""
    return RegisterType("bit", 1)


def opaque_register_type(width: RegisterWidth) -> RegisterType:
    """Preserve symbolic-first validation of an opaque quantum word."""
    if not is_symbolic_value(width) and not isinstance(width, int):
        raise ValueError(f"Bad bitsize for opaque register: {width}")
    return RegisterType("opaque", width)


def unsigned_register_type(width: RegisterWidth) -> RegisterType:
    """Retain the declared unsigned width without invoking its numerical codec."""
    return RegisterType("uint", width)


def bounded_unsigned_register_type(
    width: RegisterWidth,
    bound: RegisterWidth | _DefaultBound = _DefaultBound.OMITTED,
) -> RegisterType:
    """Construct an unsigned word with an exclusive iteration bound.

    An omitted bound is computed before symbolic classification. Numerical value
    validation and physical binding apply at their existing owning boundaries.
    """
    resolved = _resolve_unsigned_bound(
        width,
        bound,
        description="Bounded unsigned register",
        bound_name="iteration length",
    )
    return RegisterType("uint", width, parameters=(("bound", resolved),))


def _resolve_unsigned_bound(
    width: RegisterWidth,
    bound: RegisterWidth | _DefaultBound,
    *,
    description: str,
    bound_name: str,
) -> RegisterWidth:
    if bound is _DefaultBound.OMITTED:
        bound = 2**width
    if not is_symbolic_value(width, bound):
        if bound > 2**width:
            raise ValueError(
                f"{description}({width}, {bound}) {bound_name} "
                f"is too large for given bitsize. {bound} vs {2**width}"
            )
    return bound


def validate_ones_complement_width(width: RegisterWidth) -> None:
    """Retain the shared original constructor guard without broadening it."""
    # Original constructors reject only a Python-int one, not all widths <=1.
    if isinstance(width, int) and width == 1:
        raise ValueError("bitsize must be > 1.")


def signed_register_type(width: RegisterWidth) -> RegisterType:
    """Retain a two's-complement quantum word's declared width."""
    return RegisterType("signed", width)


def ones_complement_register_type(width: RegisterWidth) -> RegisterType:
    """Construct a quantum ones'-complement word with the original width guard."""
    validate_ones_complement_width(width)
    return RegisterType("ones_complement", width)


def classical_bit_type() -> RegisterType:
    """Return the existing single-classical-bit identity."""
    return RegisterType("bit", 0, 1)


def classical_unsigned_register_type(width: RegisterWidth) -> RegisterType:
    """Retain classical storage identity even when the declared width is zero."""
    return RegisterType("uint", 0, width, parameters=(("storage", "classical"),))


def classical_signed_register_type(width: RegisterWidth) -> RegisterType:
    """Construct a classical two's-complement word without numerical validation."""
    return RegisterType("signed", 0, width, parameters=(("storage", "classical"),))


def classical_ones_complement_register_type(width: RegisterWidth) -> RegisterType:
    """Preserve the shared ones'-complement guard and classical storage identity."""
    validate_ones_complement_width(width)
    return RegisterType(
        "ones_complement", 0, width, parameters=(("storage", "classical"),)
    )


def classical_bounded_unsigned_register_type(
    width: RegisterWidth,
    bound: RegisterWidth | _DefaultBound = _DefaultBound.OMITTED,
) -> RegisterType:
    """Construct a bounded classical word with source default and failure order."""
    resolved = _resolve_unsigned_bound(
        width,
        bound,
        description="Bounded classical unsigned register",
        bound_name="bound",
    )
    return RegisterType(
        "uint", 0, width, parameters=(("storage", "classical"), ("bound", resolved))
    )


def montgomery_register_type(
    width: RegisterWidth, modulus: RegisterWidth | None = None
) -> RegisterType:
    """Retain declared quantum Montgomery fields without numerical validation."""
    parameters = () if modulus is None else (("modulus", modulus),)
    return RegisterType("montgomery", width, parameters=parameters)


def classical_montgomery_register_type(
    width: RegisterWidth, modulus: RegisterWidth | None = None
) -> RegisterType:
    """Retain classical Montgomery storage identity and an optional modulus."""
    parameters: tuple[tuple[str, TypeParameter], ...] = (("storage", "classical"),)
    if modulus is not None:
        parameters += (("modulus", modulus),)
    return RegisterType("montgomery", 0, width, parameters=parameters)


def validate_fixed_point_parameters(
    width: RegisterWidth,
    fractional_width: RegisterWidth,
    *,
    signed: bool = False,
    width_label: str = "bitsize",
    encoding_label: str = "Fxp",
) -> None:
    """Preserve source shape checks with the constructor's diagnostic context."""
    if not is_symbolic_value(width) and width == 1 and signed:
        raise ValueError(f"{width_label} must be > 1.")
    if not is_symbolic_value(width) and not is_symbolic_value(fractional_width):
        if signed and width == fractional_width:
            raise ValueError(
                f"num_frac must be less than bitsize if the {encoding_label} is signed."
            )
        if width < fractional_width:
            raise ValueError("bitsize must be >= num_frac.")


def fixed_point_register_type(
    width: RegisterWidth, fractional_width: RegisterWidth, signed: bool = False
) -> RegisterType:
    """Construct quantum fixed-point metadata with the source eager shape checks."""
    validate_fixed_point_parameters(
        width,
        fractional_width,
        signed=signed,
        width_label="num_qubits",
        encoding_label="fixed-point register",
    )
    return RegisterType(
        "fixed_point",
        width,
        parameters=(("fractional_width", fractional_width), ("signed", signed)),
    )


def classical_fixed_point_register_type(
    width: RegisterWidth, fractional_width: RegisterWidth, signed: bool = False
) -> RegisterType:
    """Preserve classical storage identity and the source fixed-point checks."""
    validate_fixed_point_parameters(
        width,
        fractional_width,
        signed=signed,
        width_label="num_qubits",
        encoding_label="fixed-point register",
    )
    return RegisterType(
        "fixed_point",
        0,
        width,
        parameters=(
            ("storage", "classical"),
            ("fractional_width", fractional_width),
            ("signed", signed),
        ),
    )
