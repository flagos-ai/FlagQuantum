# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 check_dtypes_consistent and _check_uint_fxp_consistent.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/data_types.py

"""Source-ordered consistency checks over neutral register declarations."""

from enum import Enum
from typing import cast

from ...value_math import SymbolicInt
from .register_types import validate_fixed_point_parameters
from .registers import RegisterType


class RegisterTypeCheckingSeverity(Enum):
    """Preserve the source's loose, opaque-permitting and strict decisions."""

    LOOSE = 0
    ANY = 1
    STRICT = 2


def _same_declared_type(left: RegisterType, right: RegisterType) -> bool:
    if left.encoding == right.encoding == "finite_field":
        first, second = dict(left.parameters), dict(right.parameters)
        same_declaration = (
            first.get("storage") == second.get("storage")
            and first["characteristic"] == second["characteristic"]
            and first["degree"] == second["degree"]
        )
        if same_declaration and (("polynomial" in first) != ("polynomial" in second)):
            # Pinned Poly.Like rejects None during source nominal equality.
            raise TypeError(
                "A 'poly-like' object must be an int, str, tuple, list, np.ndarray, "
                "or galois.Poly, not <class 'NoneType'>."
            )
    return left == right


def _fixed_point_num_int(dtype: RegisterType) -> SymbolicInt:
    parameters = dict(dtype.parameters)
    fractional_width = cast(SymbolicInt, parameters["fractional_width"])
    validate_fixed_point_parameters(
        dtype.num_qubits,
        fractional_width,
        signed=cast(bool, parameters["signed"]),
    )
    return dtype.num_qubits - fractional_width


def _unsigned_fixed_point_consistent(
    unsigned: RegisterType, fixed_point: RegisterType
) -> bool:
    parameters = dict(fixed_point.parameters)
    if parameters["signed"]:
        return False
    return unsigned.num_qubits == fixed_point.num_qubits and (
        parameters["fractional_width"] == 0 or _fixed_point_num_int(fixed_point) == 0
    )


def check_register_types_consistent(
    left: RegisterType,
    right: RegisterType,
    severity: RegisterTypeCheckingSeverity = RegisterTypeCheckingSeverity.LOOSE,
) -> bool:
    """Check declaration compatibility using the original ordered decision tree.

    This predicate does not perform a numerical cast or admit executable graph
    connections. Nominal field identity covers canonical numerical factories.
    """
    same_types = _same_declared_type(left, right)
    same_quantum_width = left.num_qubits == right.num_qubits
    if same_types:
        return True
    if left.num_qubits == 1 and same_quantum_width:
        return True
    if severity == RegisterTypeCheckingSeverity.STRICT:
        return False
    if left.encoding == "opaque" or right.encoding == "opaque":
        return same_quantum_width
    if severity == RegisterTypeCheckingSeverity.ANY:
        return False
    left_quantum = ("storage", "classical") not in left.parameters
    right_quantum = ("storage", "classical") not in right.parameters
    integers = ("signed", "uint", "montgomery")
    unsigned = ("uint", "montgomery", "finite_field")
    if (
        left_quantum
        and right_quantum
        and left.encoding in integers
        and right.encoding in integers
    ):
        return same_quantum_width
    if (
        left_quantum
        and right_quantum
        and left.encoding in unsigned
        and right.encoding == "fixed_point"
    ):
        return _unsigned_fixed_point_consistent(left, right)
    if (
        left_quantum
        and right_quantum
        and right.encoding in unsigned
        and left.encoding == "fixed_point"
    ):
        return _unsigned_fixed_point_consistent(right, left)
    return False
