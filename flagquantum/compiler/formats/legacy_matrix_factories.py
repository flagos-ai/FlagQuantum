# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor metadata follows locked Qualtran v0.7.0 for_testing/matrix_gate.py.

"""Resolve explicit matrix data through the existing owned Program leaf."""

from collections.abc import Mapping
from math import isfinite

import numpy as np

from ...core.ir.program import BlockRecord, FunctionRecord, Operation, ValueRef
from ...core.ir.program.matrix import matrix_error
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...errors import SerializationError
from .legacy_component_resolution import ComponentFactory, ResolvedArgument


def _matrix_factory(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord:
    if not {"bitsize", "matrix"} <= arguments.keys() or arguments.keys() - {
        "bitsize",
        "matrix",
        "atol",
    }:
        raise SerializationError("Matrix constructor requires bitsize and matrix")
    width = arguments["bitsize"]
    if type(width) is not int or width < 1:
        raise SerializationError(
            "Matrix execution requires a positive concrete bitsize"
        )
    assert isinstance(width, int)
    tolerance = arguments.get("atol", 1e-10)
    if (
        not isinstance(tolerance, (int, float))
        or isinstance(tolerance, bool)
        or not isfinite(tolerance)
        or tolerance < 0
    ):
        raise SerializationError("Matrix tolerance must be finite and nonnegative")
    signature = tuple(Signature.build(q=width))
    if registers is not None and registers != signature:
        raise SerializationError("Invalid register signature for MatrixGate")
    raw = arguments["matrix"]
    if not isinstance(raw, (tuple, list, np.ndarray)):
        raise SerializationError("Matrix data requires a numeric array")
    try:
        array = np.asarray(raw)
    except (TypeError, ValueError) as error:
        raise SerializationError(
            "Matrix data requires a rectangular numeric array"
        ) from error
    if array.ndim != 2 or array.dtype.kind not in "iufc":
        raise SerializationError("Matrix data requires a numeric two-dimensional array")
    matrix = tuple(tuple(complex(value) for value in row) for row in array)
    problem = matrix_error(matrix, width)
    if problem is not None:
        raise SerializationError("Invalid explicit matrix: " + problem)
    incoming = ValueRef("entry", 0, "register")
    bits = tuple(ValueRef("entry", 1 + i, "qubit") for i in range(width))
    successors = tuple(ValueRef("entry", 1 + width + i, "qubit") for i in range(width))
    outgoing = ValueRef("entry", 2 * width + 1, "register")
    return FunctionRecord(
        name,
        (
            BlockRecord(
                "entry",
                (incoming,),
                operations=(
                    Operation("split_register", (incoming,), bits),
                    Operation("any", bits, successors, matrix=matrix),
                    Operation(
                        "join_register",
                        successors,
                        (outgoing,),
                        register_type=signature[0].dtype,
                    ),
                ),
                return_values=(outgoing,),
            ),
        ),
        registers=signature,
    )


def matrix_factories() -> dict[str, ComponentFactory]:
    """Supply a concrete matrix constructor without importing an optional SDK."""
    return {"qualtran.bloqs.for_testing.matrix_gate.MatrixGate": _matrix_factory}
