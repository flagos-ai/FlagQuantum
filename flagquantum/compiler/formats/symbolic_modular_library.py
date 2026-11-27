# Copyright 2023-2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Ordered recipes adapted from Qualtran v0.7.0 mod_addition.py/mod_subtraction.py.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic

"""Keep finite modular recipes and unresolved constructor data in owned libraries."""

from typing import Literal

from ...core.ir.program.register_types import (
    montgomery_register_type,
    quantum_bit_type,
    unsigned_register_type,
)
from ...core.ir.program.registers import Register, RegisterType
from ...core.value_math import SymbolicInt
from ._modular_addition_library import _addition_root, _constant_root, _scaled_root
from ._modular_library_builder import _ModularGraph
from ._modular_subtraction_library import _negation_root, _subtraction_root
from .legacy_component_arguments import (
    ComponentArgument,
    ComponentArgumentValue,
)
from .legacy_component_library import (
    ComponentDeclaration,
    ComponentLibrary,
)

ModularGraphKind = Literal[
    "ModAdd",
    "CModAdd",
    "CModAddK",
    "CtrlScaleModAdd",
    "ModNeg",
    "CModNeg",
    "ModSub",
    "CModSub",
]
_BIT = quantum_bit_type()
_ADD = "qualtran.bloqs.mod_arithmetic.mod_addition."
_SUB = "qualtran.bloqs.mod_arithmetic.mod_subtraction."


def modular_component_library(
    kind: ModularGraphKind,
    dtype: RegisterType,
    modulus: SymbolicInt,
    *,
    constant: SymbolicInt = 0,
    control_value: int = 1,
) -> ComponentLibrary:
    """Construct the complete finite source root, retaining symbolic child data.

    Words and indexed arrays are MSB first. Root-created sign, predicate and
    constant work ends at the exact source release/effect call. No clean-work or
    physical execution claim is made while any arithmetic parameter is unbound.
    """
    if (
        dtype.encoding not in {"uint", "montgomery"}
        or type(dtype.num_qubits) is int
        and dtype.num_qubits < 1
        or type(dtype.num_qubits) is not int
        and kind not in {"CModAddK", "ModSub", "CModSub"}
    ):
        raise ValueError(
            "Modular logical recipes require finite unsigned or Montgomery words"
        )
    if type(control_value) is not int or control_value not in (0, 1):
        raise ValueError("Modular control value must be binary")
    n = dtype.num_qubits
    controlled = kind.startswith("C")
    add = kind in ("ModAdd", "CModAdd", "CModAddK", "CtrlScaleModAdd")
    if kind not in (
        "ModAdd",
        "CModAdd",
        "CModAddK",
        "CtrlScaleModAdd",
        "ModNeg",
        "CModNeg",
        "ModSub",
        "CModSub",
    ):
        raise ValueError("Unknown modular logical recipe")
    if kind == "ModAdd":
        dtype = montgomery_register_type(n, modulus)
    elif kind in ("CModAddK", "CtrlScaleModAdd"):
        dtype = unsigned_register_type(n)
    ports = [Register("ctrl", _BIT)] if controlled else []
    ports.append(Register("x", dtype))
    if kind not in ("CModAddK", "ModNeg", "CModNeg"):
        ports.append(Register("y", dtype))
    args: dict[str, ComponentArgumentValue] = {"mod": modulus}
    if kind in ("ModAdd", "CModAddK", "CtrlScaleModAdd"):
        args["bitsize"] = n
    else:
        args["dtype"] = dtype
    if kind in ("CModAddK", "CtrlScaleModAdd"):
        args["k"] = constant
    if kind in ("CModAdd", "CModNeg", "CModSub"):
        args["cv"] = control_value
    graph = _ModularGraph(
        ComponentDeclaration(
            (_ADD if add else _SUB) + kind,
            tuple(ComponentArgument(k, v) for k, v in args.items()),
            tuple(ports),
        )
    )
    if kind in ("ModAdd", "CModAdd"):
        outputs = _addition_root(graph, dtype, modulus, controlled, control_value)
    elif kind == "CModAddK":
        outputs = _constant_root(graph, dtype, modulus, constant)
    elif kind == "CtrlScaleModAdd":
        outputs = _scaled_root(graph, dtype, modulus, constant)
    elif kind in ("ModSub", "CModSub"):
        outputs = _subtraction_root(graph, dtype, modulus, controlled, control_value)
    else:
        outputs = _negation_root(graph, dtype, modulus, controlled, control_value)
    return graph.finish(outputs)
