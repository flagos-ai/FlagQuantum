# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Call declarations from locked Qualtran v0.7.0 cryptography/ecc/ec_add.py.

"""Keep the seven curve-addition resource declarations independent of gates."""

from collections.abc import Hashable

from ...compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from ...core.ir.program.register_types import (
    montgomery_register_type,
    quantum_bit_type,
)
from ...core.value_math import SymbolicInt

_STAGE_NAMES = tuple(
    "ECAddStep" + suffix for suffix in ("One", "Two", "Three", "Four", "Five", "Six")
)


def elliptic_curve_addition_callees(
    width: SymbolicInt,
    modulus: SymbolicInt,
    window: SymbolicInt = 1,
    *,
    stage: int | None = None,
) -> dict[Hashable, SymbolicInt]:
    """Retain actual child parameters, directions and encoded control clauses.

    Keys are ordinary hashable tuples used by the existing graph estimator, not
    executable components or a serialized schema. Controlled clauses retain
    (dtype, array shape, flattened values). Unknown-length patterns remain
    HasLength identities, including when width is a symbolic integer constant.
    No price, physical decomposition or numerical input admission is inferred.
    """
    if stage is None:
        return {
            (name, width, modulus, *(() if i in (0, 5) else (window,))): 1
            for i, name in enumerate(_STAGE_NAMES)
        }
    if type(stage) is not int or not 1 <= stage <= 6:
        raise ValueError("Curve addition resource stage must be an integer from 1 to 6")
    word = montgomery_register_type(width)
    bit = quantum_bit_type()
    zeros = (0,) * (2 * width) if isinstance(width, int) else ("HasLength", 2 * width)
    equals = ("Equals", word)
    subtract = ("ModSub", word, modulus)
    csubtract = ("CModSub", word, modulus, 1)
    add = ("ModAdd", width, modulus)
    cadd = ("CModAdd", word, modulus, 1)
    inverse = ("KaliskiModInverse", width, modulus, False)
    uninverse = ("KaliskiModInverse", width, modulus, True)
    product = ("DirtyOutOfPlaceMontgomeryModMul", width, window, modulus, False)
    unproduct = ("DirtyOutOfPlaceMontgomeryModMul", width, window, modulus, True)
    if stage == 1:
        return {
            equals: 2,
            ("ModNeg", word, modulus): 2,
            ("MultiControlX", zeros): 2,
            ("MultiControlX", (0, 0, 0)): 1,
        }
    if stage == 2:
        return {
            equals: 2,
            ("ControlledViaAnd", equals, ((bit, (), (0,)),)): 1,
            subtract: 1,
            csubtract: 1,
            inverse: 1,
            product: 1,
            ("MultiControlX", (0, 1, 1)): width,
            ("MultiControlX", (1, 1, 1)): width,
            unproduct: 1,
            uninverse: 1,
        }
    if stage == 3:
        double = ("ModDbl", word, modulus)
        return {
            csubtract: 1,
            product: 1,
            unproduct: 1,
            ("CNOT",): 2 * width,
            double: 1,
            add: 1,
            cadd: 1,
            ("Adjoint", add): 1,
            ("Adjoint", double): 1,
        }
    if stage == 4:
        return {subtract: 1, product: 2, unproduct: 2, ("CNOT",): 3 * width}
    if stage == 5:
        clear = (
            "ControlledViaAnd",
            ("Xor", word),
            ((bit, (), (1,)), (word, (), (0,))),
        )
        return {
            csubtract: 1,
            inverse: 1,
            product: 1,
            unproduct: 1,
            uninverse: 1,
            add: 1,
            ("MultiControlX", (1, 1)): width,
            clear: 1,
            ("CModNeg", word, modulus, 1): 1,
        }
    conjunction = ("MultiAnd", zeros)
    return {
        ("MultiControlX", zeros): 1,
        ("ControlledViaAnd", ("XGate",), ((word, (3,), (0, 0, 0)),)): 4,
        ("MultiControlX", (0, 0, 0)): 1,
        csubtract: 1,
        cadd: 1,
        ("Toffoli",): 2 * width + 4,
        ("Equals", montgomery_register_type(2 * width)): 1,
        conjunction: 1,
        ("MultiTargetCNOT", 2): 1,
        ("Adjoint", conjunction): 1,
    }


def elliptic_curve_addition_parameterized_graph(
    width: SymbolicInt,
    modulus: SymbolicInt,
    window: SymbolicInt = 1,
    *,
    stage: int | None = None,
) -> SymbolicCallGraph[Hashable]:
    """Expand the complete declared root or one stage, retaining opaque leaves.

    Price terminal identities explicitly with estimate_leaf_cost. In particular,
    a symbolic product has no fabricated numerical price or emitted gate stream.
    """
    direct = elliptic_curve_addition_callees(width, modulus, window, stage=stage)
    if stage is None:
        root: Hashable = ("ECAdd", width, modulus, window)
        declarations = {root: direct}
        for index, child in enumerate(direct, 1):
            declarations[child] = elliptic_curve_addition_callees(
                width, modulus, window, stage=index
            )
    else:
        root = (
            _STAGE_NAMES[stage - 1],
            width,
            modulus,
            *(() if stage in (1, 6) else (window,)),
        )
        declarations = {root: direct}
    return symbolic_call_graph(root, declarations)
