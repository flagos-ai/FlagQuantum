# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the complete locked Qualtran v0.7.0 comparison notebook.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/comparison.ipynb

"""Fixed comparison teaching instances using native signatures and graph drawing.

These are the notebook's one-level declarations. Labels retain all source
callee fields, including distinct adjoints. They do not allocate physical work
or admit a foreign operation to execution. Existing append examples own that path.
"""

from sympy import symbols

from flagquantum.core.ir.program.register_types import (
    opaque_register_type,
    quantum_bit_type,
    signed_register_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature

ComparisonExample = tuple[str, Signature, dict[str, int]]


def comparison_examples() -> tuple[ComparisonExample, ...]:
    """Retain every original instance, symbolic assumption and ordered port."""
    n, n1, n2 = symbols("n n1 n2")
    bit = quantum_bit_type()
    uint = unsigned_register_type
    binary = Signature(
        (Register("a", uint(4)), Register("b", uint(4)), Register("target", bit))
    )
    constant = Signature((Register("x", uint(4)), Register("target", bit)))
    examples: list[ComparisonExample] = [
        (
            "greater_than",
            binary,
            {"LessThanEqual(x_bitsize=4, y_bitsize=4)": 1, "XGate()": 1},
        ),
        ("gt_k", constant, {"LessThanConstant(bitsize=4, less_than_val=13)": 1}),
        ("eq_k", constant, {"MultiControlX(cvs=(1, 1, 0, 1))": 1}),
        ("lt_k_symb", Signature((Register("x", uint(n)), Register("target", bit))), {}),
        (
            "lt_k",
            Signature((Register("x", uint(8)), Register("target", bit))),
            {
                "And(cv1=1, cv2=1, uncompute=False)": 8,
                "And(cv1=1, cv2=1, uncompute=True)": 8,
                "CNOT()": 18,
                "XGate()": 6,
            },
        ),
        (
            "bi_qubits_mixer",
            Signature(
                (
                    Register("x", uint(2)),
                    Register("y", uint(2)),
                    Register(
                        "ancilla", opaque_register_type(3), side=RegisterSide.RIGHT
                    ),
                )
            ),
            {"XGate()": 1, "CNOT()": 9, "And(cv1=1, cv2=1, uncompute=False)": 2},
        ),
        (
            "sq_cmp",
            Signature(
                (
                    Register("a", bit),
                    Register("b", bit),
                    Register("less_than", bit, side=RegisterSide.RIGHT),
                    Register("greater_than", bit, side=RegisterSide.RIGHT),
                )
            ),
            {"XGate()": 1, "CNOT()": 4, "And(cv1=1, cv2=1, uncompute=False)": 1},
        ),
        (
            "leq_symb",
            Signature(
                (
                    Register("x", uint(n1)),
                    Register("y", uint(n2)),
                    Register("target", bit),
                )
            ),
            {},
        ),
        (
            "leq",
            Signature(
                (
                    Register("x", uint(4)),
                    Register("y", uint(8)),
                    Register("target", bit),
                )
            ),
            {
                "MultiAnd(cvs=(0, 0, 0, 0))": 1,
                "Adjoint(subbloq=MultiAnd(cvs=(0, 0, 0, 0)))": 1,
                "CNOT()": 2,
                "XGate()": 1,
                "BiQubitsMixer(is_adjoint=False)": 3,
                "BiQubitsMixer(is_adjoint=True)": 3,
                "SingleQubitCompare(is_adjoint=False)": 1,
                "SingleQubitCompare(is_adjoint=True)": 1,
                "And(cv1=1, cv2=0, uncompute=False)": 1,
                "And(cv1=1, cv2=0, uncompute=True)": 1,
            },
        ),
        (
            "clineardepthgreaterthan_example",
            Signature(
                (
                    Register("ctrl", bit),
                    Register("a", signed_register_type(5)),
                    Register("b", signed_register_type(5)),
                    Register("target", bit),
                )
            ),
            {
                "BitwiseNot(dtype=QInt(bitsize=6))": 2,
                "BitwiseNot(dtype=QUInt(bitsize=7))": 2,
                "OutOfPlaceAdder(bitsize=6, is_adjoint=True, include_most_significant_bit=True)": 1,
                "OutOfPlaceAdder(bitsize=6, is_adjoint=False, include_most_significant_bit=True)": 1,
                "MultiControlX(cvs=(1, 1))": 1,
                "SignExtend(inp_dtype=QInt(bitsize=5), out_dtype=QInt(bitsize=6))": 2,
                "SignTruncate(inp_dtype=QInt(bitsize=6), out_dtype=QInt(bitsize=5))": 2,
            },
        ),
        (
            "equals",
            Signature(
                (
                    Register("x", uint(4)),
                    Register("y", uint(4)),
                    Register("target", bit),
                )
            ),
            {
                "Xor(dtype=QUInt(bitsize=4))": 2,
                "MultiControlX(cvs=(0, 0, 0, 0))": 1,
            },
        ),
    ]
    half_signature = Signature(
        (
            Register("a", uint(3)),
            Register("b", uint(3)),
            Register("c", uint(4), side=RegisterSide.RIGHT),
            Register("target", bit, side=RegisterSide.RIGHT),
        )
    )
    half_calls = {
        "BitwiseNot(dtype=QUInt(bitsize=4))": 3,
        "CNOT()": 1,
        "OutOfPlaceAdder(bitsize=4, is_adjoint=False, include_most_significant_bit=False)": 1,
    }
    for relation in ("greaterthan", "greaterthanequal", "lessthan", "lessthanequal"):
        calls = dict(half_calls)
        if relation.endswith("equal"):
            calls["XGate()"] = 1
        examples.append((f"lineardepthhalf{relation}_small", half_signature, calls))
    return tuple(examples)
