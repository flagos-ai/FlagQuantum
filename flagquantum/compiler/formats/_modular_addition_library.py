# Copyright 2023-2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Ordered recipes adapted from Qualtran v0.7.0 mod_addition.py/mod_subtraction.py.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic

"""Declare the ordered addition, constant-loading and scaled modular source roots."""

import numpy as np

from ...core.ir.program.register_types import (
    montgomery_register_type,
    opaque_register_type,
    quantum_bit_type,
    unsigned_register_type,
)
from ...core.ir.program.registers import Register, RegisterSide, RegisterType
from ...core.value_math import SymbolicInt
from ..registers.component_builder import BindingCollection
from ._modular_library_builder import _ModularGraph

_BIT = quantum_bit_type()
_ADD = "qualtran.bloqs.mod_arithmetic.mod_addition."
_ARITH = "qualtran.bloqs.arithmetic."
_BASIC = "qualtran.bloqs.basic_gates."
_BOOK = "qualtran.bloqs.bookkeeping."


def _addition_root(
    g: _ModularGraph, dtype: RegisterType, p: SymbolicInt, controlled: bool, cv: int
) -> dict[str, BindingCollection]:
    b = g.builder
    n = int(dtype.num_qubits)
    word = unsigned_register_type(n) if controlled else montgomery_register_type(n)
    wide = (
        unsigned_register_type(n + 1) if controlled else montgomery_register_type(n + 1)
    )
    x, y = b.inputs["x"], b.inputs["y"]
    junk = None
    if not controlled:
        junk, sign = g.allocate(), g.allocate()
        xs, ys = g.split(x), g.split(y)
        x, y = g.join([junk, *xs], wide), g.join([sign, *ys], wide)
        added = g.call(
            _ARITH + "addition.Add",
            (Register("a", wide), Register("b", wide)),
            {"a": x, "b": y},
            {"a_dtype": wide, "b_dtype": wide},
        )
        xs = g.split(added["a"])
        junk, x = xs[0], g.join(xs[1:], word)
    else:
        ys = g.split(y)
        sign = g.allocate()
        x = g.call(
            _BOOK + "cast.Cast",
            (
                Register("reg", dtype, side=RegisterSide.LEFT),
                Register("reg", word, side=RegisterSide.RIGHT),
            ),
            {"reg": x},
            {
                "inp_dtype": dtype,
                "out_dtype": word,
                "shape": (),
                "allow_quantum_to_classical": False,
            },
        )["reg"]
        y = g.join([sign, *ys], wide)
        added = g.call(
            _ARITH + "controlled_addition.CAdd",
            (Register("ctrl", _BIT), Register("a", word), Register("b", wide)),
            {"ctrl": b.inputs["ctrl"], "a": x, "b": y},
            {"a_dtype": word, "b_dtype": wide, "cv": cv},
        )
        x = added["a"]
    y = g.add_constant(wide, -p, {"x": added["b"]})["x"]
    ys = g.split(y)
    y = g.join(ys[1:], opaque_register_type(n) if controlled else word)
    corrected = g.add_constant(word, p, {"ctrl": ys[0], "x": y}, controlled=True)
    cmp_ports = ([Register("ctrl", _BIT)] if controlled else []) + [
        Register("a", unsigned_register_type(n)),
        Register("b", unsigned_register_type(n)),
        Register("target", _BIT),
    ]
    values = {"a": x, "b": corrected["x"], "target": corrected["ctrl"]}
    if controlled:
        values["ctrl"] = added["ctrl"]
    compared = g.call(
        _ARITH
        + "comparison."
        + ("CLinearDepthGreaterThan" if controlled else "LinearDepthGreaterThan"),
        tuple(cmp_ports),
        values,
        {"dtype": word, "cv": cv} if controlled else {"bitsize": n, "signed": False},
    )
    sign = g.x(compared["target"])
    if junk is not None:
        g.free(junk)
    g.free(sign)
    if not controlled:
        return {"x": compared["a"], "y": compared["b"]}
    result = {"ctrl": compared["ctrl"]}
    for key in ("x", "y"):
        result[key] = g.call(
            _BOOK + "cast.Cast",
            (
                Register("reg", word, side=RegisterSide.LEFT),
                Register("reg", dtype, side=RegisterSide.RIGHT),
            ),
            {"reg": compared["a" if key == "x" else "b"]},
            {
                "inp_dtype": word,
                "out_dtype": dtype,
                "shape": (),
                "allow_quantum_to_classical": False,
            },
        )["reg"]
    return result


def _constant_root(
    g: _ModularGraph, dtype: RegisterType, p: SymbolicInt, k: SymbolicInt
) -> dict[str, BindingCollection]:
    n = dtype.num_qubits
    if type(k) is int and type(n) is int and not 0 <= k < 1 << n:
        raise ValueError("Constant state must fit the source unsigned word")
    state_type = _BIT if n == 1 else opaque_register_type(n)
    value = g.call(
        _BASIC + "z_basis.IntState",
        (Register("val", state_type, side=RegisterSide.RIGHT),),
        {},
        {"val": k, "bitsize": n},
    )["val"]
    added = g.call(
        _ADD + "CModAdd",
        (Register("ctrl", _BIT), Register("x", dtype), Register("y", dtype)),
        {"ctrl": g.builder.inputs["ctrl"], "x": value, "y": g.builder.inputs["x"]},
        {"dtype": dtype, "mod": p, "cv": 1},
    )
    g.call(
        _BASIC + "z_basis.IntEffect",
        (Register("val", state_type, side=RegisterSide.LEFT),),
        {"val": added["x"]},
        {"val": k, "bitsize": n},
    )
    return {"ctrl": added["ctrl"], "x": added["y"]}


def _scaled_root(
    g: _ModularGraph, dtype: RegisterType, p: SymbolicInt, k: SymbolicInt
) -> dict[str, BindingCollection]:
    n = int(dtype.num_qubits)
    ctrl, y = g.builder.inputs["ctrl"], g.builder.inputs["y"]
    bits = g.split(g.builder.inputs["x"])
    for i in range(n):
        ports = (
            Register("ctrl", _BIT, (2,)),
            Register("target", _BIT, side=RegisterSide.RIGHT),
        )
        tested = g.call(
            "qualtran.bloqs.mcmt.and_bloq.And",
            ports,
            {"ctrl": [ctrl, bits[i]]},
            {"cv1": 1, "cv2": 1, "uncompute": False},
        )
        active = g.call(
            _ADD + "CModAddK",
            (Register("ctrl", _BIT), Register("x", dtype)),
            {"ctrl": tested["target"], "x": y},
            {"k": k * 2 ** (n - 1 - i) % p, "mod": p, "bitsize": n},
        )
        cleared = g.call(
            "qualtran.bloqs.mcmt.and_bloq.And",
            (
                Register("ctrl", _BIT, (2,)),
                Register("target", _BIT, side=RegisterSide.LEFT),
            ),
            {"ctrl": tested["ctrl"], "target": active["ctrl"]},
            {"cv1": 1, "cv2": 1, "uncompute": True},
        )["ctrl"]
        assert isinstance(cleared, np.ndarray)
        ctrl, bits[i] = cleared[0], cleared[1]
        y = active["x"]
    return {"ctrl": ctrl, "x": g.join(bits, dtype), "y": y}
