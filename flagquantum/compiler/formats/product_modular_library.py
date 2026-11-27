# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Ordered recipes adapted from locked Qualtran v0.7.0 mod_multiplication.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_multiplication.py

"""Preserve finite modular doubling and constant-product constructor graphs."""

from __future__ import annotations

from dataclasses import replace
from math import gcd
from typing import Literal

import sympy

from ...core.ir.program.register_types import (
    opaque_register_type,
    quantum_bit_type,
    signed_register_type,
    unsigned_register_type,
)
from ...core.ir.program.registers import Register, RegisterType
from ...core.value_math import SymbolicInt
from ._modular_library_builder import _ModularGraph
from .legacy_component_arguments import ComponentArgument, ComponentArgumentValue
from .legacy_component_library import ComponentDeclaration, ComponentLibrary

_PREFIX = "qualtran.bloqs.mod_arithmetic.mod_multiplication."
_BIT = quantum_bit_type()


def product_modular_component_library(
    kind: Literal["ModDbl", "CModMulK"],
    dtype: RegisterType,
    modulus: SymbolicInt,
    *,
    constant: SymbolicInt = 1,
) -> ComponentLibrary:
    """Retain literal source calls and expressions without numerical admission.

    In the symbolic product branch the original reciprocal lacks a minus sign.
    It remains source data here, not a replacement for the mathematical inverse.
    """
    if (
        kind not in ("ModDbl", "CModMulK")
        or dtype.encoding not in ("uint", "montgomery")
        or dtype.classical_bits != 0
    ):
        raise ValueError("Modular product graphs require a supported unsigned word")
    n = dtype.num_qubits
    if kind == "ModDbl":
        if type(n) is not int or n < 1:
            raise ValueError("Source doubling requires a concrete positive split width")
        if type(modulus) is int and modulus % 2 != 1:
            raise ValueError("Source doubling requires an odd modulus")
    elif type(modulus) is int and type(constant) is int:
        if not 0 < constant < modulus or gcd(constant, modulus) != 1:
            raise ValueError("Source modular multiplier requires a unit constant")
    ports: tuple[Register, ...] = (Register("x", dtype),)
    args: dict[str, ComponentArgumentValue] = {"dtype": dtype, "mod": modulus}
    if kind == "CModMulK":
        ports = (Register("ctrl", _BIT), *ports)
    arguments = [ComponentArgument(key, value) for key, value in args.items()]
    if kind == "CModMulK":
        arguments.append(ComponentArgument("k", constant))
    graph = _ModularGraph(ComponentDeclaration(_PREFIX + kind, tuple(arguments), ports))
    b = graph.builder
    if kind == "CModMulK":
        y = graph.allocate(opaque_register_type(n))
        reciprocal = (
            sympy.Mod(sympy.Pow(constant, -1), modulus)
            if isinstance(modulus, sympy.Expr) or isinstance(constant, sympy.Expr)
            else -pow(constant, -1, modulus)
        )
        word = unsigned_register_type(n)
        scale_ports = (Register("ctrl", _BIT), Register("x", word), Register("y", word))
        first = graph.call(
            "qualtran.bloqs.mod_arithmetic.mod_addition.CtrlScaleModAdd",
            scale_ports,
            {"ctrl": b.inputs["ctrl"], "x": b.inputs["x"], "y": y},
            {"k": constant, "bitsize": n, "mod": modulus},
        )
        second = graph.call(
            "qualtran.bloqs.mod_arithmetic.mod_addition.CtrlScaleModAdd",
            scale_ports,
            {"ctrl": first["ctrl"], "x": first["y"], "y": first["x"]},
            {"k": reciprocal, "bitsize": n, "mod": modulus},
        )
        opaque = opaque_register_type(n)
        swapped = graph.call(
            "qualtran.bloqs.basic_gates.swap.CSwap",
            (Register("ctrl", _BIT), Register("x", opaque), Register("y", opaque)),
            {"ctrl": second["ctrl"], "x": second["y"], "y": second["x"]},
            {"bitsize": n},
        )
        graph.free(swapped["y"])
        return graph.finish({"ctrl": swapped["ctrl"], "x": swapped["x"]})
    assert isinstance(n, int)
    lower, sign = graph.allocate(), graph.allocate()
    bits = graph.split(b.inputs["x"])
    value = graph.join([sign, *bits, lower], replace(dtype, quantum_bits=n + 2))
    value = graph.add_constant(signed_register_type(n + 2), -modulus, {"x": value})["x"]
    bits = graph.split(value)
    sign, value = bits[0], graph.join(bits[1:], replace(dtype, quantum_bits=n + 1))
    corrected = graph.add_constant(
        unsigned_register_type(n + 1),
        modulus,
        {"ctrl": sign, "x": value},
        controlled=True,
    )
    bits = graph.split(corrected["x"])
    lower = graph.x(bits[-1])
    flipped = graph.call(
        "qualtran.bloqs.basic_gates.cnot.CNOT",
        (Register("ctrl", _BIT), Register("target", _BIT)),
        {"ctrl": lower, "target": corrected["ctrl"]},
    )
    lower = graph.x(flipped["ctrl"])
    value = graph.join([*bits[1:-1], lower], dtype)
    graph.free(bits[0])
    graph.free(flipped["target"])
    return graph.finish({"x": value})
