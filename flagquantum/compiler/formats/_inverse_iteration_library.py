# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Recipes adapted from locked Qualtran v0.7.0 mod_division.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_division.py

"""Declare the source comparison, exchange, arithmetic and iteration layers."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ...core.ir.program.register_types import (
    montgomery_register_type,
    opaque_register_type,
    quantum_bit_type,
    unsigned_register_type,
)
from ...core.ir.program.registers import Register, RegisterSide
from ...core.value_math import SymbolicInt
from ..registers.component_builder import BindingCollection
from ._modular_library_builder import _ModularGraph
from .legacy_component_arguments import ComponentArgumentValue

_BIT = quantum_bit_type()
_PREFIX = "qualtran.bloqs.mod_arithmetic.mod_division."
_STEP_PORTS = {
    1: ("v", "m", "f", "is_terminal"),
    2: ("u", "v", "b", "a", "m", "f"),
    3: ("u", "v", "b", "a", "m", "f"),
    4: ("u", "v", "r", "s", "a"),
    5: ("u", "v", "r", "s", "b", "f"),
    6: ("u", "v", "r", "s", "b", "a", "m", "f"),
}


def inverse_ports(
    kind: str, width: SymbolicInt, modulus: SymbolicInt
) -> tuple[Register, ...]:
    """Retain the original word metadata and ordered scalar flag ports."""
    if kind.startswith("_KaliskiIterationStep"):
        step = int(kind[-1])
        dtype = (
            montgomery_register_type(width, modulus)
            if step == 6
            else montgomery_register_type(width)
        )
        return tuple(
            Register(name, dtype if name in ("u", "v", "r", "s") else _BIT)
            for name in _STEP_PORTS[step]
        )
    words = tuple(
        Register(name, montgomery_register_type(width, modulus))
        for name in ("u", "v", "r", "s")
    )
    if kind == "_KaliskiIteration":
        return (
            *words,
            Register("m", _BIT),
            Register("f", _BIT),
            Register("is_terminal", _BIT),
        )
    return (
        *words,
        Register("m", opaque_register_type(2 * width)),
        Register("f", _BIT),
        Register("terminal_condition", opaque_register_type(2 * width)),
    )


def _cnot(
    g: _ModularGraph, ctrl: BindingCollection, target: BindingCollection
) -> dict[str, BindingCollection]:
    return g.call(
        "qualtran.bloqs.basic_gates.cnot.CNOT",
        (Register("ctrl", _BIT), Register("target", _BIT)),
        {"ctrl": ctrl, "target": target},
    )


def inverse_step_root(
    g: _ModularGraph, step: int, n: SymbolicInt
) -> dict[str, BindingCollection]:
    """Retain the source layer calls and their explicitly owned predicate work."""
    values = dict(g.builder.inputs)
    dtype = montgomery_register_type(n)
    if step == 4:
        for left, right in (("u", "v"), ("r", "s")):
            swapped = g.call(
                "qualtran.bloqs.basic_gates.swap.CSwap",
                (
                    Register("ctrl", _BIT),
                    Register("x", opaque_register_type(n)),
                    Register("y", opaque_register_type(n)),
                ),
                {"ctrl": values["a"], "x": values[left], "y": values[right]},
                {"bitsize": n},
            )
            values.update(
                a=swapped["ctrl"], **{left: swapped["x"], right: swapped["y"]}
            )
        return values
    if step == 5:
        tested = g.call(
            "qualtran.bloqs.mcmt.and_bloq.And",
            (
                Register("ctrl", _BIT, (2,)),
                Register("target", _BIT, side=RegisterSide.RIGHT),
            ),
            {"ctrl": [values["f"], values["b"]]},
            {"cv1": 1, "cv2": 0, "uncompute": False},
        )
        current = tested["target"]
        not_ports = (Register("x", dtype),)
        values["v"] = g.call(
            "qualtran.bloqs.arithmetic.bitwise.BitwiseNot",
            not_ports,
            {"x": values["v"]},
            {"dtype": dtype},
        )["x"]
        for left, right in (("u", "v"), ("r", "s")):
            added = g.call(
                "qualtran.bloqs.arithmetic.controlled_addition.CAdd",
                (Register("ctrl", _BIT), Register("a", dtype), Register("b", dtype)),
                {"ctrl": current, "a": values[left], "b": values[right]},
                {"a_dtype": dtype, "b_dtype": dtype, "cv": 1},
            )
            current, values[left], values[right] = added["ctrl"], added["a"], added["b"]
            if left == "u":
                values["v"] = g.call(
                    "qualtran.bloqs.arithmetic.bitwise.BitwiseNot",
                    not_ports,
                    {"x": values["v"]},
                    {"dtype": dtype},
                )["x"]
        cleared = g.call(
            "qualtran.bloqs.mcmt.and_bloq.And",
            (
                Register("ctrl", _BIT, (2,)),
                Register("target", _BIT, side=RegisterSide.LEFT),
            ),
            {"ctrl": tested["ctrl"], "target": current},
            {"cv1": 1, "cv2": 0, "uncompute": True},
        )["ctrl"]
        assert isinstance(cleared, np.ndarray)
        values["f"], values["b"] = cleared[0], cleared[1]
        return values
    if step != 3:
        raise ValueError("This source symbolic layer must be step three, four or five")
    comparison_ports = (
        Register("a", dtype),
        Register("b", dtype),
        Register("c", unsigned_register_type(n + 1), side=RegisterSide.RIGHT),
        Register("target", _BIT, side=RegisterSide.RIGHT),
    )
    compared = g.call(
        "qualtran.bloqs.arithmetic.comparison.LinearDepthHalfGreaterThan",
        comparison_ports,
        {"a": values["u"], "b": values["v"]},
        {"dtype": dtype, "uncompute": False},
    )
    multi_ports = (
        Register("ctrl", _BIT, (3,)),
        Register("junk", _BIT, (1,), RegisterSide.RIGHT),
        Register("target", _BIT, side=RegisterSide.RIGHT),
    )
    multi = g.declare(
        "qualtran.bloqs.mcmt.and_bloq.MultiAnd",
        multi_ports,
        {"cvs": np.asarray((1, 1, 0))},
    )
    tested = g.builder.call_dict(
        multi, {"ctrl": [compared["target"], values["f"], values["b"]]}
    )
    current = tested["target"]
    for target in ("a", "m"):
        copied = _cnot(g, current, values[target])
        current, values[target] = copied["ctrl"], copied["target"]
    cleared = g.builder.call_dict(
        g.declare_adjoint(multi),
        {"ctrl": tested["ctrl"], "junk": tested["junk"], "target": current},
    )["ctrl"]
    assert isinstance(cleared, np.ndarray)
    greater, values["f"], values["b"] = cleared[0], cleared[1], cleared[2]
    restored = g.call(
        "qualtran.bloqs.arithmetic.comparison.LinearDepthHalfGreaterThan",
        (
            Register("a", dtype),
            Register("b", dtype),
            Register("c", unsigned_register_type(n + 1), side=RegisterSide.LEFT),
            Register("target", _BIT, side=RegisterSide.LEFT),
        ),
        {"a": compared["a"], "b": compared["b"], "c": compared["c"], "target": greater},
        {"dtype": dtype, "uncompute": True},
    )
    values["u"], values["v"] = restored["a"], restored["b"]
    return values


def inverse_iteration_root(
    g: _ModularGraph,
    n: SymbolicInt,
    p: SymbolicInt,
    inputs: Mapping[str, BindingCollection],
) -> dict[str, BindingCollection]:
    """Apply the six source layers and release only their two branch flags."""
    values = dict(inputs)
    values.update(a=g.allocate(), b=g.allocate())
    for step in range(1, 7):
        kind = f"_KaliskiIterationStep{step}"
        ports = inverse_ports(kind, n, p)
        arguments: dict[str, ComponentArgumentValue] = {"bitsize": n}
        if step == 6:
            arguments["mod"] = p
        values.update(
            g.call(
                _PREFIX + kind,
                ports,
                {name: values[name] for name in _STEP_PORTS[step]},
                arguments,
            )
        )
    g.free(values.pop("a"))
    g.free(values.pop("b"))
    return values
