# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Ordered recipes adapted from Qualtran v0.7.0 mod_division.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_division.py

"""Retain source-defined Kaliski graphs independently of unresolved arithmetic."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Literal

import numpy as np

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import (
    montgomery_register_type,
    opaque_register_type,
    quantum_bit_type,
)
from ...core.ir.program.registers import Register, RegisterSide
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt
from ..registers.component_adjoint import adjoint_component_program
from ..registers.component_builder import BindingCollection
from ._inverse_iteration_library import (
    inverse_iteration_root,
    inverse_ports,
    inverse_step_root,
)
from ._modular_library_builder import _ModularGraph
from .legacy_component_arguments import (
    ComponentArgument,
    ComponentReference,
    ControlSpecification,
)
from .legacy_component_library import ComponentDeclaration, ComponentLibrary

InverseGraphKind = Literal[
    "_KaliskiIterationStep3",
    "_KaliskiIterationStep4",
    "_KaliskiIterationStep5",
    "_KaliskiIteration",
    "_KaliskiModInverseImpl",
    "KaliskiModInverse",
]
_PREFIX = "qualtran.bloqs.mod_arithmetic.mod_division."
_BIT = quantum_bit_type()


def inverse_modular_component_library(
    kind: InverseGraphKind,
    width: SymbolicInt,
    modulus: SymbolicInt | None = None,
    *,
    uncompute: bool = False,
) -> ComponentLibrary:
    """Keep complete finite source calls, raw symbols and explicit work ownership.

    Words are MSB first; retained history is an output, not clean private work.
    Symbolic leaf declarations and this graph grant no numerical execution.
    """
    if kind not in (
        "_KaliskiIterationStep3",
        "_KaliskiIterationStep4",
        "_KaliskiIterationStep5",
        "_KaliskiIteration",
        "_KaliskiModInverseImpl",
        "KaliskiModInverse",
    ):
        raise ValueError("Unsupported source-defined Kaliski graph")
    if type(uncompute) is not bool or uncompute and kind != "KaliskiModInverse":
        raise ValueError("Only the public Kaliski wrapper has a direction field")
    step = int(kind[-1]) if kind.startswith("_KaliskiIterationStep") else None
    if step is None and modulus is None:
        raise ValueError("Kaliski iteration and algorithm require a modulus")
    p = 1 if modulus is None else modulus
    if kind in ("_KaliskiModInverseImpl", "KaliskiModInverse") and (
        type(width) is not int or width < 1
    ):
        raise ValueError("The source algorithm requires a concrete positive width")
    if kind == "KaliskiModInverse":
        if type(p) is int and p % 2 != 1:
            raise ValueError("The source public inverse requires an odd modulus")
        ports: tuple[Register, ...] = (
            Register("x", montgomery_register_type(width, p)),
            Register(
                "junk",
                opaque_register_type(4 * width),
                side=RegisterSide.LEFT if uncompute else RegisterSide.RIGHT,
            ),
        )
    else:
        ports = inverse_ports(kind, width, p)
    arguments = [ComponentArgument("bitsize", width)]
    if step is None:
        arguments.append(ComponentArgument("mod", p))
    if kind == "KaliskiModInverse":
        arguments.append(ComponentArgument("uncompute", uncompute))
    graph = _ModularGraph(ComponentDeclaration(_PREFIX + kind, tuple(arguments), ports))
    if step is not None:
        outputs = inverse_step_root(graph, step, width)
    elif kind == "_KaliskiIteration":
        outputs = inverse_iteration_root(graph, width, p, graph.builder.inputs)
    elif kind == "_KaliskiModInverseImpl":
        assert isinstance(width, int)
        outputs = _algorithm_root(graph, width, p, graph.builder.inputs)
    else:
        assert isinstance(width, int)
        outputs = _public_root(graph, width, p, uncompute)
    return graph.finish(outputs)


def _xor(
    g: _ModularGraph,
    n: int,
    k: SymbolicInt,
    word: BindingCollection,
    ctrl: BindingCollection | None = None,
) -> dict[str, BindingCollection]:
    dtype = montgomery_register_type(n)
    child = g.declare(
        "qualtran.bloqs.arithmetic.bitwise.XorK",
        (Register("x", dtype),),
        {"dtype": dtype, "k": k},
    )
    if ctrl is None:
        return g.builder.call_dict(child, {"x": word})
    wrapper = g.declare(
        "qualtran._infra.controlled.Controlled",
        (Register("ctrl", _BIT), Register("x", dtype)),
        {
            "subbloq": ComponentReference(tuple(g.declarations).index(child)),
            "ctrl_spec": ControlSpecification(((_BIT, np.asarray(1)),)),
        },
    )
    return g.builder.call_dict(wrapper, {"ctrl": ctrl, "x": word})


def _algorithm_root(
    g: _ModularGraph, n: int, p: SymbolicInt, inputs: Mapping[str, BindingCollection]
) -> dict[str, BindingCollection]:
    values = dict(inputs)
    values["f"] = g.x(values["f"])
    values["u"] = _xor(g, n, p, values["u"])["x"]
    values["s"] = _xor(g, n, 1, values["s"])["x"]
    ms, terminals = g.split(values["m"]), g.split(values["terminal_condition"])
    ports = inverse_ports("_KaliskiIteration", n, p)
    for i in range(2 * n):
        stepped = g.call(
            _PREFIX + "_KaliskiIteration",
            ports,
            {key: values[key] for key in ("u", "v", "r", "s", "f")}
            | {"m": ms[i], "is_terminal": terminals[i]},
            {"bitsize": n, "mod": p},
        )
        ms[i], terminals[i] = stepped.pop("m"), stepped.pop("is_terminal")
        values.update(stepped)
    dtype = montgomery_register_type(n)
    values["r"] = g.call(
        "qualtran.bloqs.arithmetic.bitwise.BitwiseNot",
        (Register("x", dtype),),
        {"x": values["r"]},
        {"dtype": dtype},
    )["x"]
    values["r"] = g.add_constant(dtype, p + 1, {"x": values["r"]})["x"]
    values["u"] = _xor(g, n, 1, values["u"])["x"]
    values["s"] = _xor(g, n, p, values["s"])["x"]
    for key, pattern in (("r", p), ("u", p - 1), ("s", p - 1)):
        corrected = _xor(g, n, pattern, values[key], terminals[0])
        terminals[0], values[key] = corrected["ctrl"], corrected["x"]
    values["m"] = g.join(ms, opaque_register_type(2 * n))
    values["terminal_condition"] = g.join(terminals, opaque_register_type(2 * n))
    return values


def _reverse_algorithm(
    g: _ModularGraph, outputs: Mapping[str, object]
) -> ProgramRecord:
    """Reuse the exact adjoint engine with only source-explicit leaf decisions."""
    record = g.builder.finalize(outputs)
    used = {op.callee for op in record.functions[0].blocks[0].operations}
    mapping: dict[str, str] = {}
    declarations = list(g.declarations.items())
    for name, declaration in declarations:
        if name not in used:
            continue
        identity = declaration.name
        if identity.endswith((".XGate", ".XorK", ".BitwiseNot")):
            mapping[name] = name
        elif identity == "qualtran._infra.controlled.Controlled":
            reference = next(
                arg.value for arg in declaration.arguments if arg.name == "subbloq"
            )
            assert isinstance(reference, ComponentReference)
            if not declarations[reference.identifier][1].name.endswith(".XorK"):
                raise ValueError("No source adjoint decision for controlled child")
            mapping[name] = name
        elif identity in (
            "qualtran.bloqs.bookkeeping.split.Split",
            "qualtran.bloqs.bookkeeping.join.Join",
        ):
            assert declaration.registers is not None
            signature = Signature(declaration.registers).adjoint()
            reverse_ports = (*signature.lefts(), *signature.rights())
            counterpart = "qualtran.bloqs.bookkeeping." + (
                "join.Join" if identity.endswith(".Split") else "split.Split"
            )
            mapping[name] = g.declare(
                counterpart,
                reverse_ports,
                {a.name: a.value for a in declaration.arguments},
            )
        elif identity in (
            _PREFIX + "_KaliskiIteration",
            "qualtran.bloqs.arithmetic.addition.AddK",
        ):
            mapping[name] = g.declare_adjoint(name)
        else:
            raise ValueError(f"No source adjoint decision for {identity}")
    existing = {function.name for function in record.functions}
    extras = tuple(
        FunctionRecord(name, (), registers=declaration.registers)
        for name, declaration in g.declarations.items()
        if name not in existing
    )
    return adjoint_component_program(
        replace(record, functions=(*record.functions, *extras)), mapping
    )


def _public_root(
    g: _ModularGraph, n: int, p: SymbolicInt, inverse: bool
) -> dict[str, BindingCollection]:
    dtype = montgomery_register_type(n)
    u, r, s = g.allocate(dtype), g.allocate(dtype), g.allocate(dtype)
    f = g.allocate()
    if inverse:
        junk_bits = g.split(g.builder.inputs["junk"])
        m = g.join(junk_bits[: 2 * n], opaque_register_type(2 * n))
        terminal = g.join(junk_bits[2 * n :], opaque_register_type(2 * n))
    else:
        m, terminal = g.allocate(opaque_register_type(2 * n)), g.allocate(
            opaque_register_type(2 * n)
        )
    root = ComponentDeclaration(
        _PREFIX + "_KaliskiModInverseImpl",
        (ComponentArgument("bitsize", n), ComponentArgument("mod", p)),
        inverse_ports("_KaliskiModInverseImpl", n, p),
    )
    inner = _ModularGraph(root)
    inner_outputs = _algorithm_root(inner, n, p, inner.builder.inputs)
    record = (
        _reverse_algorithm(inner, inner_outputs)
        if inverse
        else inner.builder.finalize(inner_outputs)
    )
    inputs = {
        "u": u,
        "v": r if inverse else g.builder.inputs["x"],
        "r": g.builder.inputs["x"] if inverse else r,
        "s": s,
        "m": m,
        "f": f,
        "terminal_condition": terminal,
    }
    computed = g.inline(record, inner.declarations, inputs)
    if inverse:
        for key in ("u", "r", "s", "m", "f", "terminal_condition"):
            g.free(computed[key])
        return {"x": computed["v"]}
    for key in ("u", "v", "s", "f"):
        g.free(computed[key])
    junk = g.join(
        [*g.split(computed["m"]), *g.split(computed["terminal_condition"])],
        opaque_register_type(4 * n),
    )
    return {"x": computed["r"], "junk": junk}
