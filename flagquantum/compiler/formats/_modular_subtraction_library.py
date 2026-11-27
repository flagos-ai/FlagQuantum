# Copyright 2023-2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Ordered recipes adapted from Qualtran v0.7.0 mod_addition.py/mod_subtraction.py.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic

"""Declare the ordered negation and subtraction modular source roots."""

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
from .legacy_component_arguments import (
    ComponentArgumentValue,
    ComponentReference,
    ControlSpecification,
)

_BIT = quantum_bit_type()
_ADD = "qualtran.bloqs.mod_arithmetic.mod_addition."
_ARITH = "qualtran.bloqs.arithmetic."
_BASIC = "qualtran.bloqs.basic_gates."


def _subtraction_root(
    g: _ModularGraph, dtype: RegisterType, p: SymbolicInt, controlled: bool, cv: int
) -> dict[str, BindingCollection]:
    b = g.builder
    not_ports = (Register("x", dtype),)
    x = g.call(
        _ARITH + "bitwise.BitwiseNot", not_ports, {"x": b.inputs["x"]}, {"dtype": dtype}
    )["x"]
    x = g.add_constant(dtype, p + 1, {"x": x})["x"]
    args: dict[str, ComponentArgumentValue] = {"mod": p}
    ports = [Register("ctrl", _BIT)] if controlled else []
    inner = dtype if controlled else montgomery_register_type(dtype.num_qubits, p)
    ports.extend((Register("x", inner), Register("y", inner)))
    values = {"x": x, "y": b.inputs["y"]}
    if controlled:
        args.update(dtype=dtype, cv=cv)
        values["ctrl"] = b.inputs["ctrl"]
    else:
        args["bitsize"] = dtype.num_qubits
    added = g.call(
        _ADD + ("CModAdd" if controlled else "ModAdd"), tuple(ports), values, args
    )
    x = g.add_constant(dtype, p + 1, {"x": added["x"]}, adjoint=True)["x"]
    result = {
        "x": g.call(
            _ARITH + "bitwise.BitwiseNot", not_ports, {"x": x}, {"dtype": dtype}
        )["x"],
        "y": added["y"],
    }
    if controlled:
        result["ctrl"] = added["ctrl"]
    return result


def _negation_root(
    g: _ModularGraph, dtype: RegisterType, p: SymbolicInt, controlled: bool, cv: int
) -> dict[str, BindingCollection]:
    n = int(dtype.num_qubits)
    ctrl = g.builder.inputs.get("ctrl")
    x = g.builder.inputs["x"]
    flag = g.x(g.allocate())

    def zero(
        value: BindingCollection, target: BindingCollection
    ) -> dict[str, BindingCollection]:
        if controlled:
            bits = g.split(value)
            tested = g.call(
                "qualtran.bloqs.mcmt.multi_control_pauli.MultiControlX",
                (Register("controls", _BIT, (n,)), Register("target", _BIT)),
                {"controls": bits, "target": target},
                {"cvs": np.zeros(n, dtype=int)},
            )
            return {
                "x": g.join(tested["controls"], opaque_register_type(n)),
                "flag": tested["target"],
            }
        child = g.declare(_BASIC + "x_basis.XGate", (Register("q", _BIT),))
        wrapper = "qualtran.bloqs.mcmt.controlled_via_and.ControlledViaAnd"
        result = g.call(
            wrapper,
            (Register("ctrl", dtype), Register("q", _BIT)),
            {"ctrl": value, "q": target},
            {
                "subbloq": ComponentReference(tuple(g.declarations).index(child)),
                "ctrl_spec": ControlSpecification(((dtype, np.asarray(0)),)),
            },
        )
        return {"x": result["ctrl"], "flag": result["q"]}

    tested = zero(x, flag)
    flag, x = tested["flag"], tested["x"]
    active = flag
    pair = None
    if controlled:
        pair = g.call(
            "qualtran.bloqs.mcmt.and_bloq.And",
            (
                Register("ctrl", _BIT, (2,)),
                Register("target", _BIT, side=RegisterSide.RIGHT),
            ),
            {"ctrl": [ctrl, flag]},
            {"cv1": cv, "cv2": 1, "uncompute": False},
        )
        active = pair["target"]
    spread = g.call(
        "qualtran.bloqs.mcmt.multi_target_cnot.MultiTargetCNOT",
        (
            Register("control", _BIT),
            Register("targets", _BIT if n == 1 else opaque_register_type(n)),
        ),
        {"control": active, "targets": x},
        {"bitsize": n},
    )
    added = g.add_constant(
        dtype if controlled else unsigned_register_type(n),
        p + 1,
        {"ctrl": spread["control"], "x": spread["targets"]},
        controlled=True,
    )
    x, flag = added["x"], added["ctrl"]
    if pair is not None:
        cleared = g.call(
            "qualtran.bloqs.mcmt.and_bloq.And",
            (
                Register("ctrl", _BIT, (2,)),
                Register("target", _BIT, side=RegisterSide.LEFT),
            ),
            {"ctrl": pair["ctrl"], "target": flag},
            {"cv1": cv, "cv2": 1, "uncompute": True},
        )["ctrl"]
        assert isinstance(cleared, np.ndarray)
        ctrl, flag = cleared[0], cleared[1]
    tested = zero(x, flag)
    g.free(g.x(tested["flag"]))
    result = {"x": tested["x"]}
    if controlled:
        assert ctrl is not None
        result["ctrl"] = ctrl
    return result
