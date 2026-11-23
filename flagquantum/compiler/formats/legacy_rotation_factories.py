# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors adapted from locked Qualtran v0.7.0 basic_gates/rotation.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/basic_gates/rotation.py

"""Resolve the complete Pauli rotation constructor group through native owners."""

from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from typing import Literal

import sympy

from ...core.ir import CircuitIR
from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import quantum_bit_type
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicReal
from ...errors import SerializationError
from ..registers.component_gate_function import bit_gate_function
from ..registers.component_graph import _namespace_helpers
from ..registers.primitives.controlled_rotation_component import (
    controlled_rz_program,
    controlled_z_power_program,
)
from ..registers.primitives.global_phase import _finite_angle
from ..registers.primitives.pauli_rotation import (
    PauliAxis,
    pauli_power_ir,
    pauli_rotation_ir,
)
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .symbolic_parameters import symbolic_parameter_value

_Kind = Literal[
    "Rx", "Ry", "Rz", "CRz", "XPowGate", "YPowGate", "ZPowGate", "CZPowGate"
]


def _real_argument(
    arguments: Mapping[str, ResolvedArgument], key: str, default: float | None = None
) -> SymbolicReal:
    value = arguments.get(key, default)
    if isinstance(value, sympy.Expr):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _finite_angle(value)
    raise SerializationError(f"Rotation constructor requires a real scalar at {key!r}")


def _rotation_factory(
    kind: _Kind,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    power = kind.endswith("PowGate")
    key = "exponent" if power else "angle"
    shifted = kind in ("XPowGate", "YPowGate")
    allowed = {key, "eps", *(("global_shift",) if shifted else ())}
    if arguments.keys() - allowed:
        raise SerializationError(f"Unexpected constructor arguments for {kind}")
    value = _real_argument(arguments, key, 1.0 if power else None)
    # Precision remains in the serialized library and the independent resource
    # declarations. The source explicitly excludes it from numerical action.
    _real_argument(arguments, "eps", 1e-11)
    angle = symbolic_parameter_value(value) if isinstance(value, sympy.Expr) else value
    signature: tuple[Register, ...]
    if kind == "CZPowGate":
        signature = (Register("q", quantum_bit_type(), (2,)),)
    else:
        signature = tuple(
            Signature.build(ctrl=1, q=1) if kind == "CRz" else Signature.build(q=1)
        )
    if registers is not None and registers != signature:
        raise SerializationError(f"Invalid register signature for {kind}")
    if kind in ("CZPowGate", "CRz"):
        program = (
            controlled_z_power_program(angle)
            if kind == "CZPowGate"
            else controlled_rz_program(angle)
        )
        program = replace(
            program,
            functions=(
                replace(program.functions[0], name=name),
                *program.functions[1:],
            ),
        )
        return _namespace_helpers(program, name)
    axes: dict[str, PauliAxis] = {
        "Rx": "x",
        "Ry": "y",
        "Rz": "z",
        "XPowGate": "x",
        "YPowGate": "y",
        "ZPowGate": "z",
    }
    if power:
        shift = _real_argument(arguments, "global_shift", 0.0) if shifted else 0.0
        if isinstance(shift, sympy.Expr):
            shift = symbolic_parameter_value(shift)
        if not isinstance(shift, (int, float)):
            raise SerializationError(
                "Pauli power global_shift requires a concrete real scalar"
            )
        instructions = pauli_power_ir(axes[kind], 0, angle, global_shift=shift)
    else:
        instructions = pauli_rotation_ir(axes[kind], 0, angle)
    return bit_gate_function(name, signature, CircuitIR(1, instructions))


def rotation_factories() -> dict[str, ComponentFactory]:
    """Supply all eight rotation.py constructors to explicit native resolution.

    Existing component bodies retain exact phases, source control decompositions
    and workspace ownership. Owned expressions bind through normal Circuit calls.
    """
    kinds: tuple[_Kind, ...] = (
        "Rx",
        "Ry",
        "Rz",
        "CRz",
        "XPowGate",
        "YPowGate",
        "ZPowGate",
        "CZPowGate",
    )
    return {
        f"qualtran.bloqs.basic_gates.rotation.{kind}": partial(_rotation_factory, kind)
        for kind in kinds
    }
