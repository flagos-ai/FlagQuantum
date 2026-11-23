# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Fixed gate names and constructors adapted from Qualtran v0.7.0.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/serialization/resolver_dict.py

"""Explicit native constructors for fixed unitary library leaves."""

from collections.abc import Mapping
from functools import partial

from ...core.ir import CircuitIR, Instruction
from ...core.ir.program import FunctionRecord
from ...core.ir.program.register_types import quantum_bit_type
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...errors import SerializationError
from ..registers.component_gate_function import bit_gate_function
from .legacy_component_resolution import ComponentFactory, ResolvedArgument


def _fixed_gate_factory(
    gate: str,
    signature: tuple[Register, ...],
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord:
    if registers is not None and registers != signature:
        raise SerializationError(f"Invalid register signature for {gate!r}")
    allowed = {"is_adjoint"} if gate in ("s", "t") else set()
    if arguments.keys() - allowed:
        raise SerializationError(f"Unexpected constructor arguments for {gate!r}")
    adjoint = arguments.get("is_adjoint", False)
    # The wire codec represents source booleans as integer arguments.
    if type(adjoint) not in (bool, int) or adjoint not in (0, 1):
        raise SerializationError("Gate is_adjoint must be a boolean or binary integer")
    operation = f"{gate}dg" if adjoint else gate
    width = sum(1 for register in signature for _ in register.all_idxs())
    return bit_gate_function(
        name,
        signature,
        CircuitIR(width, (Instruction(operation, tuple(range(width))),)),
    )


def fixed_gate_factories() -> dict[str, ComponentFactory]:
    """Supply explicit constructors to the existing library resolver.

    Each call returns an independent mapping. Names are interchange identifiers
    only. Execution uses native typed gate lowering and never imports source
    constructors. Serialized S/T adjoints retain their exact gate direction.
    """
    single = tuple(Signature.build(q=1))
    controlled = tuple(Signature.build(ctrl=1, target=1))
    toffoli = (
        Register("ctrl", quantum_bit_type(), (2,)),
        Register("target", quantum_bit_type()),
    )
    definitions = (
        ("x_basis.XGate", "x", single),
        ("z_basis.ZGate", "z", single),
        ("z_basis.CZ", "cz", tuple(Signature.build(q1=1, q2=1))),
        ("cnot.CNOT", "cx", controlled),
        ("hadamard.Hadamard", "h", single),
        ("hadamard.CHadamard", "ch", controlled),
        ("s_gate.SGate", "s", single),
        ("t_gate.TGate", "t", single),
        ("toffoli.Toffoli", "ccx", toffoli),
        ("y_gate.YGate", "y", single),
        ("y_gate.CYGate", "cy", controlled),
    )
    return {
        f"qualtran.bloqs.basic_gates.{source}": partial(
            _fixed_gate_factory, gate, signature
        )
        for source, gate, signature in definitions
    }
