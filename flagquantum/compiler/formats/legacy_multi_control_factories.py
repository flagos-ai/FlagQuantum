# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors adapted from locked Qualtran v0.7.0 mcmt/multi_control_pauli.py.

"""Resolve literal multi-control Pauli dependencies through owned components."""

from collections.abc import Mapping
from functools import partial
from typing import Literal

import numpy as np

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import quantum_bit_type
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...errors import SerializationError
from ..registers.component_builder import ComponentProgramBuilder
from ..registers.control.multi_control_component import (
    multi_control_x_program,
    multi_control_z_program,
)
from .legacy_component_arguments import ControlSpecification
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_conjunction_factories import _control_value
from .legacy_fixed_gate_factories import fixed_gate_factories
from .legacy_word_factories import _rename_program


def _literal_values(values: object) -> tuple[int, ...]:
    if isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise SerializationError(
                "Multi-control cvs requires a one-dimensional array"
            )
        values = tuple(values)
    if not isinstance(values, (tuple, list)):
        raise SerializationError("Multi-control cvs requires a literal sequence")
    result = tuple(_control_value(value) for value in values)
    if not all(isinstance(value, int) for value in result):
        raise SerializationError("Multi-control requires concrete binary values")
    return tuple(int(value) for value in result)


def _pauli_program(axis: Literal["x", "z"], values: tuple[int, ...]) -> ProgramRecord:
    return (multi_control_x_program if axis == "x" else multi_control_z_program)(values)


def _checked(
    program: ProgramRecord, name: str, registers: tuple[Register, ...] | None
) -> ProgramRecord:
    if registers is not None and registers != program.functions[0].registers:
        raise SerializationError("Invalid register signature for multi-control")
    return _rename_program(program, name)


def _multi_factory(
    axis: Literal["x", "z"],
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> ProgramRecord:
    if set(arguments) != {"cvs"}:
        raise SerializationError("Multi-control requires only cvs")
    return _checked(
        _pauli_program(axis, _literal_values(arguments["cvs"])), name, registers
    )


def _controlled_factory(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> ProgramRecord:
    if set(arguments) != {"subbloq", "ctrl_spec"}:
        raise SerializationError("ControlledViaAnd requires subbloq and ctrl_spec")
    child, spec = arguments["subbloq"], arguments["ctrl_spec"]
    if not isinstance(child, FunctionRecord):
        raise SerializationError("ControlledViaAnd requires a resolved Pauli child")
    leaves = fixed_gate_factories()
    axis: Literal["x", "z"]
    if child == leaves["qualtran.bloqs.basic_gates.x_basis.XGate"](
        child.name, None, {}
    ):
        axis = "x"
    elif child == leaves["qualtran.bloqs.basic_gates.z_basis.ZGate"](
        child.name, None, {}
    ):
        axis = "z"
    else:
        raise SerializationError("ControlledViaAnd supports the scalar X/Z dependency")
    if not isinstance(spec, ControlSpecification) or len(spec.clauses) != 1:
        raise SerializationError("ControlledViaAnd requires one QBit control clause")
    dtype, activation = spec.clauses[0]
    if dtype != quantum_bit_type():
        raise SerializationError("ControlledViaAnd requires QBit control storage")
    values = _literal_values(activation.reshape(-1))
    if not values:
        raise SerializationError("ControlledViaAnd requires nonempty controls")
    signature = Signature(
        (Register("ctrl", dtype, activation.shape), Register("q", dtype))
    )
    builder = ComponentProgramBuilder(signature)
    result = builder.call_program(
        _pauli_program(axis, values),
        {
            "controls": np.asarray(builder.inputs["ctrl"]).reshape(-1),
            "target": builder.inputs["q"],
        },
    )
    controls = np.asarray(result["controls"]).reshape(activation.shape)
    return _checked(
        builder.finalize(
            {
                "ctrl": controls.item() if not activation.shape else controls,
                "q": result["target"],
            }
        ),
        name,
        registers,
    )


def multi_control_factories() -> dict[str, ComponentFactory]:
    """Supply X/Z specializations and their scalar-Pauli controlled dependency.

    Generic controlled children and non-QBit clauses retain their separate
    admission boundary. Symbolic HasLength fails in the unchanged source codec.
    """
    prefix = "qualtran.bloqs.mcmt."
    return {
        prefix + "multi_control_pauli.MultiControlX": partial(_multi_factory, "x"),
        prefix + "multi_control_pauli.MultiControlZ": partial(_multi_factory, "z"),
        prefix + "controlled_via_and.ControlledViaAnd": _controlled_factory,
    }
