# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors adapted from locked Qualtran v0.7.0 mcmt/multi_control_pauli.py.

"""Resolve literal multi-control Pauli dependencies through owned components."""

from collections.abc import Callable, Mapping
from dataclasses import replace
from functools import partial
from typing import Literal

import numpy as np

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import quantum_bit_type
from ...core.ir.program.registers import Register, RegisterSide
from ...core.ir.program.signatures import Signature
from ...errors import SerializationError
from ..registers.component_builder import ComponentProgramBuilder
from ..registers.component_graph import _namespace_helpers
from ..registers.control.multi_control_component import (
    multi_control_x_program,
    multi_control_z_program,
)
from .legacy_component_arguments import ControlSpecification
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_conjunction_factories import _control_value
from .legacy_controlled_factories import (
    PositiveControlProvider,
    _fixed_positive_control,
    controlled_via_and_factories,
)
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


def multi_control_factories(
    *,
    control_system: PositiveControlProvider = _fixed_positive_control,
    target_adjoint: Callable[[FunctionRecord], ProgramRecord] | None = None,
    adjoint_factory: ComponentFactory | None = None,
) -> dict[str, ComponentFactory]:
    """Supply X/Z specializations and the shared structured-control constructor.

    Generic targets delegate to the existing predicate and explicit child owners.
    An adjoint fallback enables scoped wrapper inversion without replacing an
    unrelated catalogue's inverse dispatch. Symbolic HasLength retains its
    unchanged source codec failure.
    """
    prefix = "qualtran.bloqs.mcmt."
    controls = controlled_via_and_factories(control_system=control_system)
    recipes: dict[FunctionRecord, tuple[FunctionRecord, tuple[int, ...]]] = {}

    def make(
        name: str,
        registers: tuple[Register, ...] | None,
        child: FunctionRecord,
        values: tuple[int, ...],
        child_module: ProgramRecord | None = None,
    ) -> ProgramRecord:
        if child.registers is None or len(child.registers) != 1:
            raise SerializationError("MultiControlPauli requires one target register")
        (target,) = child.registers
        if (
            target.shape_symbolic
            or target.side != RegisterSide.THRU
            or target.dtype.num_qubits != 1
            or target.dtype.num_cbits != 0
        ):
            raise SerializationError(
                "MultiControlPauli requires a scalar quantum-bit target"
            )
        bit = quantum_bit_type()
        signature = Signature(
            (
                *((Register("controls", bit, (len(values),)),) if values else ()),
                Register("target", bit),
            )
        )
        if registers is not None and registers != tuple(signature):
            raise SerializationError("Invalid register signature for MultiControlPauli")
        if values:
            controlled = controls[prefix + "controlled_via_and.ControlledViaAnd"](
                name + "_controlled",
                None,
                {
                    "subbloq": child,
                    "ctrl_spec": ControlSpecification(
                        ((bit, np.array(values, dtype=int)),)
                    ),
                },
            )
            assert isinstance(controlled, ProgramRecord)
            ports = controlled.functions[0].registers
            assert ports is not None
            control_name = ports[0].name
            builder = ComponentProgramBuilder(signature, name=name)
            outputs = builder.call_program(
                controlled,
                {
                    control_name: builder.inputs["controls"],
                    target.name: builder.inputs["target"],
                },
                entry=controlled.functions[0].name,
            )
            program = builder.finalize(
                {"controls": outputs[control_name], "target": outputs[target.name]}
            )
        else:
            # The enclosing resolver retains the forward child and its helpers.
            # An explicit inverse owner instead supplies its complete new module.
            definitions = (replace(child, blocks=()),) if child_module is None else ()
            builder = ComponentProgramBuilder(signature, definitions, name=name)
            inputs = {target.name: builder.inputs["target"]}
            outputs = (
                builder.call_dict(child.name, inputs)
                if child_module is None
                else builder.call_program(child_module, inputs, entry=child.name)
            )
            program = builder.finalize({"target": outputs[target.name]})
        result = _namespace_helpers(
            program, name, external=(child.name,) if child_module is None else ()
        )
        recipes[result.functions[0]] = child, values
        return result

    def generic(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        if set(arguments) != {"cvs", "target_bloq"} or not isinstance(
            arguments["target_bloq"], FunctionRecord
        ):
            raise SerializationError(
                "MultiControlPauli requires cvs and a resolved target_bloq"
            )
        return make(
            name, registers, arguments["target_bloq"], _literal_values(arguments["cvs"])
        )

    def inverse(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        child = arguments.get("subbloq")
        if (
            set(arguments) != {"subbloq"}
            or not isinstance(child, FunctionRecord)
            or child not in recipes
        ):
            assert adjoint_factory is not None
            return adjoint_factory(name, registers, arguments)
        target, values = recipes[child]
        if target_adjoint is None:
            leaf_inverse = None
            leaves = fixed_gate_factories()
            for identifier in ("s_gate.SGate", "t_gate.TGate"):
                factory = leaves["qualtran.bloqs.basic_gates." + identifier]
                for direction in (0, 1):
                    if target == factory(target.name, None, {"is_adjoint": direction}):
                        leaf_inverse = factory(
                            target.name + "_inverse",
                            None,
                            {"is_adjoint": 1 - direction},
                        )
            if leaf_inverse is not None:
                assert isinstance(leaf_inverse, FunctionRecord)
                return make(
                    name,
                    registers,
                    leaf_inverse,
                    values,
                    ProgramRecord((leaf_inverse,)),
                )
            try:
                _fixed_positive_control(target)
            except SerializationError as error:
                raise SerializationError(
                    "MultiControlPauli requires an exact target inverse owner"
                ) from error
            module = ProgramRecord((target,))
        else:
            module = target_adjoint(target)
        if not module.functions:
            raise SerializationError("MultiControlPauli target inverse module is empty")
        return make(name, registers, module.functions[0], values, module)

    result: dict[str, ComponentFactory] = {
        prefix + "multi_control_pauli.MultiControlX": partial(_multi_factory, "x"),
        prefix + "multi_control_pauli.MultiControlZ": partial(_multi_factory, "z"),
        prefix + "multi_control_pauli.MultiControlPauli": generic,
        **controls,
    }
    if adjoint_factory is not None:
        result["qualtran._infra.adjoint.Adjoint"] = inverse
    return result
