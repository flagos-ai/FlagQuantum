# Copyright 2023-2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor control protocols follow locked Qualtran v0.7.0 mod_arithmetic.

"""Keep modular constructor provenance across exact inverses and added controls."""

from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from typing import Literal

import sympy

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import montgomery_register_type, quantum_bit_type
from ...core.ir.program.registers import Register, RegisterType
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt
from ...errors import SerializationError
from ..registers.component_control import ControlAdder, _stored_control_adder
from ..registers.modular.additional_control_component import (
    additional_modular_addition_program,
    additional_modular_subtraction_program,
)
from ..registers.modular.controlled_modular_addition_component import (
    controlled_modular_addition_program,
)
from ..registers.modular.controlled_modular_subtraction_component import (
    controlled_modular_subtraction_program,
)
from ..registers.modular.modular_addition_component import modular_addition_program
from ..registers.modular.modular_subtraction_component import (
    modular_subtraction_program,
)
from .legacy_component_arguments import ControlSpecification
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_controlled_factories import controlled_via_and_factories
from .legacy_modular_addition_factories import (
    _integer,
    _symbolic_addition,
    modular_addition_factories,
)
from .legacy_modular_subtraction_factories import modular_subtraction_factories
from .legacy_symbolic_modular_controls import (
    symbolic_additional_modular_addition_program,
    symbolic_additional_modular_subtraction_program,
    symbolic_modular_subtraction_program,
)
from .legacy_word_factories import _rename_program
from .montgomery_port_views import restore_montgomery_ports

_Kind = Literal["ModAdd", "CModAdd", "ModSub", "CModSub"]


def modular_control_factories(
    *,
    controlled_factory: ComponentFactory | None = None,
    adjoint_factory: ComponentFactory | None = None,
) -> dict[str, ComponentFactory]:
    """Resolve finite-width modular source owners and one additional control.

    Existing forward factories retain their symbolic scope. Added controls and
    exact inverse construction support concrete or unresolved symbolic moduli;
    symbolic constants retain actual AddK refusals. Other source-success
    contexts remain explicit unsupported native boundaries.
    Unknown children delegate only to supplied owners, never signature guesses.
    """
    factories = {**modular_addition_factories(), **modular_subtraction_factories()}
    recipes: dict[FunctionRecord, tuple[_Kind, dict[str, ResolvedArgument], bool]] = {}

    def parameters(
        kind: _Kind, arguments: Mapping[str, ResolvedArgument]
    ) -> tuple[RegisterType, SymbolicInt, int | None]:
        modulus = _integer(arguments, "mod")
        dtype = (
            montgomery_register_type(_integer(arguments, "bitsize"), modulus)
            if kind == "ModAdd"
            else arguments["dtype"]
        )
        if (
            not isinstance(dtype, RegisterType)
            or type(dtype.num_qubits) is not int
            or (
                type(modulus) is not int
                and not (isinstance(modulus, sympy.Expr) and modulus.free_symbols)
            )
        ):
            raise SerializationError(
                "This modular control scope requires concrete width and integer or unresolved modulus"
            )
        cv = arguments.get("cv", 1) if kind.startswith("C") else None
        if cv is not None and (type(cv) is not int or cv not in (0, 1)):
            raise SerializationError("Modular control value must be binary")
        assert isinstance(cv, int) or cv is None
        return dtype, modulus, cv

    def make(
        kind: _Kind,
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
        inverse: bool = False,
    ) -> FunctionRecord | ProgramRecord:
        module = "mod_addition" if kind.endswith("Add") else "mod_subtraction"
        result = factories[f"qualtran.bloqs.mod_arithmetic.{module}.{kind}"](
            name, registers, arguments
        )
        if inverse:
            dtype, modulus, cv = parameters(kind, arguments)
            if isinstance(modulus, sympy.Expr):
                program = (
                    _symbolic_addition(
                        dtype,
                        modulus,
                        controlled=cv is not None,
                        control_value=1 if cv is None else cv,
                        adjoint=True,
                    )
                    if kind.endswith("Add")
                    else symbolic_modular_subtraction_program(
                        dtype, modulus, control_value=cv, adjoint=True
                    )
                )
            elif kind == "ModAdd":
                program = modular_addition_program(
                    dtype.num_qubits, modulus, adjoint=True
                )
            elif kind == "CModAdd":
                assert cv is not None
                program = controlled_modular_addition_program(
                    dtype, modulus, control_value=cv, adjoint=True
                )
            elif kind == "ModSub":
                program = modular_subtraction_program(dtype, modulus, adjoint=True)
            else:
                assert cv is not None
                program = controlled_modular_subtraction_program(
                    dtype, modulus, control_value=cv, adjoint=True
                )
            if registers is not None and registers != program.functions[0].registers:
                program = restore_montgomery_ports(program, registers)
            result = _rename_program(program, name)
        root = result.functions[0] if isinstance(result, ProgramRecord) else result
        recipes[root] = kind, dict(arguments), inverse
        return result

    def inverse(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        child = arguments.get("subbloq")
        if (
            set(arguments) == {"subbloq"}
            and isinstance(child, FunctionRecord)
            and child in recipes
        ):
            kind, values, direction = recipes[child]
            return make(kind, name, registers, values, not direction)
        if adjoint_factory is not None:
            return adjoint_factory(name, registers, arguments)
        raise SerializationError("Missing exact modular adjoint constructor context")

    def positive(child: FunctionRecord) -> tuple[ProgramRecord, ControlAdder]:
        if child not in recipes:
            raise SerializationError(
                "Missing modular positive-control constructor context"
            )
        kind, values, direction = recipes[child]
        dtype, modulus, cv = parameters(kind, values)
        if isinstance(modulus, sympy.Expr):
            program = (
                symbolic_additional_modular_addition_program
                if kind.endswith("Add")
                else symbolic_additional_modular_subtraction_program
            )(dtype, modulus, control_value=cv, adjoint=direction)
        else:
            program = (
                additional_modular_addition_program
                if kind.endswith("Add")
                else additional_modular_subtraction_program
            )(dtype, modulus, control_value=cv, adjoint=direction)
        program = _rename_program(program, child.name + "_positive")
        return program, _stored_control_adder(program.functions[0])

    def controlled(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        child, spec = arguments.get("subbloq"), arguments.get("ctrl_spec")
        if not isinstance(child, FunctionRecord) or child not in recipes:
            if controlled_factory is not None:
                return controlled_factory(name, registers, arguments)
            raise SerializationError("Missing modular controlled constructor context")
        if (
            set(arguments) != {"subbloq", "ctrl_spec"}
            or not isinstance(spec, ControlSpecification)
            or len(spec.clauses) != 1
            or spec.clauses[0][0] != quantum_bit_type()
            or spec.clauses[0][1].shape != ()
            or spec.clauses[0][1].item() != 1
        ):
            raise SerializationError(
                "Modular control requires one positive scalar QBit"
            )
        program, _ = positive(child)
        if registers is not None and registers != program.functions[0].registers:
            program = restore_montgomery_ports(
                _rename_program(program, "main"), registers
            )
        return _rename_program(program, name)

    predicate_factories = controlled_via_and_factories(control_system=positive)
    predicate_name = "qualtran.bloqs.mcmt.controlled_via_and.ControlledViaAnd"

    def predicate(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        child = arguments.get("subbloq")
        if (
            isinstance(child, FunctionRecord)
            and child in recipes
            and recipes[child][0] == "ModAdd"
        ):
            kind, values, direction = recipes[child]
            dtype, _, _ = parameters(kind, values)
            qualified = replace(
                child,
                name=child.name + "_qualified",
                blocks=(),
                registers=tuple(Signature.build_from_dtypes(x=dtype, y=dtype)),
            )
            recipes[qualified] = kind, values, direction
            program = predicate_factories[predicate_name](
                name, None, {**arguments, "subbloq": qualified}
            )
            assert isinstance(program, ProgramRecord)
            if registers is not None and registers != program.functions[0].registers:
                program = restore_montgomery_ports(
                    _rename_program(program, "main"), registers
                )
            return _rename_program(program, name)
        return predicate_factories[predicate_name](name, registers, arguments)

    result = dict(factories)
    for kind in ("ModAdd", "CModAdd", "ModSub", "CModSub"):
        module = "mod_addition" if kind.endswith("Add") else "mod_subtraction"
        result[f"qualtran.bloqs.mod_arithmetic.{module}.{kind}"] = partial(make, kind)
    result["qualtran.bloqs.mod_arithmetic.ModAdd"] = result[
        "qualtran.bloqs.mod_arithmetic.mod_addition.ModAdd"
    ]
    result.update(
        {
            "qualtran._infra.adjoint.Adjoint": inverse,
            "qualtran._infra.controlled.Controlled": controlled,
            predicate_name: predicate,
        }
    )
    return result
