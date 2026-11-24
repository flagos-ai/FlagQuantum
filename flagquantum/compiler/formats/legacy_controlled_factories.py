# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor adaptation from Qualtran v0.7.0 mcmt/controlled_via_and.py.

"""Resolve structured controls through existing positive-control providers."""

from collections.abc import Callable, Mapping
from dataclasses import replace

import numpy as np

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import quantum_bit_type
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...errors import SerializationError
from ..registers.component_builder import BindingCollection, ComponentProgramBuilder
from ..registers.component_control import (
    ControlAdder,
    control_component_via_predicate,
    quantum_control_signature,
)
from .legacy_component_arguments import ControlSpecification
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_control_specs import control_registers_from_spec
from .legacy_fixed_gate_factories import fixed_gate_factories
from .legacy_word_factories import _rename_program

PositiveControlProvider = Callable[[FunctionRecord], tuple[ProgramRecord, ControlAdder]]


def _fixed_positive_control(
    child: FunctionRecord,
) -> tuple[ProgramRecord, ControlAdder]:
    leaves = fixed_gate_factories()
    for source, target, control_port, data_port in (
        ("x_basis.XGate", "cnot.CNOT", "ctrl", "target"),
        ("y_gate.YGate", "y_gate.CYGate", "ctrl", "target"),
        ("z_basis.ZGate", "z_basis.CZ", "q1", "q2"),
        ("hadamard.Hadamard", "hadamard.CHadamard", "ctrl", "target"),
    ):
        if child == leaves["qualtran.bloqs.basic_gates." + source](
            child.name, None, {}
        ):
            function = leaves["qualtran.bloqs.basic_gates." + target](
                child.name + "_positive", None, {}
            )
            assert isinstance(function, FunctionRecord)

            def add(
                owner: ComponentProgramBuilder,
                control: BindingCollection,
                values: dict[str, BindingCollection],
                *,
                callee: str = function.name,
                control_key: str = control_port,
                data_key: str = data_port,
            ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
                outputs = owner.call_dict(
                    callee, {control_key: control, data_key: values["q"]}
                )
                return outputs[control_key], (outputs[data_key],)

            return ProgramRecord((function,)), add
    raise SerializationError(
        "ControlledViaAnd child requires a registered positive-control provider"
    )


def controlled_via_and_factories(
    *, control_system: PositiveControlProvider = _fixed_positive_control
) -> dict[str, ComponentFactory]:
    """Adapt the complete constructor without duplicating child implementations.

    The injected owner returns the actual positive-control module and its adder.
    It owns arbitrary target semantics, helper definitions and private work.
    Default fixed X/Y/Z/H providers preserve their source atomic controls.
    Nested constructor recipes retain complete functions and ordered clauses
    within this catalogue, so controlling a wrapper combines new controls first.
    Unresolved symbolic widths retain declarations and require concrete binding.
    """
    recipes: dict[FunctionRecord, tuple[FunctionRecord, ControlSpecification]] = {}

    def positive(child: FunctionRecord) -> tuple[ProgramRecord, ControlAdder]:
        if child not in recipes:
            return control_system(child)
        original, previous = recipes[child]
        combined = ControlSpecification(
            ((quantum_bit_type(), np.array(1)), *previous.clauses)
        )
        module = make_program(child.name + "_positive", None, original, combined)
        if isinstance(module, FunctionRecord):
            raise SerializationError(
                "Symbolic controlled child requires concrete binding"
            )
        assert child.registers is not None
        old_names = tuple(r.name for r in child.registers[: len(previous.clauses)])
        root = module.functions[0]
        assert root.registers is not None
        new_names = tuple(r.name for r in root.registers[: len(combined.clauses)])

        def add(
            builder: ComponentProgramBuilder,
            control: BindingCollection,
            values: dict[str, BindingCollection],
        ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
            old_controls = tuple(values.pop(name) for name in old_names)
            inputs = dict(zip(new_names, (control, *old_controls), strict=True))
            outputs = builder.call_tuple(root.name, inputs | values)
            return outputs[0], outputs[1:]

        return module, add

    def make_program(
        name: str,
        registers: tuple[Register, ...] | None,
        child: FunctionRecord,
        spec: ControlSpecification,
    ) -> FunctionRecord | ProgramRecord:
        if child.registers is None or not spec.clauses:
            raise SerializationError(
                "ControlledViaAnd requires typed child and controls"
            )
        signature = quantum_control_signature(
            Signature(child.registers),
            tuple((dtype, values.shape) for dtype, values in spec.clauses),
        )
        if registers is not None and registers != tuple(signature):
            raise SerializationError("Invalid register signature for ControlledViaAnd")
        if any(register.is_symbolic() for register in signature):
            result: FunctionRecord | ProgramRecord = FunctionRecord(
                name, (), registers=tuple(signature)
            )
        else:
            module, adder = positive(child)
            if any(function.name == child.name for function in module.functions):
                raise SerializationError(
                    "Positive-control definitions conflict with the original child"
                )
            # The predicate owner traverses typed calls. Its explicit adder
            # replaces this declaration call, including primitive-only children.
            entry = child.name + "_control_entry"
            builder = ComponentProgramBuilder(
                Signature(child.registers),
                (replace(child, blocks=()), *module.functions),
                name=entry,
            )
            record = builder.finalize(builder.call_dict(child.name, builder.inputs))
            controls = tuple(signature)[: len(spec.clauses)]
            controlled = control_component_via_predicate(
                record,
                {entry: adder},
                controls,
                control_registers_from_spec(spec.clauses),
                entry,
            )
            result = _rename_program(
                replace(
                    controlled,
                    functions=tuple(
                        f for f in controlled.functions if f.name != child.name
                    ),
                ),
                name,
            )
        root = result.functions[0] if isinstance(result, ProgramRecord) else result
        recipes[root] = child, spec
        return result

    def factory(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        if (
            set(arguments) != {"subbloq", "ctrl_spec"}
            or not isinstance(arguments["subbloq"], FunctionRecord)
            or not isinstance(arguments["ctrl_spec"], ControlSpecification)
        ):
            raise SerializationError(
                "ControlledViaAnd requires a resolved child and ctrl_spec"
            )
        try:
            return make_program(
                name, registers, arguments["subbloq"], arguments["ctrl_spec"]
            )
        except ValueError as error:
            raise SerializationError(f"Invalid ControlledViaAnd: {error}") from error

    return {"qualtran.bloqs.mcmt.controlled_via_and.ControlledViaAnd": factory}
