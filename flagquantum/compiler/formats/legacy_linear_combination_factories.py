# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructor behavior follows Qualtran v0.7.0 block_encoding/linear_combination.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/block_encoding/linear_combination.py

"""Bind lossy source LCU declarations to explicitly supplied native contexts."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt
from ...errors import SerializationError
from ..registers.component_control import ControlAdder, _stored_control_adder
from ..registers.control.specialized_component_control import (
    multi_controlled_component_pair,
    specialized_component_control_system,
)
from ..registers.preparation.linear_combination_component import (
    linear_combination_component_program,
)
from .legacy_component_arguments import ControlSpecification
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_controlled_factories import controlled_via_and_factories
from .legacy_modular_addition_factories import _integer
from .legacy_multi_control_factories import _literal_values
from .legacy_word_factories import _rename_program

ExactPair = tuple[ProgramRecord, ProgramRecord]


@dataclass(frozen=True)
class LinearCombinationContext:
    """Own explicit native roles that the original producer omits from its wire.

    Binding keys are assigned definition identities, never semantic name
    patterns. The boundary supplying this context owns its correspondence to
    the source object and the semantic validity of both exact child pairs.
    """

    signature: Signature
    constituent_ancilla: SymbolicInt
    constituent_resource: SymbolicInt
    selection: SymbolicInt
    junk: SymbolicInt
    precision: SymbolicInt
    controlled: bool
    prepare_pair: ExactPair
    select_pair: ExactPair

    def program(self, *, adjoint: bool = False) -> ProgramRecord:
        """Compose the shared root; SELECT already has the requested control."""
        return linear_combination_component_program(
            self.signature,
            constituent_ancilla=self.constituent_ancilla,
            constituent_resource=self.constituent_resource,
            prepare_selection=self.selection,
            prepare_junk=self.junk,
            prepare_pair=self.prepare_pair,
            select_pair=self.select_pair,
            adjoint=adjoint,
        )


def linear_combination_factories(
    contexts: Mapping[str, LinearCombinationContext],
    *,
    positive_context: (
        Callable[[LinearCombinationContext], LinearCombinationContext] | None
    ) = None,
    adjoint_factory: ComponentFactory | None = None,
    controlled_factory: ComponentFactory | None = None,
) -> dict[str, ComponentFactory]:
    """Require a scoped context per assigned definition, with no wire guessing.

    The upstream shallow declaration contains only precision and control state.
    Widths, coefficients and child identities cannot be reconstructed from it.
    Unknown bindings fail even when another context has the same signature.
    """
    bindings = dict(contexts)
    recipes: dict[FunctionRecord, tuple[LinearCombinationContext, bool]] = {}
    controls: dict[FunctionRecord, tuple[FunctionRecord, ControlSpecification]] = {}
    specialized: dict[
        FunctionRecord, tuple[FunctionRecord, tuple[int, ...], str, bool]
    ] = {}

    def make(
        name: str,
        registers: tuple[Register, ...] | None,
        context: LinearCombinationContext,
        inverse: bool,
    ) -> ProgramRecord:
        if registers is not None and registers != tuple(context.signature):
            raise SerializationError("Invalid linear-combination context signature")
        record = context.program(adjoint=inverse)
        result = _rename_program(record, name)
        recipes[result.functions[0]] = context, inverse
        return result

    def forward(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        if "lambd_bits" not in arguments or arguments.keys() - {
            "lambd_bits",
            "is_controlled",
        }:
            raise SerializationError("Invalid shallow linear-combination arguments")
        if name not in bindings:
            raise SerializationError(
                "Missing explicit native linear-combination context"
            )
        context = bindings[name]
        precision = _integer(arguments, "lambd_bits")
        control = arguments.get("is_controlled", 0)
        if type(control) not in (int, bool) or control not in (0, 1):
            raise SerializationError("Linear-combination control flag must be binary")
        if precision != context.precision or bool(control) != context.controlled:
            raise SerializationError(
                "Linear-combination context disagrees with source payload"
            )
        return make(name, registers, context, False)

    def adjoint(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        child = arguments.get("subbloq")
        if (
            arguments.keys() == {"subbloq"}
            and isinstance(child, FunctionRecord)
            and child in specialized
        ):
            original, values, control_name, inverse = specialized[child]
            return make_specialized(
                name, registers, original, values, control_name, not inverse
            )
        if (
            arguments.keys() == {"subbloq"}
            and isinstance(child, FunctionRecord)
            and child in recipes
        ):
            context, inverse = recipes[child]
            return make(name, registers, context, not inverse)
        if (
            arguments.keys() == {"subbloq"}
            and isinstance(child, FunctionRecord)
            and child in controls
        ):
            original, spec = controls[child]
            inverse_child = adjoint(
                name + "_child_inverse", None, {"subbloq": original}
            )
            function = (
                inverse_child.functions[0]
                if isinstance(inverse_child, ProgramRecord)
                else inverse_child
            )
            return controlled(name, registers, {"subbloq": function, "ctrl_spec": spec})
        if adjoint_factory is not None:
            return adjoint_factory(name, registers, arguments)
        raise SerializationError(
            "Missing exact native linear-combination adjoint context"
        )

    def positive(child: FunctionRecord) -> tuple[ProgramRecord, ControlAdder]:
        if child not in recipes:
            raise SerializationError(
                "Missing explicit positive linear-combination context"
            )
        context, inverse = recipes[child]
        if context.controlled:
            record = _rename_program(
                context.program(adjoint=inverse), child.name + "_positive"
            )
            return specialized_component_control_system(
                record, "ctrl", 1, 1, entry=record.functions[0].name
            )
        if positive_context is None:
            raise SerializationError(
                "Missing explicit positive linear-combination context"
            )
        selected = positive_context(context)
        if not selected.controlled or tuple(selected.signature)[1:] != tuple(
            context.signature
        ):
            raise SerializationError(
                "Invalid positive linear-combination context signature"
            )
        record = _rename_program(
            selected.program(adjoint=inverse), child.name + "_positive"
        )
        return record, _stored_control_adder(record.functions[0])

    via_and = controlled_via_and_factories(control_system=positive)[
        "qualtran.bloqs.mcmt.controlled_via_and.ControlledViaAnd"
    ]

    def controlled(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        child = arguments.get("subbloq")
        if (
            isinstance(child, FunctionRecord)
            and child not in recipes
            and child not in controls
        ):
            if controlled_factory is not None:
                return controlled_factory(name, registers, arguments)
            raise SerializationError("Missing owned linear-combination control child")
        result = via_and(name, registers, arguments)
        child, spec = arguments.get("subbloq"), arguments.get("ctrl_spec")
        assert isinstance(child, FunctionRecord) and isinstance(
            spec, ControlSpecification
        )
        function = result.functions[0] if isinstance(result, ProgramRecord) else result
        controls[function] = child, spec
        return result

    def make_specialized(
        name: str,
        registers: tuple[Register, ...] | None,
        child: FunctionRecord,
        values: tuple[int, ...],
        control_name: str,
        inverse: bool,
    ) -> ProgramRecord:
        if child not in recipes:
            raise SerializationError(
                "Missing specialized linear-combination child context"
            )
        context, child_inverse = recipes[child]
        if not context.controlled or control_name != "ctrl":
            raise SerializationError(
                "Specialized LCU requires its existing positive ctrl port"
            )
        pair = multi_controlled_component_pair(
            context.program(adjoint=child_inverse),
            context.program(adjoint=not child_inverse),
            control_name,
            values,
        )
        record = pair[int(inverse)]
        if registers is not None and registers != record.functions[0].registers:
            raise SerializationError("Invalid specialized linear-combination signature")
        result = _rename_program(record, name)
        specialized[result.functions[0]] = child, values, control_name, inverse
        return result

    def specialize(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        child = arguments.get("ctrl_bloq")
        control_name = arguments.get("ctrl_reg_name")
        if (
            arguments.keys() != {"ctrl_bloq", "ctrl_reg_name", "cvs"}
            or not isinstance(child, FunctionRecord)
            or not isinstance(control_name, str)
        ):
            raise SerializationError("Invalid specialized linear-combination arguments")
        return make_specialized(
            name,
            registers,
            child,
            _literal_values(arguments["cvs"]),
            control_name,
            False,
        )

    return {
        "qualtran.bloqs.block_encoding.linear_combination.LinearCombination": forward,
        "qualtran._infra.adjoint.Adjoint": adjoint,
        "qualtran.bloqs.mcmt.controlled_via_and.ControlledViaAnd": controlled,
        "qualtran.bloqs.mcmt.specialized_ctrl._MultiControlledFromSinglyControlled": specialize,
    }
