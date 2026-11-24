# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow locked Qualtran v0.7.0 reflections source files.

"""Resolve identity PREPARE and explicitly owned prepared-reflection inverses."""

from collections.abc import Callable, Mapping
from dataclasses import replace

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import Register, RegisterSide
from ...errors import SerializationError
from ..registers.component_graph import _namespace_helpers
from ..registers.preparation.identity_preparation_component import (
    identity_preparation_program,
)
from ..registers.preparation.prepared_reflection_component import (
    prepared_reflection_program,
)
from ..registers.primitives.global_phase import global_phase_exponent
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_rotation_factories import _real_argument
from .legacy_scalar_factories import _global_phase_factory
from .legacy_word_factories import _rename_program

PreparationInverse = Callable[[FunctionRecord], tuple[ProgramRecord, tuple[str, ...]]]


def reflection_factories(
    *, preparation_inverse: PreparationInverse | None = None
) -> dict[str, ComponentFactory]:
    """Track identity selection metadata and require exact inverses for other PREPAREs.

    The provider owns selection/junk classification and an exact inverse with
    its complete helper closure. Source Adjoint returns self for reflection,
    even when its coefficient is nonreal; no conjugation is silently added.
    """
    identities: dict[FunctionRecord, tuple[str, ...]] = {}

    def identity(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        if set(arguments) != {"selection_regs"}:
            raise SerializationError("Identity PREPARE requires selection_regs")
        raw = arguments["selection_regs"]
        ports = (raw,) if isinstance(raw, Register) else raw
        if not isinstance(ports, (tuple, list)) or not all(
            isinstance(reg, Register) for reg in ports
        ):
            raise SerializationError(
                "Identity PREPARE requires typed selection registers"
            )
        selections = tuple(reg for reg in ports if isinstance(reg, Register))
        if registers is not None and registers != selections:
            raise SerializationError("Invalid identity PREPARE register signature")
        if any(
            type(reg.bitsize) is not int
            or reg.bitsize == 0
            or reg.shape_symbolic
            or reg.side != RegisterSide.THRU
            or reg.dtype.num_cbits
            for reg in selections
        ):
            result: FunctionRecord | ProgramRecord = FunctionRecord(
                name, (), registers=selections
            )
        else:
            result = _rename_program(identity_preparation_program(selections), name)
        root = result.functions[0] if isinstance(result, ProgramRecord) else result
        identities[root] = tuple(reg.name for reg in selections)
        return result

    def reflection(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        if "prepare_gate" not in arguments or arguments.keys() - {
            "prepare_gate",
            "control_val",
            "global_phase",
            "eps",
        }:
            raise SerializationError(
                "Invalid prepared reflection constructor arguments"
            )
        child = arguments["prepare_gate"]
        if not isinstance(child, FunctionRecord):
            raise SerializationError("Reflection requires a resolved PREPARE function")
        forward = ProgramRecord((replace(child, blocks=()),))
        if child in identities:
            inverse, selections = forward, identities[child]
        elif preparation_inverse is not None:
            inverse, selections = preparation_inverse(child)
        else:
            raise SerializationError(
                "Missing exact PREPARE inverse constructor context"
            )
        control = arguments.get("control_val")
        if control is not None and (type(control) is not int or control not in (0, 1)):
            raise SerializationError("Reflection control must be zero or one")
        coefficient = arguments.get("global_phase", 1)
        if not isinstance(coefficient, (int, float, complex)):
            raise SerializationError("Reflection requires a concrete phase coefficient")
        precision = _real_argument(arguments, "eps", 1e-11)
        scalar = (
            _global_phase_factory(
                "reflection_scalar",
                (),
                {
                    "exponent": global_phase_exponent(complex(coefficient)),
                    "eps": precision,
                },
            )
            if control is None and coefficient != 1
            else None
        )
        record = prepared_reflection_program(
            forward,
            inverse,
            selections,
            control_value=control,
            global_phase=complex(coefficient),
            scalar_phase=scalar,
        )
        if registers is not None and registers != record.functions[0].registers:
            raise SerializationError("Invalid prepared reflection register signature")
        record = replace(
            record,
            functions=(replace(record.functions[0], name=name), *record.functions[1:]),
        )
        return _namespace_helpers(record, name, external=(child.name,))

    return {
        "qualtran.bloqs.reflections.prepare_identity.PrepareIdentity": identity,
        "qualtran.bloqs.reflections.reflection_using_prepare.ReflectionUsingPrepare": reflection,
    }
