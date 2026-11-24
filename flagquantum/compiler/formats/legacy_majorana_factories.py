# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow locked Qualtran v0.7.0 selected_majorana_fermion.py.

"""Resolve Majorana metadata with explicitly owned gate and inverse programs."""

from collections.abc import Callable, Mapping

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import opaque_register_type, quantum_bit_type
from ...core.ir.program.registers import Register
from ...errors import SerializationError
from ..registers.lookup.majorana_component import majorana_program
from .legacy_component_arguments import ExternalGatePayload
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_word_factories import _rename_program

_GatePair = tuple[ProgramRecord, ProgramRecord]
MajoranaGateProvider = Callable[
    [ExternalGatePayload | None, int], tuple[_GatePair, _GatePair]
]


def _registers(value: object) -> tuple[Register, ...]:
    values = (value,) if isinstance(value, Register) else value
    if not isinstance(values, (tuple, list)) or not all(
        isinstance(reg, Register) for reg in values
    ):
        raise SerializationError("Majorana requires typed register metadata")
    return tuple(reg for reg in values if isinstance(reg, Register))


def majorana_factories(
    *,
    gate_programs: MajoranaGateProvider,
    adjoint_factory: ComponentFactory | None = None,
) -> dict[str, ComponentFactory]:
    """Keep gate payloads opaque and compose native through-word dependencies.

    The provider returns controlled-target and accumulator-initializer pairs.
    SDK decoding belongs to its Ecosystem owner. Separate control registers
    retain an explicit source dirty-accumulator boundary rather than being
    silently combined into a corrected source operation.
    """
    recipes: dict[
        FunctionRecord,
        tuple[
            tuple[Register, ...], tuple[Register, ...], ExternalGatePayload | None, bool
        ],
    ] = {}

    def make(
        name: str,
        registers: tuple[Register, ...] | None,
        selections: tuple[Register, ...],
        controls: tuple[Register, ...],
        payload: ExternalGatePayload | None,
        inverse: bool,
    ) -> ProgramRecord:
        count = 1
        for reg in selections:
            bound = dict(reg.dtype.parameters).get("bound")
            if reg.dtype.encoding != "uint" or type(bound) is not int or bound < 1:
                raise SerializationError(
                    "Majorana requires concrete bounded iteration extents"
                )
            count *= bound
        ports = (
            *controls,
            *selections,
            Register("target", opaque_register_type(count)),
        )
        if registers is not None and registers != ports:
            raise SerializationError("Invalid Majorana register signature")
        if len(controls) > 1:
            raise SerializationError(
                "Source Majorana separate controls can leave a dirty accumulator"
            )
        if any(reg.is_symbolic() for reg in ports):
            record = ProgramRecord((FunctionRecord("main", (), registers=ports),))
        else:
            control = controls[0] if controls else None
            size = int(control.total_bits()) if control is not None else 0
            target, initial = gate_programs(payload, size)
            record = majorana_program(
                selections,
                control=control,
                adjoint=inverse,
                target_pair=target,
                initial_pair=initial,
            )
        record = _rename_program(record, name)
        recipes[record.functions[0]] = selections, controls, payload, inverse
        return record

    def forward(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        if "selection_regs" not in arguments or arguments.keys() - {
            "selection_regs",
            "control_regs",
            "target_gate",
        }:
            raise SerializationError(
                "Majorana requires registers and an owned gate payload"
            )
        payload = arguments.get("target_gate")
        if "target_gate" in arguments and not isinstance(payload, ExternalGatePayload):
            raise SerializationError("Majorana target requires an owned gate payload")
        assert payload is None or isinstance(payload, ExternalGatePayload)
        return make(
            name,
            registers,
            _registers(arguments["selection_regs"]),
            _registers(
                arguments.get(
                    "control_regs", (Register("control", quantum_bit_type()),)
                )
            ),
            payload,
            False,
        )

    def adjoint(
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
            selections, controls, payload, inverse = recipes[child]
            return make(name, registers, selections, controls, payload, not inverse)
        if adjoint_factory is not None:
            return adjoint_factory(name, registers, arguments)
        raise SerializationError("Missing Majorana adjoint constructor context")

    return {
        "qualtran.bloqs.multiplexers.selected_majorana_fermion.SelectedMajoranaFermion": forward,
        "qualtran._infra.adjoint.Adjoint": adjoint,
    }
