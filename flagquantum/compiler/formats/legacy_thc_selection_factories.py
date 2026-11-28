# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow locked Qualtran v0.7.0 chemistry/thc/select_bloq.py.

"""Resolve literal THC selection without supplying an invented rotation body."""

from collections.abc import Mapping

import sympy

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import quantum_bit_type
from ...core.ir.program.registers import Register
from ...errors import SerializationError
from ..registers.component_control import quantum_control_signature
from ..registers.control.thc_selection_component import thc_selection_program
from ..registers.layout.thc_signatures import (
    thc_rotation_signature,
    thc_selection_signature,
)
from ..registers.primitives.controlled_z_word_component import (
    controlled_z_word_program,
    controlled_z_word_signature,
)
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_exchange_factories import exchange_factories
from .legacy_modular_addition_factories import _integer
from .legacy_multi_control_factories import _literal_values
from .legacy_word_factories import _rename_program


def _binary(arguments: Mapping[str, ResolvedArgument], key: str) -> bool:
    value = arguments.get(key, False)
    if type(value) not in (int, bool) or value not in (0, 1):
        raise SerializationError(f"THC {key} requires a binary flag")
    return bool(value)


def _ports(
    name: str, registers: tuple[Register, ...] | None, ports: tuple[Register, ...]
) -> FunctionRecord:
    if registers is not None and registers != ports:
        raise SerializationError("Invalid literal THC register signature")
    return FunctionRecord(name, (), registers=ports)


def _rotations(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord:
    required = {"num_mu", "num_spin_orb", "num_bits_theta"}
    if not required <= arguments.keys() or arguments.keys() - (
        required | {"kr1", "kr2", "two_body_only", "is_adjoint"}
    ):
        raise SerializationError("Invalid THC rotation constructor arguments")
    m = _integer(arguments, "num_mu")
    if type(m) is not int:
        raise SerializationError("THC address dimension requires concrete bit_length")
    assert isinstance(m, int)
    spin, theta = (
        _integer(arguments, key) for key in ("num_spin_orb", "num_bits_theta")
    )
    for key in ("kr1", "kr2"):
        _integer({key: arguments.get(key, 1)}, key)
    _binary(arguments, "two_body_only")
    _binary(arguments, "is_adjoint")
    return _ports(name, registers, tuple(thc_rotation_signature(m, spin, theta)))


def _phase(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    if set(arguments) != {"cvs", "bitsize"}:
        raise SerializationError("Controlled Z word requires cvs and bitsize")
    raw = arguments["cvs"]
    values = _literal_values((raw,) if type(raw) is int else raw)
    width = _integer(arguments, "bitsize")
    ports = tuple(controlled_z_word_signature(width, len(values)))
    declaration = _ports(name, registers, ports)
    if not values or isinstance(width, sympy.Expr) or width == 0:
        return declaration
    assert isinstance(width, int)
    return _rename_program(controlled_z_word_program(width, values), name)


def _selection(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
    *,
    adjoint: bool = False,
    generic_control: bool = False,
) -> FunctionRecord | ProgramRecord:
    required = {"num_mu", "num_spin_orb", "num_bits_theta", "keep_bitsize"}
    if not required <= arguments.keys() or arguments.keys() - (
        required | {"kr1", "kr2", "control_val"}
    ):
        raise SerializationError("Invalid THC SELECT constructor arguments")
    m = _integer(arguments, "num_mu")
    if type(m) is not int:
        raise SerializationError("THC address dimension requires concrete bit_length")
    assert isinstance(m, int)
    spin = _integer(arguments, "num_spin_orb")
    keep = _integer(arguments, "keep_bitsize")
    _integer(arguments, "num_bits_theta")
    control = arguments.get("control_val")
    if control is not None and (type(control) is not int or control not in (0, 1)):
        raise SerializationError("THC SELECT control must be zero, one or None")
    assert control is None or isinstance(control, int)
    signature = thc_selection_signature(m, spin, keep, control_value=control)
    width = spin // 2
    if m == 0 or width == 0:
        if generic_control:
            signature = quantum_control_signature(
                signature, ((quantum_bit_type(), ()),)
            )
        return _ports(name, registers, tuple(signature))
    _ports(
        name,
        registers,
        tuple(
            quantum_control_signature(signature, ((quantum_bit_type(), ()),))
            if generic_control
            else signature
        ),
    )
    rotation_args = {
        key: arguments[key] for key in ("num_mu", "num_spin_orb", "num_bits_theta")
    }
    rotation_args.update({key: arguments.get(key, 1) for key in ("kr1", "kr2")})
    rotations = tuple(
        _rotations(
            f"thc_rotation_{int(two)}_{int(inverse)}",
            None,
            {**rotation_args, "two_body_only": two, "is_adjoint": inverse},
        )
        for two, inverse in ((False, False), (False, True), (True, False), (True, True))
    )
    c_swap = exchange_factories()["qualtran.bloqs.basic_gates.swap.CSwap"]

    def swap(label: str, bits: int | sympy.Expr) -> ProgramRecord:
        result = c_swap(label, None, {"bitsize": bits})
        return result if isinstance(result, ProgramRecord) else ProgramRecord((result,))

    phase1 = _phase("thc_phase_1", None, {"cvs": (1,), "bitsize": width})
    phase2 = _phase("thc_phase_10", None, {"cvs": (1, 0), "bitsize": width})
    assert len(rotations) == 4
    program = thc_selection_program(
        signature,
        (rotations[0], rotations[1], rotations[2], rotations[3]),
        swap("thc_swap_system", width),
        swap("thc_swap_flags", 1),
        swap("thc_swap_addresses", m.bit_length()),
        (
            phase1 if isinstance(phase1, ProgramRecord) else ProgramRecord((phase1,)),
            phase2 if isinstance(phase2, ProgramRecord) else ProgramRecord((phase2,)),
        ),
        adjoint=adjoint,
        generic_control=generic_control,
    )
    return _rename_program(program, name)


def thc_selection_factories(
    *, adjoint_factory: ComponentFactory | None = None
) -> dict[str, ComponentFactory]:
    """Bind both THC classes and their direct first-system-bit phase wrapper."""
    pairs: dict[FunctionRecord, tuple[ProgramRecord, ProgramRecord]] = {}

    def store(
        name: str,
        registers: tuple[Register, ...] | None,
        pair: tuple[ProgramRecord, ProgramRecord],
    ) -> ProgramRecord:
        forward, inverse = pair
        if registers is not None and registers != forward.functions[0].registers:
            raise SerializationError("Invalid literal THC SELECT signature")
        result = _rename_program(forward, name)
        pairs[result.functions[0]] = result, inverse
        return result

    def selection(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        forward = _selection(name, registers, arguments)
        if isinstance(forward, FunctionRecord):
            return forward
        inverse = _selection(name + "_inverse", None, arguments, adjoint=True)
        assert isinstance(inverse, ProgramRecord)
        return store(name, registers, (forward, inverse))

    def adjoint(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        child = arguments.get("subbloq")
        if (
            arguments.keys() == {"subbloq"}
            and isinstance(child, FunctionRecord)
            and child in pairs
        ):
            forward, inverse = pairs[child]
            return store(name, registers, (inverse, forward))
        if adjoint_factory is not None:
            return adjoint_factory(name, registers, arguments)
        raise SerializationError("Missing literal THC SELECT inverse context")

    prefix = "qualtran.bloqs.chemistry."
    return {
        prefix + "thc.select_bloq.THCRotations": _rotations,
        prefix + "thc.select_bloq.SelectTHC": selection,
        prefix + "black_boxes.ApplyControlledZs": _phase,
        "qualtran._infra.adjoint.Adjoint": adjoint,
    }
