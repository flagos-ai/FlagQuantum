# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow locked Qualtran v0.7.0 select_hubbard.py.

"""Resolve Hubbard selection through shared typed call and scalar owners."""

from collections.abc import Mapping
from functools import partial
from typing import Literal

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.registers import Register
from ...errors import SerializationError
from ..registers.control.hubbard_select_component import hubbard_select_program
from ..registers.layout.hubbard_signatures import hubbard_signature
from ..registers.lookup.hubbard_majorana_component import hubbard_majorana_program
from ..registers.lookup.hubbard_spin_up_component import hubbard_spin_up_program
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_modular_addition_factories import _integer
from .legacy_word_factories import _rename_program

_Kind = Literal["select", "majorana", "spin_up"]


def _program(
    kind: _Kind, arguments: Mapping[str, ResolvedArgument], *, adjoint: bool
) -> ProgramRecord:
    allowed = {"x_dim", "y_dim", "control_val"} | (
        {"gate"} if kind == "majorana" else set()
    )
    if not {"x_dim", "y_dim"} <= arguments.keys() or arguments.keys() - allowed:
        raise SerializationError("Invalid Hubbard selection constructor arguments")
    x, y = (_integer(arguments, key) for key in ("x_dim", "y_dim"))
    if x != y:
        raise NotImplementedError("Currently only supports the case where x_dim=y_dim.")
    cv = arguments.get("control_val")
    if cv is not None and (type(cv) is not int or cv not in (0, 1)):
        raise SerializationError("Hubbard control must be zero, one or None")
    if kind == "select" and cv == 0:
        raise NotImplementedError(
            "control_val=0 not supported, use an outer controlled operation"
        )
    signature = hubbard_signature(kind, x, controlled=cv is not None)
    pauli = arguments.get("gate", "Y")
    if (
        type(x) is not int
        or x == 1
        or kind == "majorana"
        and (cv == 0 or pauli not in ("X", "Y"))
    ):
        return ProgramRecord((FunctionRecord("main", (), registers=tuple(signature)),))
    assert isinstance(x, int)
    if kind == "select":
        return hubbard_select_program(x, controlled=cv is not None, adjoint=adjoint)
    if kind == "majorana":
        assert pauli in ("X", "Y")
        assert isinstance(pauli, str)
        return hubbard_majorana_program(
            x, "X" if pauli == "X" else "Y", controlled=cv is not None, adjoint=adjoint
        )
    # The original spin-up body tests presence, not polarity, of control_val.
    # Preserve its positive body for zero, separately from the native validator.
    return hubbard_spin_up_program(x, controlled=cv is not None, adjoint=adjoint)


def hubbard_selection_factories() -> dict[str, ComponentFactory]:
    """Scope exact constructor inverse recipes to one library resolution."""
    recipes: dict[
        FunctionRecord, tuple[_Kind, Mapping[str, ResolvedArgument], bool]
    ] = {}

    def make(
        kind: _Kind,
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
        *,
        adjoint: bool = False,
    ) -> ProgramRecord:
        record = _rename_program(_program(kind, arguments, adjoint=adjoint), name)
        root = record.functions[0]
        if registers is not None and registers != root.registers:
            raise SerializationError("Invalid Hubbard selection register signature")
        recipes[root] = kind, dict(arguments), adjoint
        return record

    def inverse(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        child = arguments.get("subbloq")
        if (
            set(arguments) != {"subbloq"}
            or not isinstance(child, FunctionRecord)
            or child not in recipes
        ):
            raise SerializationError("Missing exact Hubbard selection adjoint context")
        kind, metadata, direction = recipes[child]
        return make(kind, name, registers, metadata, adjoint=not direction)

    prefix = "qualtran.bloqs.chemistry.hubbard_model.qubitization.select_hubbard."
    return {
        prefix + "SelectHubbard": partial(make, "select"),
        prefix + "HubbardMajorannaOperator": partial(make, "majorana"),
        prefix + "HubbardSpinUpZ": partial(make, "spin_up"),
        "qualtran._infra.adjoint.Adjoint": inverse,
    }
