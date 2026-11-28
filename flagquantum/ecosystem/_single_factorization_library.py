# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow locked Qualtran v0.7.0 chemistry/sf source files.

"""Translate original SF metadata and preserve its literal, partly opaque roots."""

from collections.abc import Mapping
from functools import partial

from ..algorithms.primitives.single_factorization_encoding_model import (
    source_single_factorization_block_signature,
    source_single_factorization_one_body_signature,
)
from ..algorithms.primitives.single_factorization_resource_model import (
    source_single_factorization_inner_prepare_signature,
    source_single_factorization_outer_prepare_signature,
    source_single_factorization_select_signature,
)
from ..compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    ResolvedArgument,
)
from ..compiler.formats.legacy_exchange_factories import exchange_factories
from ..compiler.formats.legacy_fixed_gate_factories import fixed_gate_factories
from ..compiler.formats.legacy_modular_addition_factories import _integer
from ..compiler.formats.legacy_word_factories import _rename_program
from ..compiler.registers.preparation.identity_preparation_component import (
    identity_preparation_program,
)
from ..compiler.registers.preparation.prepared_reflection_component import (
    prepared_reflection_program,
)
from ..compiler.registers.preparation.single_factorization_component import (
    single_factorization_block_program,
    single_factorization_one_body_program,
)
from ..core.ir.program import FunctionRecord, ProgramRecord
from ..core.ir.program.register_types import opaque_register_type
from ..core.ir.program.registers import Register
from ..core.ir.program.signatures import Signature
from ..core.value_math import SymbolicInt
from ..errors import SerializationError
from ._nu_preparation_library import _finite
from ._source_constructor_adjoints import constructor_adjoint_factories

_BASE = "qualtran.bloqs.chemistry.sf."
_FIELDS = {
    "InnerPrepareSingleFactorization": (
        {"num_aux", "num_spin_orb", "num_bits_state_prep"},
        {"num_bits_rot_aa": 8, "kp1": 1, "kp2": 1},
    ),
    "OuterPrepareSingleFactorization": (
        {"num_aux", "num_bits_state_prep"},
        {"num_bits_rot_aa": 8},
    ),
    "SelectSingleFactorization": ({"num_spin_orb"}, {"additional_control": 0}),
    "SingleFactorizationOneBody": (
        {"num_aux", "num_spin_orb", "num_bits_state_prep"},
        {
            "num_bits_rot_aa": 8,
            "is_adjoint": 0,
            "kp1": 1,
            "kp2": 1,
            "kp1_inv": 1,
            "kp2_inv": 1,
        },
    ),
    "SingleFactorizationBlockEncoding": (
        {"num_aux", "num_spin_orb", "num_bits_state_prep"},
        {
            "num_bits_rot_aa_outer": 8,
            "num_bits_rot_aa_inner": 8,
            "kp1": 1,
            "kp2": 1,
            "kp1_inv": 1,
            "kp2_inv": 1,
        },
    ),
}


def _declaration(name: str, signature: Signature) -> ProgramRecord:
    return ProgramRecord((FunctionRecord(name, (), registers=tuple(signature)),))


def _one_body(values: Mapping[str, SymbolicInt]) -> ProgramRecord:
    aux, spin = _finite(values, "num_aux"), _finite(values, "num_spin_orb")
    signature = source_single_factorization_one_body_signature(aux, spin)
    width = (spin // 2 - 1).bit_length()
    if width == 0 or aux == 0:
        return _declaration("one_body", signature)
    inner = source_single_factorization_inner_prepare_signature(aux, spin)
    hadamard = fixed_gate_factories()["qualtran.bloqs.basic_gates.hadamard.Hadamard"](
        "hadamard", None, {}
    )
    swap = exchange_factories()["qualtran.bloqs.basic_gates.swap.CSwap"](
        "swap", None, {"bitsize": width}
    )
    return single_factorization_one_body_program(
        signature,
        {
            "prepare": _declaration("inner", inner),
            "unprepare": _declaration("inner_adjoint", inner),
            "select": _declaration(
                "select", source_single_factorization_select_signature(spin)
            ),
            "hadamard": (
                hadamard
                if isinstance(hadamard, ProgramRecord)
                else ProgramRecord((hadamard,))
            ),
            "swap": swap if isinstance(swap, ProgramRecord) else ProgramRecord((swap,)),
        },
    )


def _block(
    values: Mapping[str, SymbolicInt], *, adjoint: bool = False
) -> ProgramRecord:
    aux, spin = _finite(values, "num_aux"), _finite(values, "num_spin_orb")
    signature = source_single_factorization_block_signature(aux, spin)
    width = (spin // 2 - 1).bit_length()
    if width == 0 or aux == 0:
        return _declaration("block", signature)
    selections = tuple(
        Register(f"reg{i}_", opaque_register_type(n))
        for i, n in enumerate((width, width, 1, 1))
    )
    identity = identity_preparation_program(selections)
    reflection = prepared_reflection_program(
        identity,
        identity,
        tuple(r.name for r in selections),
        control_value=1,
        global_phase=-1,
    )
    outer = source_single_factorization_outer_prepare_signature(aux)
    body_values = {**values, "num_bits_rot_aa": values["num_bits_rot_aa_inner"]}
    return single_factorization_block_program(
        signature,
        {
            "prepare": _declaration("outer", outer),
            "unprepare": _declaration("outer_adjoint", outer),
            "one_body": _one_body(body_values),
            "reflection": reflection,
        },
        adjoint=adjoint,
    )


def _constructor(
    kind: str,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
    *,
    adjoint: bool = False,
) -> FunctionRecord | ProgramRecord:
    required, defaults = _FIELDS[kind]
    if (
        not required <= arguments.keys()
        or arguments.keys() - required - defaults.keys()
    ):
        raise SerializationError("Invalid single-factorization constructor fields")
    raw: dict[str, ResolvedArgument] = {**defaults, **arguments}
    values = {field: _integer(raw, field) for field in raw}
    spin = _finite(values, "num_spin_orb") if "num_spin_orb" in values else 0
    aux = _finite(values, "num_aux") if "num_aux" in values else 0
    if kind == "SingleFactorizationOneBody":
        program = _one_body(values)
    elif kind == "SingleFactorizationBlockEncoding":
        program = _block(values, adjoint=adjoint)
    else:
        if kind == "InnerPrepareSingleFactorization":
            signature = source_single_factorization_inner_prepare_signature(aux, spin)
        elif kind == "OuterPrepareSingleFactorization":
            signature = source_single_factorization_outer_prepare_signature(aux)
        else:
            control = values["additional_control"]
            if control not in (0, 1):
                raise SerializationError(
                    "Single-factorization SELECT control must be binary"
                )
            signature = source_single_factorization_select_signature(
                spin, additional_control=bool(control)
            )
        program = _declaration(name, signature)
    if registers is not None and registers != program.functions[0].registers:
        raise SerializationError(
            "Invalid single-factorization source register signature"
        )
    return _rename_program(program, name)


def single_factorization_factories() -> dict[str, ComponentFactory]:
    """Resolve all five original identifiers through their native signature owners."""
    owners = {
        _BASE + module + "." + kind: partial(_constructor, kind)
        for module, kinds in (
            (
                "prepare",
                ("InnerPrepareSingleFactorization", "OuterPrepareSingleFactorization"),
            ),
            ("select_bloq", ("SelectSingleFactorization",)),
            (
                "single_factorization",
                ("SingleFactorizationOneBody", "SingleFactorizationBlockEncoding"),
            ),
        )
        for kind in kinds
    }

    inverses = {
        key: partial(factory.func, *factory.args, **factory.keywords, adjoint=True)
        for key, factory in owners.items()
    }
    return constructor_adjoint_factories(owners, inverses)
