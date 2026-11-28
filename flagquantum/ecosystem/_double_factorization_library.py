# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow locked Qualtran v0.7.0 chemistry/df source files.

"""Translate original DF constructors through native register and child owners."""

from collections.abc import Mapping
from functools import partial

from ..algorithms.primitives.double_factorization_encoding_model import (
    _double_factorization_block_declaration,
    _double_factorization_one_body_declaration,
)
from ..algorithms.primitives.double_factorization_prepare_model import (
    _double_factorization_indexed_declaration,
    _double_factorization_inner_declaration,
    source_double_factorization_outer_prepare_signature,
)
from ..algorithms.primitives.double_factorization_rotation_model import (
    _double_factorization_rotation_declaration,
)
from ..algorithms.primitives.prepared_state_reflection import (
    PreparedStateReflection,
    zero_state_reflection_program,
)
from ..compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    ResolvedArgument,
)
from ..compiler.formats.legacy_exchange_factories import exchange_factories
from ..compiler.formats.legacy_fixed_gate_factories import fixed_gate_factories
from ..compiler.formats.legacy_modular_addition_factories import _integer
from ..compiler.formats.legacy_word_factories import _rename_program
from ..compiler.registers.preparation.double_factorization_component import (
    double_factorization_block_program,
    double_factorization_one_body_program,
)
from ..compiler.registers.primitives.controlled_z_word_component import (
    controlled_z_word_program,
)
from ..core.ir.program import FunctionRecord, ProgramRecord
from ..core.ir.program.registers import Register
from ..core.ir.program.signatures import Signature
from ..core.value_math import SymbolicInt
from ..errors import SerializationError
from ._nu_preparation_library import _finite
from ._source_constructor_adjoints import constructor_adjoint_factories

_BASE = "qualtran.bloqs.chemistry.df."
_FIELDS = {
    "InnerPrepareDoubleFactorization": (
        {
            "num_aux",
            "num_spin_orb",
            "num_eig",
            "num_bits_rot_aa",
            "num_bits_state_prep",
        },
        {},
    ),
    "OuterPrepareDoubleFactorization": (
        {"num_aux", "num_bits_state_prep"},
        {"num_bits_rot_aa": 8},
    ),
    "OutputIndexedData": (
        {"num_aux", "num_spin_orb", "num_eig"},
        {"num_bits_rot_aa": 8},
    ),
    "ProgRotGateArray": ({"num_aux", "num_spin_orb", "num_eig", "num_bits_rot"}, {}),
    "DoubleFactorizationOneBody": (
        {"num_aux", "num_spin_orb", "num_eig", "num_bits_state_prep"},
        {"num_bits_rot_aa": 8, "num_bits_rot": 24},
    ),
    "DoubleFactorizationBlockEncoding": (
        {"num_spin_orb", "num_aux", "num_eig"},
        {
            "num_bits_state_prep": 8,
            "num_bits_rot": 24,
            "num_bits_rot_aa_outer": 8,
            "num_bits_rot_aa_inner": 8,
        },
    ),
}


def _declaration(name: str, signature: Signature) -> ProgramRecord:
    return ProgramRecord((FunctionRecord(name, (), registers=tuple(signature)),))


def _leaf_signature(kind: str, values: Mapping[str, SymbolicInt]) -> Signature:
    if kind == "OuterPrepareDoubleFactorization":
        return source_double_factorization_outer_prepare_signature(
            _finite(values, "num_aux")
        )
    spin, eig = _finite(values, "num_spin_orb"), _finite(values, "num_eig")
    if kind == "InnerPrepareDoubleFactorization":
        return _double_factorization_inner_declaration(
            spin, eig, values["num_bits_rot_aa"]
        )
    if kind == "OutputIndexedData":
        return _double_factorization_indexed_declaration(
            _finite(values, "num_aux"), spin, eig, values["num_bits_rot_aa"]
        )
    return _double_factorization_rotation_declaration(spin, eig, values["num_bits_rot"])


def _one_body(
    values: Mapping[str, SymbolicInt], *, adjoint: bool = False
) -> ProgramRecord:
    spin, eig, rotation = (
        _finite(values, "num_spin_orb"),
        _finite(values, "num_eig"),
        values["num_bits_rot_aa"],
    )
    signature = _double_factorization_one_body_declaration(
        spin, eig, values["num_bits_rot"], rotation
    )
    if spin // 2 == 1 or rotation == 0:
        return _declaration("one_body", signature)
    inner = _double_factorization_inner_declaration(spin, eig, rotation)
    basis = _double_factorization_rotation_declaration(
        spin, eig, values["num_bits_rot"]
    )
    hadamard = fixed_gate_factories()["qualtran.bloqs.basic_gates.hadamard.Hadamard"](
        "hadamard", None, {}
    )
    swap = exchange_factories()["qualtran.bloqs.basic_gates.swap.CSwap"](
        "swap", None, {"bitsize": spin // 2}
    )
    return double_factorization_one_body_program(
        signature,
        {
            "prepare": _declaration("inner", inner),
            "unprepare": _declaration("inner_adjoint", inner),
            "rotate": _declaration("rotate", basis),
            "unrotate": _declaration("rotate_adjoint", basis),
            "hadamard": (
                hadamard
                if isinstance(hadamard, ProgramRecord)
                else ProgramRecord((hadamard,))
            ),
            "swap": swap if isinstance(swap, ProgramRecord) else ProgramRecord((swap,)),
            "phase": controlled_z_word_program(spin // 2, (1, 1)),
        },
        adjoint=adjoint,
    )


def _block(
    values: Mapping[str, SymbolicInt], *, adjoint: bool = False
) -> ProgramRecord:
    aux, spin, eig = (
        _finite(values, field) for field in ("num_aux", "num_spin_orb", "num_eig")
    )
    signature = _double_factorization_block_declaration(
        aux, spin, eig, values["num_bits_rot"], values["num_bits_rot_aa_inner"]
    )
    width = (spin // 2 - 1).bit_length()
    if width == 0 or aux == 0 or values["num_bits_rot_aa_inner"] == 0:
        return _declaration("block", signature)
    outer = source_double_factorization_outer_prepare_signature(aux)
    indexed = _double_factorization_indexed_declaration(
        aux, spin, eig, values["num_bits_rot_aa_inner"]
    )
    return double_factorization_block_program(
        signature,
        {
            "prepare": _declaration("outer", outer),
            "unprepare": _declaration("outer_adjoint", outer),
            "load": _declaration("indexed", indexed),
            "unload": _declaration("indexed_adjoint", indexed),
            "one_body": _one_body(
                {**values, "num_bits_rot_aa": values["num_bits_rot_aa_inner"]}
            ),
            "reflection": zero_state_reflection_program(
                PreparedStateReflection.around_zero(
                    (width, 1), control_value=1, global_phase=-1
                )
            ),
        },
        one_body_inverse=(
            _one_body(
                {**values, "num_bits_rot_aa": values["num_bits_rot_aa_inner"]},
                adjoint=True,
            )
            if adjoint
            else None
        ),
    )


def _constructor(
    kind: str,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
    *,
    adjoint: bool = False,
) -> ProgramRecord:
    required, defaults = _FIELDS[kind]
    if (
        not required <= arguments.keys()
        or arguments.keys() - required - defaults.keys()
    ):
        raise SerializationError("Invalid double-factorization constructor fields")
    raw: dict[str, ResolvedArgument] = {**defaults, **arguments}
    values = {field: _integer(raw, field) for field in raw}
    if kind == "DoubleFactorizationOneBody":
        program = _one_body(values, adjoint=adjoint)
    elif kind == "DoubleFactorizationBlockEncoding":
        program = _block(values, adjoint=adjoint)
    else:
        program = _declaration(name, _leaf_signature(kind, values))
    if registers is not None and registers != program.functions[0].registers:
        raise SerializationError(
            "Invalid double-factorization source register signature"
        )
    return _rename_program(program, name)


def double_factorization_factories() -> dict[str, ComponentFactory]:
    """Bind six source identifiers without supplying absent physical leaf bodies."""
    owners = {
        _BASE + module + "." + kind: partial(_constructor, kind)
        for module, kinds in (
            (
                "prepare",
                (
                    "InnerPrepareDoubleFactorization",
                    "OuterPrepareDoubleFactorization",
                    "OutputIndexedData",
                ),
            ),
            ("select_bloq", ("ProgRotGateArray",)),
            (
                "double_factorization",
                ("DoubleFactorizationOneBody", "DoubleFactorizationBlockEncoding"),
            ),
        )
        for kind in kinds
    }

    inverses = {
        key: partial(factory.func, *factory.args, **factory.keywords, adjoint=True)
        for key, factory in owners.items()
    }
    return constructor_adjoint_factories(owners, inverses)
