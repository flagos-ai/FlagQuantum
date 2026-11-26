# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow locked Qualtran v0.7.0 chemistry/df source files.

"""Translate original DF constructors through native register and child owners."""

from collections.abc import Mapping
from functools import partial

from ..algorithms.primitives.double_factorization_encoding_model import (
    source_double_factorization_block_signature,
    source_double_factorization_one_body_signature,
)
from ..algorithms.primitives.double_factorization_prepare_model import (
    source_double_factorization_indexed_data_signature,
    source_double_factorization_inner_prepare_signature,
    source_double_factorization_outer_prepare_signature,
)
from ..algorithms.primitives.double_factorization_rotation_model import (
    source_double_factorization_programmed_rotation_signature,
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
from ..errors import SerializationError
from ._nu_preparation_library import _finite

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


def _leaf_signature(kind: str, values: Mapping[str, int]) -> Signature:
    if kind == "OuterPrepareDoubleFactorization":
        return source_double_factorization_outer_prepare_signature(values["num_aux"])
    spin, eig = values["num_spin_orb"], values["num_eig"]
    if kind == "InnerPrepareDoubleFactorization":
        return source_double_factorization_inner_prepare_signature(
            spin, eig, values["num_bits_rot_aa"]
        )
    if kind == "OutputIndexedData":
        return source_double_factorization_indexed_data_signature(
            values["num_aux"], spin, eig, values["num_bits_rot_aa"]
        )
    return source_double_factorization_programmed_rotation_signature(
        spin, eig, values["num_bits_rot"]
    )


def _one_body(values: Mapping[str, int]) -> ProgramRecord:
    spin, eig, rotation = (
        values["num_spin_orb"],
        values["num_eig"],
        values["num_bits_rot_aa"],
    )
    signature = source_double_factorization_one_body_signature(
        spin, eig, values["num_bits_rot"], rotation
    )
    if spin // 2 == 1:
        return _declaration("one_body", signature)
    inner = source_double_factorization_inner_prepare_signature(spin, eig, rotation)
    basis = source_double_factorization_programmed_rotation_signature(
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
    )


def _block(values: Mapping[str, int]) -> ProgramRecord:
    aux, spin, eig = values["num_aux"], values["num_spin_orb"], values["num_eig"]
    signature = source_double_factorization_block_signature(
        aux, spin, eig, values["num_bits_rot"], values["num_bits_rot_aa_inner"]
    )
    width = (spin // 2 - 1).bit_length()
    if width == 0:
        return _declaration("block", signature)
    outer = source_double_factorization_outer_prepare_signature(aux)
    indexed = source_double_factorization_indexed_data_signature(
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
    )


def _constructor(
    kind: str,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> ProgramRecord:
    required, defaults = _FIELDS[kind]
    if (
        not required <= arguments.keys()
        or arguments.keys() - required - defaults.keys()
    ):
        raise SerializationError("Invalid double-factorization constructor fields")
    raw: dict[str, ResolvedArgument] = {**defaults, **arguments}
    values = {field: _finite(raw, field) for field in raw}
    if kind == "DoubleFactorizationOneBody":
        program = _one_body(values)
    elif kind == "DoubleFactorizationBlockEncoding":
        program = _block(values)
    else:
        program = _declaration(name, _leaf_signature(kind, values))
    if registers is not None and registers != program.functions[0].registers:
        raise SerializationError(
            "Invalid double-factorization source register signature"
        )
    return _rename_program(program, name)


def double_factorization_factories() -> dict[str, ComponentFactory]:
    """Bind six source identifiers without supplying absent physical leaf bodies."""
    return {
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
