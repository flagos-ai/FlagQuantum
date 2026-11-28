# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow locked Qualtran v0.7.0 projectile/select_and_prepare.py.

"""Translate finite projectile roots with literal source controls and lifetimes."""

from collections.abc import Mapping
from functools import partial

import numpy as np

from ..algorithms.first_quantization import _momentum_exchange_declaration
from ..algorithms.first_quantization_projectile import (
    _projectile_prepare_declaration,
    _projectile_select_declaration,
    _projectile_select_t_declaration,
    _projectile_select_uv_declaration,
    projectile_prepare_tuv_signature,
)
from ..algorithms.first_quantization_resource_model import source_prepare_ij_signature
from ..compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    ResolvedArgument,
)
from ..compiler.formats.legacy_modular_addition_factories import _integer
from ..compiler.formats.legacy_word_factories import _rename_program
from ..compiler.registers.component_adjoint import adjoint_component_calls
from ..compiler.registers.control.projectile_selection_component import (
    projectile_selection_program,
)
from ..compiler.registers.layout.momentum_exchange_component import (
    momentum_exchange_program,
)
from ..compiler.registers.preparation.first_quantization_preparation_component import (
    first_quantization_preparation_program,
)
from ..compiler.registers.primitives.controlled_swap_component import (
    controlled_swap_program,
)
from ..core.ir.program import FunctionRecord, ProgramRecord
from ..core.ir.program.register_types import opaque_register_type, quantum_bit_type
from ..core.ir.program.registers import Register
from ..core.ir.program.signatures import Signature
from ..errors import SerializationError
from ._kinetic_preparation_library import _kinetic, kinetic_preparation_factories
from ._nu_preparation_library import _finite
from ._potential_preparation_library import _potential, potential_preparation_factories
from ._source_constructor_adjoints import constructor_adjoint_factories

_BASE = "qualtran.bloqs.chemistry.pbc.first_quantization."
_ROOT_FIELDS = {"num_bits_p", "num_bits_n", "eta", "num_atoms", "lambda_zeta"}
_DEFAULTS = {
    "m_param": 256,
    "num_bits_nuc_pos": 16,
    "num_bits_t": 16,
    "num_bits_rot_aa": 8,
}
_LEAF_FIELDS = {
    "PrepareTUVSuperpositions": {"num_bits_t", "eta", "lambda_zeta"},
    "UniformSuperpostionIJFirstQuantization": {"eta", "num_bits_rot_aa"},
    "SelectTFirstQuantizationWithProj": {"num_bits_n", "eta"},
    "SelectUVFirstQuantizationWithProj": {
        "num_bits_p",
        "num_bits_n",
        "eta",
        "num_atoms",
        "num_bits_nuc_pos",
    },
}


def _finish(
    record: FunctionRecord | ProgramRecord,
    name: str,
    registers: tuple[Register, ...] | None,
) -> FunctionRecord | ProgramRecord:
    root = record.functions[0] if isinstance(record, ProgramRecord) else record
    if registers is not None and registers != root.registers:
        raise SerializationError("Invalid projectile source register signature")
    return (
        _rename_program(record, name) if isinstance(record, ProgramRecord) else record
    )


def _leaf(
    kind: str,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
    *,
    adjoint: bool = False,
) -> FunctionRecord | ProgramRecord:
    required = _LEAF_FIELDS[kind]
    optional = (
        {"num_bits_rot_aa", "is_adjoint"}
        if kind == "PrepareTUVSuperpositions"
        else set()
    )
    if not required <= arguments.keys() or arguments.keys() - required - optional:
        raise SerializationError("Invalid projectile source leaf fields")
    sizes = {field: _integer(arguments, field) for field in required}
    if kind == "PrepareTUVSuperpositions":
        if "num_bits_rot_aa" in arguments:
            _integer(arguments, "num_bits_rot_aa")
        if type(arguments.get("is_adjoint", False)) not in (bool, int):
            raise SerializationError(
                "Projectile term direction requires a boolean or integer"
            )
        signature = projectile_prepare_tuv_signature()
    elif kind == "UniformSuperpostionIJFirstQuantization":
        signature = source_prepare_ij_signature(_finite(arguments, "eta"))
    elif kind == "SelectTFirstQuantizationWithProj":
        signature = _projectile_select_t_declaration(sizes["num_bits_n"])
    else:
        signature = _projectile_select_uv_declaration(
            sizes["num_bits_p"],
            sizes["num_bits_n"],
            _finite(arguments, "num_atoms"),
            sizes["num_bits_nuc_pos"],
        )
    return _finish(
        FunctionRecord(
            name,
            (),
            registers=tuple(signature.adjoint()) if adjoint else tuple(signature),
        ),
        name,
        registers,
    )


def _exchange(
    controlled: bool,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
    *,
    adjoint: bool = False,
) -> FunctionRecord | ProgramRecord:
    required = {"num_bits_p", "eta"} | ({"num_bits_n", "cvs"} if controlled else set())
    if arguments.keys() != required:
        raise SerializationError("Invalid shaped exchange constructor fields")
    width, particles = _integer(arguments, "num_bits_p"), _finite(arguments, "eta")
    signature = _momentum_exchange_declaration(particles, width)
    if controlled:
        raw = arguments["cvs"]
        if isinstance(raw, int):
            raw = (raw,)
        if not isinstance(raw, (tuple, list, np.ndarray)) or np.asarray(raw).ndim != 1:
            raise SerializationError("Projectile cvs requires an integer or vector")
        if any(not isinstance(value, (int, np.integer, np.bool_)) for value in raw):
            raise SerializationError("Projectile exchange requires integer cvs")
        n = _integer(arguments, "num_bits_n")
        signature = Signature(
            (
                Register("ctrl", quantum_bit_type(), shape=(len(raw),)),
                *tuple(signature)[:-1],
                Register("junk", opaque_register_type(n), shape=(3,)),
            )
        )
    if (
        type(width) is not int
        or width == 0
        or particles < 2
        or (controlled and (type(n) is not int or n < width))
    ):
        return _finish(
            FunctionRecord(
                name,
                (),
                registers=tuple(signature.adjoint()) if adjoint else tuple(signature),
            ),
            name,
            registers,
        )
    return _finish(
        momentum_exchange_program(signature, controlled=controlled, adjoint=adjoint),
        name,
        registers,
    )


def _root(
    select: bool,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
    *,
    adjoint: bool = False,
) -> FunctionRecord | ProgramRecord:
    if (
        not arguments.keys() >= _ROOT_FIELDS
        or arguments.keys() - _ROOT_FIELDS - _DEFAULTS.keys()
    ):
        raise SerializationError("Invalid projectile root constructor fields")
    recipe: dict[str, ResolvedArgument] = {**_DEFAULTS, **arguments}
    sizes = {field: _integer(recipe, field) for field in recipe}
    p, n = sizes["num_bits_p"], sizes["num_bits_n"]
    eta, atoms, m = (
        _finite(recipe, field) for field in ("eta", "num_atoms", "m_param")
    )

    def leaf(kind: str, inverse: bool = False) -> ProgramRecord:
        fields = _LEAF_FIELDS[kind]
        values: dict[str, ResolvedArgument] = {field: sizes[field] for field in fields}
        if kind == "PrepareTUVSuperpositions":
            values["num_bits_rot_aa"] = sizes["num_bits_rot_aa"]
        record = _leaf(
            kind, kind + ("_adjoint" if inverse else ""), None, values, adjoint=inverse
        )
        assert isinstance(record, FunctionRecord)
        return ProgramRecord((record,))

    if not select:
        kinetic = kinetic_preparation_factories()[
            _BASE + "projectile.prepare_t.PrepareTFirstQuantizationWithProj"
        ](
            "kinetic",
            None,
            {
                field: sizes[field]
                for field in ("num_bits_p", "num_bits_n", "eta", "num_bits_rot_aa")
            },
        )
        potential = potential_preparation_factories()[
            _BASE + "projectile.prepare_uv.PrepareUVFirstQuantizationWithProj"
        ](
            "potential",
            None,
            {
                field: sizes[field]
                for field in (
                    "num_bits_p",
                    "num_bits_n",
                    "eta",
                    "num_atoms",
                    "m_param",
                    "lambda_zeta",
                    "num_bits_nuc_pos",
                )
            },
        )
        kinetic_program = (
            kinetic if isinstance(kinetic, ProgramRecord) else ProgramRecord((kinetic,))
        )
        potential_program = (
            potential
            if isinstance(potential, ProgramRecord)
            else ProgramRecord((potential,))
        )
        record = first_quantization_preparation_program(
            _projectile_prepare_declaration(n, eta, atoms, m),
            (
                leaf("PrepareTUVSuperpositions"),
                leaf("UniformSuperpostionIJFirstQuantization"),
                kinetic_program,
                potential_program,
            ),
        )
    else:
        controlled = _exchange(
            True,
            "controlled_exchange",
            None,
            {"num_bits_p": p, "num_bits_n": n, "eta": eta, "cvs": (0,)},
        )
        electron = _exchange(
            False, "electron_exchange", None, {"num_bits_p": p, "eta": eta}
        )
        controlled = (
            controlled
            if isinstance(controlled, ProgramRecord)
            else ProgramRecord((controlled,))
        )
        electron = (
            electron
            if isinstance(electron, ProgramRecord)
            else ProgramRecord((electron,))
        )
        axis = (
            controlled_swap_program(n)
            if type(n) is int and n > 0
            else ProgramRecord(
                (
                    FunctionRecord(
                        "axis_exchange",
                        (),
                        registers=(
                            Register("ctrl", quantum_bit_type()),
                            Register("x", opaque_register_type(n)),
                            Register("y", opaque_register_type(n)),
                        ),
                    ),
                )
            )
        )
        record = projectile_selection_program(
            _projectile_select_declaration(p, n, eta, atoms, m),
            {
                "controlled_exchange": controlled,
                "electron_exchange": electron,
                "axis_exchange": axis,
                "kinetic": leaf("SelectTFirstQuantizationWithProj"),
                "potential": leaf("SelectUVFirstQuantizationWithProj"),
            },
            sizes["num_bits_nuc_pos"],
        )
    pairs: tuple[tuple[ProgramRecord, ProgramRecord], ...]
    if adjoint:
        if not select:
            kinetic_inverse = _kinetic(
                "PrepareTFirstQuantizationWithProj",
                "kinetic_adjoint",
                None,
                {
                    field: sizes[field]
                    for field in ("num_bits_p", "num_bits_n", "eta", "num_bits_rot_aa")
                },
                adjoint=True,
            )
            potential_inverse = _potential(
                True,
                "potential_adjoint",
                None,
                {
                    field: sizes[field]
                    for field in (
                        "num_bits_p",
                        "num_bits_n",
                        "eta",
                        "num_atoms",
                        "m_param",
                        "lambda_zeta",
                        "num_bits_nuc_pos",
                    )
                },
                adjoint=True,
            )
            assert isinstance(potential_inverse, ProgramRecord)
            pairs = (
                (
                    leaf("PrepareTUVSuperpositions"),
                    leaf("PrepareTUVSuperpositions", True),
                ),
                (
                    leaf("UniformSuperpostionIJFirstQuantization"),
                    leaf("UniformSuperpostionIJFirstQuantization", True),
                ),
                (
                    kinetic_program,
                    (
                        ProgramRecord((kinetic_inverse,))
                        if isinstance(kinetic_inverse, FunctionRecord)
                        else kinetic_inverse
                    ),
                ),
                (potential_program, potential_inverse),
            )
        else:
            assert isinstance(controlled, ProgramRecord) and isinstance(
                electron, ProgramRecord
            )
            controlled_inverse = _exchange(
                True,
                "controlled_exchange_adjoint",
                None,
                {"num_bits_p": p, "num_bits_n": n, "eta": eta, "cvs": (0,)},
                adjoint=True,
            )
            electron_inverse = _exchange(
                False,
                "electron_exchange_adjoint",
                None,
                {"num_bits_p": p, "eta": eta},
                adjoint=True,
            )
            pairs = (
                (
                    controlled,
                    (
                        controlled_inverse
                        if isinstance(controlled_inverse, ProgramRecord)
                        else ProgramRecord((controlled_inverse,))
                    ),
                ),
                (
                    electron,
                    (
                        electron_inverse
                        if isinstance(electron_inverse, ProgramRecord)
                        else ProgramRecord((electron_inverse,))
                    ),
                ),
                (axis, axis),
                (
                    leaf("SelectTFirstQuantizationWithProj"),
                    leaf("SelectTFirstQuantizationWithProj", True),
                ),
                (
                    leaf("SelectUVFirstQuantizationWithProj"),
                    leaf("SelectUVFirstQuantizationWithProj", True),
                ),
            )
        record = adjoint_component_calls(record, pairs)
    return _finish(record, name, registers)


def projectile_component_factories() -> dict[str, ComponentFactory]:
    """Bind the complete source root file and its explicit direct declarations."""
    owners = {
        _BASE
        + "projectile.select_and_prepare.PrepareTUVSuperpositions": partial(
            _leaf, "PrepareTUVSuperpositions"
        ),
        _BASE
        + "select_and_prepare.UniformSuperpostionIJFirstQuantization": partial(
            _leaf, "UniformSuperpostionIJFirstQuantization"
        ),
        _BASE
        + "projectile.select_t.SelectTFirstQuantizationWithProj": partial(
            _leaf, "SelectTFirstQuantizationWithProj"
        ),
        _BASE
        + "projectile.select_uv.SelectUVFirstQuantizationWithProj": partial(
            _leaf, "SelectUVFirstQuantizationWithProj"
        ),
        _BASE + "select_and_prepare.MultiplexedCSwap3D": partial(_exchange, False),
        _BASE
        + "projectile.select_and_prepare.ControlledMultiplexedCSwap3D": partial(
            _exchange, True
        ),
        _BASE
        + "projectile.select_and_prepare.PrepareFirstQuantizationWithProj": partial(
            _root, False
        ),
        _BASE
        + "projectile.select_and_prepare.SelectFirstQuantizationWithProj": partial(
            _root, True
        ),
    }

    inverses = {
        key: partial(factory.func, *factory.args, **factory.keywords, adjoint=True)
        for key, factory in owners.items()
    }
    return constructor_adjoint_factories(owners, inverses)
