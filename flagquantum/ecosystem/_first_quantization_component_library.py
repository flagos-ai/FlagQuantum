# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow locked Qualtran v0.7.0 ordinary select_and_prepare.py.
# The standalone prepare.py particle-pair declaration has the same owned ports.

"""Bind ordinary first-quantization roots through shared typed dependency owners."""

from collections.abc import Mapping
from functools import partial

from ..algorithms.first_quantization import first_quantization_signature
from ..algorithms.first_quantization_kinetic import select_t_signature
from ..algorithms.first_quantization_potential import (
    apply_nuclear_phase_signature,
    select_uv_signature,
)
from ..compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    ResolvedArgument,
)
from ..compiler.formats.legacy_rotation_factories import _real_argument
from ..compiler.registers.control.first_quantization_selection_component import (
    first_quantization_selection_program,
)
from ..compiler.registers.preparation.first_quantization_preparation_component import (
    first_quantization_preparation_program,
)
from ..core.ir.program import FunctionRecord, ProgramRecord
from ..core.ir.program.registers import Register
from ..core.ir.program.signatures import Signature
from ..errors import SerializationError
from ._kinetic_preparation_library import kinetic_preparation_factories
from ._nu_preparation_library import _finite
from ._potential_preparation_library import potential_preparation_factories
from ._projectile_component_library import (
    _DEFAULTS,
    _finish,
    projectile_component_factories,
)

_BASE = "qualtran.bloqs.chemistry.pbc.first_quantization."
_FIELDS = {
    "PrepareTUVSuperpositions": {"num_bits_t", "eta", "lambda_zeta"},
    "SelectTFirstQuantization": {"num_bits_p", "eta"},
    "SelectUVFirstQuantization": {"num_bits_p", "eta", "num_atoms", "num_bits_nuc_pos"},
    "ApplyNuclearPhase": {"num_bits_p", "num_bits_nuc"},
}


def _leaf(
    kind: str,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    required = _FIELDS[kind]
    optional = {"num_bits_rot_aa"} if kind == "PrepareTUVSuperpositions" else set()
    if not required <= arguments.keys() or arguments.keys() - required - optional:
        raise SerializationError("Invalid ordinary first-quantization leaf fields")
    sizes = {field: _finite(arguments, field) for field in required}
    if kind == "PrepareTUVSuperpositions":
        if "num_bits_rot_aa" in arguments:
            _finite(arguments, "num_bits_rot_aa")
        signature = Signature.build(tuv=1, uv=1)
    elif kind == "SelectTFirstQuantization":
        signature = select_t_signature(sizes["num_bits_p"])
    elif kind == "SelectUVFirstQuantization":
        signature = select_uv_signature(
            sizes["num_bits_p"], sizes["num_atoms"], sizes["num_bits_nuc_pos"]
        )
    else:
        signature = apply_nuclear_phase_signature(
            sizes["num_bits_p"], sizes["num_bits_nuc"]
        )
    return _finish(
        FunctionRecord(name, (), registers=tuple(signature)), name, registers
    )


def _root(
    select: bool,
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord | ProgramRecord:
    required = {"num_bits_p", "eta", "num_atoms", "lambda_zeta"}
    optional = _DEFAULTS.keys() | ({"sum_of_l1_coeffs"} if not select else set())
    if not required <= arguments.keys() or arguments.keys() - required - optional:
        raise SerializationError("Invalid ordinary first-quantization root fields")
    if arguments.get("sum_of_l1_coeffs") is not None:
        # The existing library codec retains the original value; this scalar
        # never creates a phase, a gate or an inferred normalization.
        _real_argument(arguments, "sum_of_l1_coeffs")
    recipe: dict[str, ResolvedArgument] = {**_DEFAULTS, **arguments}
    sizes = {
        field: _finite(recipe, field) for field in recipe if field != "sum_of_l1_coeffs"
    }
    p, eta, atoms, m = (
        sizes[field] for field in ("num_bits_p", "eta", "num_atoms", "m_param")
    )
    signature = first_quantization_signature(p, eta, atoms, m, prepare=not select)
    shared = projectile_component_factories()

    def leaf(kind: str) -> ProgramRecord:
        values: dict[str, ResolvedArgument] = {
            field: sizes[field] for field in _FIELDS[kind]
        }
        if kind == "PrepareTUVSuperpositions":
            values["num_bits_rot_aa"] = sizes["num_bits_rot_aa"]
        record = _leaf(kind, kind, None, values)
        assert isinstance(record, FunctionRecord)
        return ProgramRecord((record,))

    if not select:
        particle = shared[
            _BASE + "select_and_prepare.UniformSuperpostionIJFirstQuantization"
        ]("particles", None, {"eta": eta, "num_bits_rot_aa": sizes["num_bits_rot_aa"]})
        kinetic = kinetic_preparation_factories()[
            _BASE + "prepare_t.PrepareTFirstQuantization"
        ](
            "kinetic",
            None,
            {field: sizes[field] for field in ("num_bits_p", "eta", "num_bits_rot_aa")},
        )
        potential = potential_preparation_factories()[
            _BASE + "prepare_uv.PrepareUVFirstQuantization"
        ](
            "potential",
            None,
            {
                field: sizes[field]
                for field in (
                    "num_bits_p",
                    "eta",
                    "num_atoms",
                    "m_param",
                    "lambda_zeta",
                    "num_bits_nuc_pos",
                )
            },
        )
        particle_program = (
            particle
            if isinstance(particle, ProgramRecord)
            else ProgramRecord((particle,))
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
            signature,
            (
                leaf("PrepareTUVSuperpositions"),
                particle_program,
                kinetic_program,
                potential_program,
            ),
        )
    else:
        exchange = shared[_BASE + "select_and_prepare.MultiplexedCSwap3D"](
            "exchange", None, {"num_bits_p": p, "eta": eta}
        )
        assert isinstance(exchange, ProgramRecord)
        record = first_quantization_selection_program(
            signature,
            {
                "exchange": exchange,
                "kinetic": leaf("SelectTFirstQuantization"),
                "potential": leaf("SelectUVFirstQuantization"),
            },
            sizes["num_bits_nuc_pos"],
        )
    return _finish(record, name, registers)


def first_quantization_component_factories() -> dict[str, ComponentFactory]:
    """Bind the complete ordinary root file and its explicit direct declarations."""
    shared = projectile_component_factories()
    return {
        _BASE
        + "prepare.UniformSuperpostionIJFirstQuantization": shared[
            _BASE + "select_and_prepare.UniformSuperpostionIJFirstQuantization"
        ],
        _BASE
        + "select_and_prepare.UniformSuperpostionIJFirstQuantization": shared[
            _BASE + "select_and_prepare.UniformSuperpostionIJFirstQuantization"
        ],
        _BASE
        + "select_and_prepare.MultiplexedCSwap3D": shared[
            _BASE + "select_and_prepare.MultiplexedCSwap3D"
        ],
        _BASE
        + "select_and_prepare.PrepareTUVSuperpositions": partial(
            _leaf, "PrepareTUVSuperpositions"
        ),
        _BASE + "select_and_prepare.PrepareFirstQuantization": partial(_root, False),
        _BASE + "select_and_prepare.SelectFirstQuantization": partial(_root, True),
        _BASE
        + "select_t.SelectTFirstQuantization": partial(
            _leaf, "SelectTFirstQuantization"
        ),
        _BASE
        + "select_uv.SelectUVFirstQuantization": partial(
            _leaf, "SelectUVFirstQuantization"
        ),
        _BASE + "select_uv.ApplyNuclearPhase": partial(_leaf, "ApplyNuclearPhase"),
    }
