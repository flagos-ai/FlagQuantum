# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow Qualtran v0.7.0 cryptography/ecc/ec_add_r.py.

"""Translate fixed-point and windowed curve constructors into owned Programs."""

from collections.abc import Mapping
from functools import partial

import numpy as np

from ..algorithms.primitives.elliptic_curve_fixed_point import (
    fixed_point_addition_signature,
)
from ..algorithms.primitives.elliptic_curve_point import ECPoint
from ..algorithms.primitives.elliptic_curve_window_addition import windowed_point_table
from ..compiler.formats.legacy_component_arguments import AffinePointArgument
from ..compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    ResolvedArgument,
)
from ..compiler.formats.legacy_modular_addition_factories import _integer
from ..compiler.formats.legacy_qroam_factories import qroam_factories
from ..compiler.formats.legacy_word_factories import _rename_program
from ..compiler.registers.elliptic_curve.window_addition_component import (
    windowed_curve_addition_program,
)
from ..compiler.registers.lookup.qroam_resource_model import _optimal_erase_log
from ..core.ir.program import FunctionRecord, ProgramRecord
from ..core.ir.program.registers import Register
from ..errors import SerializationError
from ._elliptic_curve_addition_library import elliptic_curve_addition_factories


def elliptic_curve_window_factories() -> dict[str, ComponentFactory]:
    """Keep source bodyless leaves separate from explicit measured execution."""
    lookups = qroam_factories(reserve_workspace=False)
    addition = elliptic_curve_addition_factories()[
        "qualtran.bloqs.cryptography.ecc.ec_add.ECAdd"
    ]

    def constructor(
        windowed: bool,
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        required = {"n", "R", *(("add_window_size",) if windowed else ())}
        if not required <= arguments.keys() or arguments.keys() - required - (
            {"mul_window_size"} if windowed else set()
        ):
            raise SerializationError("Invalid fixed-point curve constructor fields")
        point = arguments["R"]
        if not isinstance(point, AffinePointArgument):
            raise SerializationError(
                "Curve constructors require an affine point argument"
            )
        width = _integer(arguments, "n")
        window = _integer(arguments, "add_window_size") if windowed else None
        signature = fixed_point_addition_signature(width, window)
        if registers is not None and registers != tuple(signature):
            raise SerializationError("Invalid fixed-point curve register signature")
        mul_window = _integer(
            {"mul_window_size": arguments.get("mul_window_size", 1)}, "mul_window_size"
        )
        if not windowed or any(
            type(value) is not int
            for value in (
                width,
                window,
                mul_window,
                point.x,
                point.y,
                point.mod,
                point.curve_a,
            )
        ):
            return FunctionRecord(name, (), registers=tuple(signature))
        assert isinstance(width, int) and isinstance(window, int)
        table = windowed_point_table(
            width, window, ECPoint(point.x, point.y, point.mod, point.curve_a)
        )
        forward = lookups["qualtran.bloqs.data_loading.qroam_clean.QROAMClean"](
            "point_table",
            None,
            {
                "data_or_shape": np.asarray(table.datasets),
                "selection_bitsizes": np.array([window]),
                "target_bitsizes": np.array([width] * 3),
                "log_block_sizes": np.array(table.log_block_sizes),
            },
        )
        assert isinstance(forward, ProgramRecord)
        # Original erase block optimization is independent of the forward block.
        # The wrapper keeps this field as metadata, without a physical leaf body.
        erased = lookups[
            "qualtran.bloqs.data_loading.qroam_clean.QROAMCleanAdjointWrapper"
        ](
            "point_table_adjoint",
            None,
            {
                "qroam_clean": forward.functions[0],
                "log_block_sizes": np.array([_optimal_erase_log(1 << window)]),
            },
        )
        assert isinstance(erased, ProgramRecord)
        root = addition(
            "curve_addition",
            None,
            {"n": width, "mod": point.mod, "window_size": mul_window},
        )
        assert isinstance(root, ProgramRecord)
        return _rename_program(
            windowed_curve_addition_program(
                signature, lookup=forward, addition=root, erasure=erased
            ),
            name,
        )

    prefix = "qualtran.bloqs.cryptography.ecc.ec_add_r."
    return {
        prefix + "ECAddR": partial(constructor, False),
        prefix + "ECWindowAddR": partial(constructor, True),
    }
