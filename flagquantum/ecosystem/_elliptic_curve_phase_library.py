# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow locked Qualtran v0.7.0 ECC phase-estimation source files.

"""Translate original single and double curve phases without repairing leaves."""

from collections.abc import Mapping

from ..algorithms.primitives.elliptic_curve_phase_estimation import (
    elliptic_curve_phase_signature,
    phase_addition_points,
)
from ..algorithms.primitives.elliptic_curve_point import ECPoint
from ..compiler.formats.legacy_basis_vector_factories import (
    basis_vector_factories,
    x_basis_vector_factories,
)
from ..compiler.formats.legacy_component_arguments import AffinePointArgument
from ..compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    ResolvedArgument,
)
from ..compiler.formats.legacy_modular_addition_factories import _integer
from ..compiler.formats.legacy_word_factories import _rename_program
from ..compiler.registers.elliptic_curve.double_phase_component import (
    curve_double_phase_program,
)
from ..compiler.registers.elliptic_curve.phase_estimation_component import (
    curve_phase_estimation_program,
)
from ..core.ir.program import FunctionRecord, ProgramRecord
from ..core.ir.program.registers import Register
from ..errors import SerializationError
from ._elliptic_curve_window_library import elliptic_curve_window_factories
from ._fourier_measurement_library import fourier_measurement_factories

_PHASE = "qualtran.bloqs.cryptography.ecc.ec_phase_estimate_r.ECPhaseEstimateR"
_DOUBLE = "qualtran.bloqs.cryptography.ecc.find_ecc_private_key.FindECCPrivateKey"


def _program(record: FunctionRecord | ProgramRecord) -> ProgramRecord:
    return record if isinstance(record, ProgramRecord) else ProgramRecord((record,))


def _point(arguments: Mapping[str, ResolvedArgument], name: str) -> ECPoint:
    argument = arguments[name]
    if not isinstance(argument, AffinePointArgument):
        raise SerializationError("Curve phases require original affine point arguments")
    return ECPoint(argument.x, argument.y, argument.mod, argument.curve_a)


def _argument(point: ECPoint) -> AffinePointArgument:
    return AffinePointArgument(point.x, point.y, point.mod, point.curve_a)


def elliptic_curve_phase_factories() -> dict[str, ComponentFactory]:
    """Retain scalar opaque addition and the original unused-CBit terminal gap."""
    windows = elliptic_curve_window_factories()
    measure = fourier_measurement_factories()[
        "qualtran.bloqs.cryptography._factoring_shims.MeasureQFT"
    ]
    plus = x_basis_vector_factories()["qualtran.bloqs.basic_gates.x_basis.PlusState"]
    initialize = basis_vector_factories()["qualtran.bloqs.basic_gates.z_basis.IntState"]

    def phase(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        required = {"n", "point"}
        if not required <= arguments.keys() or arguments.keys() - required - {
            "add_window_size",
            "mul_window_size",
        }:
            raise SerializationError("Invalid original curve phase constructor fields")
        point = _point(arguments, "point")
        width = _integer(arguments, "n")
        window = _integer({"window": arguments.get("add_window_size", 1)}, "window")
        mul = _integer({"window": arguments.get("mul_window_size", 1)}, "window")
        signature = elliptic_curve_phase_signature(width)
        if registers is not None and registers != tuple(signature):
            raise SerializationError("Invalid original curve phase register signature")
        if type(width) is not int or type(window) is not int or width < 1:
            return FunctionRecord(name, (), registers=tuple(signature))
        points = phase_addition_points(width, window, point)
        children = []
        for index, current in enumerate(points):
            fields: dict[str, ResolvedArgument] = {"n": width, "R": _argument(current)}
            kind = "ECAddR"
            if window != 1:
                kind = "ECWindowAddR"
                fields.update(add_window_size=window, mul_window_size=mul)
            children.append(
                _program(
                    windows["qualtran.bloqs.cryptography.ecc.ec_add_r." + kind](
                        f"addition_{index}", None, fields
                    )
                )
            )
        graph = curve_phase_estimation_program(
            signature,
            plus=_program(plus("plus", None, {})),
            additions=tuple(children),
            measurement=_program(measure("measure_qft", None, {"n": width})),
        )
        return _rename_program(graph, name)

    def double(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> ProgramRecord:
        required = {"n", "base_point", "public_key"}
        if not required <= arguments.keys() or arguments.keys() - required - {
            "add_window_size",
            "mul_window_size",
        }:
            raise SerializationError(
                "Invalid original double curve phase constructor fields"
            )
        if registers is not None and registers != ():
            raise SerializationError("Original double curve phase has no public ports")
        base, key = (_point(arguments, field) for field in ("base_point", "public_key"))
        width = _integer(arguments, "n")
        options: dict[str, ResolvedArgument] = {
            "n": width,
            "add_window_size": arguments.get("add_window_size", 1),
            "mul_window_size": arguments.get("mul_window_size", 1),
        }
        # The source checks modulus/curve consistency in model properties only.
        # Its root still threads the two supplied points and raw initial words.
        graph = curve_double_phase_program(
            initialize_x=_program(
                initialize("initial_x", None, {"bitsize": width, "val": base.x})
            ),
            initialize_y=_program(
                initialize("initial_y", None, {"bitsize": width, "val": base.y})
            ),
            base_phase=_program(
                phase("base_phase", None, {**options, "point": _argument(base)})
            ),
            key_phase=_program(
                phase("key_phase", None, {**options, "point": _argument(key)})
            ),
        )
        return _rename_program(graph, name)

    return {_PHASE: phase, _DOUBLE: double}
