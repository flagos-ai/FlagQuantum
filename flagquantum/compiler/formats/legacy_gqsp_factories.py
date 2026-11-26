# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow locked Qualtran v0.7.0 qsp/generalized_qsp.py.

"""Resolve GQSP coefficient metadata using injected phase and exact child owners."""

from collections.abc import Callable, Mapping

import numpy as np
import sympy

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_types import quantum_bit_type
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...errors import SerializationError
from ..registers.component_control import _stored_control_adder
from ..registers.phase.generalized_qsp_component import (
    generalized_qsp_component_program,
    gqsp_control_view,
)
from .legacy_component_arguments import ControlSpecification
from .legacy_component_resolution import ComponentFactory, ResolvedArgument
from .legacy_controlled_factories import controlled_via_and_factories
from .legacy_rotation_factories import _real_argument
from .legacy_su2_factories import _su2_factory
from .legacy_unitary_block_factories import UnitaryPairOwner
from .legacy_word_factories import _rename_program

PhaseFactors = Callable[
    [tuple[complex, ...], tuple[complex, ...]],
    tuple[tuple[float, ...], tuple[float, ...], float],
]


def _coefficients(value: object) -> tuple[complex, ...]:
    if not isinstance(value, (tuple, list, np.ndarray)):
        raise SerializationError(
            "GQSP requires concrete numeric polynomial coefficients"
        )
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "iufc" or len(array) < 2:
        raise SerializationError(
            "GQSP requires numeric coefficient vectors of degree at least one"
        )
    return tuple(complex(coefficient) for coefficient in array)


def generalized_qsp_factories(
    *,
    phase_factors: PhaseFactors,
    child_owner: UnitaryPairOwner,
    positive_child_owner: UnitaryPairOwner,
) -> dict[str, ComponentFactory]:
    """Require actual native child pairs; numerical phase work stays injected."""

    def factory(
        name: str,
        registers: tuple[Register, ...] | None,
        arguments: Mapping[str, ResolvedArgument],
    ) -> FunctionRecord | ProgramRecord:
        if not {"U", "P", "Q"} <= arguments.keys() or arguments.keys() - {
            "U",
            "P",
            "Q",
            "negative_power",
            "precision",
        }:
            raise SerializationError(
                "GQSP requires U, P, Q and optional power/precision"
            )
        child = arguments["U"]
        if not isinstance(child, FunctionRecord):
            raise SerializationError("GQSP requires a resolved native U")
        left, right = _coefficients(arguments["P"]), _coefficients(arguments["Q"])
        power = arguments.get("negative_power", 0)
        if isinstance(power, sympy.Expr):
            if child.registers is None:
                raise SerializationError("GQSP requires typed U ports")
            signature = tuple(
                Signature((Register("signal", quantum_bit_type()), *child.registers))
            )
            if registers is not None and registers != signature:
                raise SerializationError(
                    "Invalid register signature for GeneralizedQSP"
                )
            return FunctionRecord(name, (), registers=signature)
        if not isinstance(
            power, (int, float, str, np.integer, np.floating, np.ndarray)
        ):
            raise SerializationError(
                "GQSP inverse power requires finite integer conversion"
            )
        try:
            # The source SDK uses int(negative_power) for its execution loop.
            # A negative result takes only forward branches and no surplus loop.
            execution_power = max(0, int(power))
        except (TypeError, ValueError, OverflowError) as error:
            raise SerializationError(
                "GQSP inverse power requires finite integer conversion"
            ) from error
        theta, phi, lambd = phase_factors(left, right)
        if len(theta) != len(left) or len(phi) != len(theta):
            raise SerializationError(
                "GQSP phase owner returned inconsistent phase lengths"
            )
        precision = _real_argument(arguments, "precision", 1e-11)
        rotations = tuple(
            _su2_factory(
                f"signal_rotation_{i}",
                None,
                {
                    "theta": t,
                    "phi": p,
                    "lambd": lambd if i == 0 else 0,
                    "eps": precision / (len(left) + 1),
                },
            )
            for i, (t, p) in enumerate(zip(theta, phi, strict=True))
        )
        ordinary = child_owner(child)
        if (
            not ordinary[0].functions
            or ordinary[0].functions[0].registers != child.registers
        ):
            raise SerializationError(
                "GQSP child owner must preserve the resolved U ports"
            )
        assert child.registers is not None
        positive = tuple(
            gqsp_control_view(record, child.registers)
            for record in positive_child_owner(child)
        )
        positive_forward = _rename_program(positive[0], child.name + "_positive")
        control = controlled_via_and_factories(
            control_system=lambda _: (
                positive_forward,
                _stored_control_adder(positive_forward.functions[0]),
            )
        )
        negative = control["qualtran.bloqs.mcmt.controlled_via_and.ControlledViaAnd"](
            "negative_forward",
            None,
            {
                "subbloq": child,
                "ctrl_spec": ControlSpecification(((quantum_bit_type(), np.array(0)),)),
            },
        )
        assert isinstance(negative, ProgramRecord)
        result = generalized_qsp_component_program(
            *ordinary, positive[1], negative, rotations, negative_power=execution_power
        )
        if registers is not None and registers != result.functions[0].registers:
            raise SerializationError("Invalid register signature for GeneralizedQSP")
        return _rename_program(result, name)

    return {"qualtran.bloqs.qsp.generalized_qsp.GeneralizedQSP": factory}
