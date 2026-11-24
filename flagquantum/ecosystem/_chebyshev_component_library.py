# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Follows locked Qualtran v0.7.0 block_encoding/chebyshev_polynomial.py.

"""Convert explicit polynomial block dependencies into native exact pairs."""

from math import pi
from typing import Any, cast

from ..compiler.formats.legacy_component_library import deserialize_component_library
from ..compiler.formats.legacy_component_resolution import resolve_component_program
from ..compiler.formats.legacy_linear_combination_factories import ExactPair
from ..compiler.formats.legacy_unitary_block_factories import unitary_block_factories
from ..compiler.registers.preparation.chebyshev_component import (
    chebyshev_component_pair,
    scaled_chebyshev_component_pair,
)
from ..compiler.registers.preparation.identity_preparation_component import (
    identity_preparation_program,
)
from ..compiler.registers.preparation.prepared_reflection_component import (
    prepared_reflection_program,
)
from ..compiler.registers.primitives.global_phase_component import scalar_phase_program
from ..core.ir.program.register_types import opaque_register_type
from ..core.ir.program.registers import Register
from ..errors import SerializationError
from ._default_linear_combination_oracles import (
    _positive_pair,
    default_linear_combination_prepare_pair,
    default_linear_combination_select_pair,
)
from ._linear_combination_component_library import capture_linear_combination_context


def capture_chebyshev_pair(source: object, *, controlled: bool = False) -> ExactPair:
    """Capture actual source coefficients and children without inferring metadata.

    Scaled roots use the original linear_combination property, including its
    normalization and coefficient rounding. All external objects stop here;
    returned Programs contain only native typed calls and exact inverses.
    Positive control follows the original child specialization independently
    of the generic inverse of the resulting root.
    """
    from qualtran import CtrlSpec
    from qualtran._infra.adjoint import Adjoint
    from qualtran.bloqs.block_encoding.chebyshev_polynomial import (
        ChebyshevPolynomial,
        ScaledChebyshevPolynomial,
    )
    from qualtran.bloqs.block_encoding.linear_combination import LinearCombination
    from qualtran.bloqs.block_encoding.unitary import Unitary
    from qualtran.serialization.bloq import bloqs_to_proto

    foreign = cast(Any, source)
    if isinstance(source, ScaledChebyshevPolynomial):
        pair = capture_chebyshev_pair(foreign.linear_combination, controlled=controlled)
        return scaled_chebyshev_component_pair(pair, controlled=controlled)
    if isinstance(source, ChebyshevPolynomial):
        child = capture_chebyshev_pair(foreign.block_encoding, controlled=controlled)
        width = foreign.ancilla_bitsize
        if type(width) is not int:
            raise SerializationError(
                "Executable Chebyshev ancilla width must be concrete"
            )
        reflection = None
        if foreign.order and width:
            preparation = identity_preparation_program(
                (Register("reg0_", opaque_register_type(width)),)
            )
            reflection = prepared_reflection_program(
                preparation,
                preparation,
                ("reg0_",),
                control_value=1 if controlled else None,
                global_phase=-1,
                scalar_phase=None if controlled else scalar_phase_program(pi),
            )
        return chebyshev_component_pair(
            *child, foreign.order, reflection=reflection, controlled=controlled
        )
    if isinstance(source, LinearCombination):
        if controlled:
            foreign, _ = foreign.get_ctrl_system(CtrlSpec())
        context = capture_linear_combination_context(
            foreign,
            prepare_owner=default_linear_combination_prepare_pair,
            select_owner=lambda child: default_linear_combination_select_pair(
                child,
                positive_child_owner=lambda block: capture_chebyshev_pair(
                    block, controlled=True
                ),
            ),
        )
        return context.program(), context.program(adjoint=True)
    if isinstance(source, Unitary):
        if controlled:
            return _positive_pair(source)
        factories = unitary_block_factories()
        records = tuple(
            resolve_component_program(
                deserialize_component_library(
                    bloqs_to_proto(block, max_depth=0).SerializeToString()
                ),
                0,
                factories,
            )
            for block in (source, Adjoint(source))
        )
        return records[0], records[1]
    raise SerializationError("Missing explicit native Chebyshev dependency owner")
