# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors follow locked Qualtran v0.7.0 state_preparation_alias_sampling.py.

"""Replace concrete alias payloads with the existing directional native owner."""

from ..compiler.formats.legacy_alias_preparation_factories import (
    alias_preparation_factories,
)
from ..compiler.formats.legacy_component_library import deserialize_component_library
from ..compiler.formats.legacy_component_resolution import resolve_component_program
from ..core.ir.program import ProgramRecord
from ..errors import SerializationError


def capture_alias_preparation_pair(
    source: object,
) -> tuple[ProgramRecord, ProgramRecord]:
    """Keep the original alias codec and exact inverse, without retaining SDK data."""
    from qualtran.bloqs.state_preparation.state_preparation_alias_sampling import (
        SparseStatePreparationAliasSampling,
        StatePreparationAliasSampling,
    )
    from qualtran.serialization.bloq import bloqs_to_proto

    if not isinstance(
        source, (StatePreparationAliasSampling, SparseStatePreparationAliasSampling)
    ):
        raise SerializationError("Expected an explicit concrete alias PREPARE")

    def resolve(child: object) -> ProgramRecord:
        return resolve_component_program(
            deserialize_component_library(
                bloqs_to_proto(child, max_depth=0).SerializeToString()
            ),
            0,
            alias_preparation_factories(),
        )

    return resolve(source), resolve(source.adjoint())
