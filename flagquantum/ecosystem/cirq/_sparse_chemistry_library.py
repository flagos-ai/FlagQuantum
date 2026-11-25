# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Gate payload assembly follows locked Qualtran v0.7.0 sparse SELECT.

"""End Majorana SDK payloads before binding native sparse chemistry graphs."""

from ...compiler.formats.legacy_component_resolution import ComponentFactory
from ...compiler.formats.legacy_fixed_gate_factories import fixed_gate_factories
from ...compiler.formats.legacy_sparse_chemistry_factories import (
    sparse_chemistry_factories,
)
from ._majorana_library import majorana_library_factories


def sparse_chemistry_library_factories() -> dict[str, ComponentFactory]:
    """Compose scoped sparse roots with the existing exact gate payload owner."""
    children = majorana_library_factories()
    return {
        **fixed_gate_factories(),
        **children,
        **sparse_chemistry_factories(
            adjoint_factory=children["qualtran._infra.adjoint.Adjoint"]
        ),
    }
