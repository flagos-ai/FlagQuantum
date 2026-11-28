# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Gate payload assembly follows locked Qualtran v0.7.0 sparse SELECT.

"""End Majorana SDK payloads before binding native sparse chemistry graphs."""

from typing import Literal

from ...compiler.formats.legacy_component_resolution import ComponentFactory
from ...compiler.formats.legacy_fixed_gate_factories import fixed_gate_factories
from ...compiler.formats.legacy_sparse_chemistry_factories import (
    sparse_chemistry_factories,
)
from ._majorana_library import majorana_library_factories
from ._named_gate_definitions import _NamedGateDefinitions
from .conversion import _cirq


def sparse_cy_profile() -> Literal["cirq", "scalar"]:
    """End the SDK gate-view choice here and return only native role metadata."""
    cirq = _cirq()
    # Canonical CY has a q-array view; ControlledGate(Y) maps to scalar CYGate.
    scalar_cy = _NamedGateDefinitions().get(cirq.Y.controlled()) is not None
    return "scalar" if scalar_cy else "cirq"


def sparse_chemistry_library_factories() -> dict[str, ComponentFactory]:
    """Compose scoped sparse roots with the existing exact gate payload owner."""
    children = majorana_library_factories()
    return {
        **fixed_gate_factories(),
        **children,
        **sparse_chemistry_factories(
            adjoint_factory=children["qualtran._infra.adjoint.Adjoint"],
            cy_profile=sparse_cy_profile(),
        ),
    }
