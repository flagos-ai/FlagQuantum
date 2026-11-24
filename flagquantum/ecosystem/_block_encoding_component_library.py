# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Constructors adapted from locked Qualtran v0.7.0 lcu_block_encoding.py.

"""Compose block constructors with the same scoped native packed-oracle owners."""

from ..compiler.formats.legacy_block_encoding_factories import (
    ControlledSelect,
    block_encoding_factories,
)
from ..compiler.formats.legacy_component_resolution import ComponentFactory
from ..compiler.formats.legacy_packed_oracle_factories import OracleContext
from ._packed_oracle_component_library import _packed_oracle_library_context


def block_encoding_library_factories(
    *,
    prepare_context: OracleContext | None = None,
    select_context: OracleContext | None = None,
    controlled_select: ControlledSelect | None = None,
) -> dict[str, ComponentFactory]:
    """Retain original child metadata and independently owned source control calls."""
    children, preparations, selections, controls = _packed_oracle_library_context(
        prepare_context=prepare_context, select_context=select_context
    )
    return {
        **children,
        **block_encoding_factories(
            prepare_context=preparations,
            select_context=selections,
            controlled_select=controlled_select or controls,
            adjoint_factory=children["qualtran._infra.adjoint.Adjoint"],
        ),
    }
