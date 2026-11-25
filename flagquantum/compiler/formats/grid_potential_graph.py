# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Recipe follows locked Qualtran v0.7.0 chemistry/trotter/grid_ham/potential.py.

"""Own symbolic pair input graphs and bind dimensions before Program lowering."""

from collections.abc import Mapping
from copy import deepcopy

import sympy

from ...errors import SerializationError
from .legacy_component_arguments import ComponentArgumentValue
from .legacy_component_library import ComponentLibrary
from .legacy_component_resolution import ResolvedArgument
from .legacy_grid_potential_factories import grid_potential_factories

_PAIR = "qualtran.bloqs.chemistry.trotter.grid_ham.potential.PairPotential"
_DIMENSIONS = {"bitsize", "poly_bitsize", "inv_sqrt_bitsize"}


def grid_pair_potential_library(
    arguments: Mapping[str, ResolvedArgument], *, name: str = "pair"
) -> ComponentLibrary:
    """Construct the full source call layer with owned static coefficient data.

    This is a compiler input graph, not an executable symbolic Program. It uses
    the same recipe as concrete construction. Original object-array protocol
    serialization limitations are retained until dimensions are bound.
    """
    graphs: dict[str, ComponentLibrary] = {}
    grid_potential_factories(_pair_graphs=graphs)[_PAIR](
        name, None, deepcopy(arguments)
    )
    return graphs[name]


def bind_grid_pair_potential_library(
    library: ComponentLibrary, bindings: Mapping[str, int]
) -> ComponentLibrary:
    """Bind constructor dimensions and rebuild the same owned input recipe.

    Registered constructor arguments retain their existing precedence over a
    stored expansion. Coefficient banks are copied unchanged from that root.
    Partial bindings keep a full symbolic graph; concrete bindings can use the
    existing library resolver. Unused names and non-integer bindings fail.
    """
    if not library.entries or library.entries[0].declaration.name != _PAIR:
        raise SerializationError("Expected a pair-potential constructor library")
    arguments: dict[str, ComponentArgumentValue] = {
        argument.name: argument.value
        for argument in library.entries[0].declaration.arguments
    }
    dimensions = {
        key: value
        for key, value in arguments.items()
        if key in _DIMENSIONS and isinstance(value, sympy.Expr)
    }
    symbols = {
        str(symbol) for value in dimensions.values() for symbol in value.free_symbols
    }
    if bindings.keys() - symbols or any(
        type(value) is not int for value in bindings.values()
    ):
        raise SerializationError(
            "Pair graph bindings require known symbols and integer dimensions"
        )
    for key, value in dimensions.items():
        replacements = {
            symbol: bindings[str(symbol)]
            for symbol in value.free_symbols
            if str(symbol) in bindings
        }
        result = value.subs(replacements)
        arguments[key] = int(result) if replacements and result.is_Integer else result
    return grid_pair_potential_library(arguments, name=library.name or "pair")
