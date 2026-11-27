# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Callee fields adapted from locked SelectSwapQROM direct calls.

"""Immutable route and copy identities shared by blocked lookup protocols."""

from dataclasses import dataclass

from ..._component_trace.symbolic_math import SymbolicInt


@dataclass(frozen=True)
class SelectSwapRouteCall:
    """The original route parameters and the direction of the exchange."""

    width: SymbolicInt
    selectors: tuple[SymbolicInt, ...]
    banks: tuple[SymbolicInt, ...]
    adjoint: bool


@dataclass(frozen=True)
class SelectSwapXorCall:
    """XOR the routed word into a matching destination word."""

    width: SymbolicInt
