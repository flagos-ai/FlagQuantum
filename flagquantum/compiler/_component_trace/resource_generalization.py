# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 resource graph generalizers.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/resource_counting/generalizers.py

"""Canonicalize declared resource call identities before aggregating edges."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Literal

from .call_graph import Generalizer, _component_calls
from .resource_model import ComponentCost
from .symbolic_math import SymbolicInt, SymbolicReal


@dataclass(frozen=True)
class RotationCall:
    """A rotation angle or power exponent, with its synthesis precision."""

    opcode: Literal["rx", "ry", "rz", "xpow", "ypow", "zpow"]
    parameter: SymbolicReal | None
    precision: SymbolicReal = 1e-11


@dataclass(frozen=True)
class PhaseCall:
    opcode: Literal["s", "t"]
    adjoint: bool = False


@dataclass(frozen=True)
class FixedTCall:
    """An explicit Toffoli or exact bitwise controlled swap."""

    opcode: Literal["toffoli", "fredkin"]


@dataclass(frozen=True)
class CliffordCall:
    """An explicit gate family recognized by the source Clifford predicate."""

    opcode: Literal["swap_bit", "h", "x", "y", "z", "cx", "cy", "cz", "arbitrary"]
    width: SymbolicInt
    adjoint: bool = False


@dataclass(frozen=True)
class ControlledRotationCall:
    """A rotation with explicit controls for cost classification."""

    call: RotationCall | PhaseCall
    controls: tuple[SymbolicInt, ...]


@dataclass(frozen=True)
class ConjunctionCall:
    """Binary And when the control tuple has two entries, otherwise MultiAnd.

    None is a generalized binary polarity. A width in place of a tuple
    represents a MultiAnd with unspecified control values.
    """

    controls: tuple[SymbolicInt | None, ...] | SymbolicInt
    adjoint: bool = False


@dataclass(frozen=True)
class ControlledSwapCall:
    width: SymbolicInt
    approximate: bool = False
    adjoint: bool = False


@dataclass(frozen=True)
class BookkeepingCall:
    opcode: Literal[
        "split", "split2", "join", "join2", "partition", "cast", "allocate", "free"
    ]
    layout: Hashable


@dataclass(frozen=True)
class PartitionedCall:
    call: ResourceCall
    layout: Hashable


ResourceCall = (
    ComponentCost
    | RotationCall
    | PhaseCall
    | FixedTCall
    | CliffordCall
    | ControlledRotationCall
    | ConjunctionCall
    | ControlledSwapCall
    | BookkeepingCall
    | PartitionedCall
)

DeclaredResourceCalls = Mapping[ResourceCall, Mapping[ResourceCall, SymbolicInt]]


def _resource_callees(
    node: ResourceCall, declared_calls: DeclaredResourceCalls | None
) -> Iterable[tuple[ResourceCall, SymbolicInt]]:
    if declared_calls is not None and node in declared_calls:
        yield from declared_calls[node].items()
    elif isinstance(node, ComponentCost):
        yield from _component_calls(node)


def _inside_partition(
    rule: Generalizer[ResourceCall], call: ResourceCall
) -> ResourceCall | None:
    if isinstance(call, PartitionedCall):
        child = rule(call.call)
        return None if child is None else replace(call, call=child)
    return call


def ignore_register_partition(call: ResourceCall) -> ResourceCall | None:
    """Omit register reshaping and unwrap auto-partitioned calls."""
    if isinstance(call, BookkeepingCall) and call.opcode not in {"allocate", "free"}:
        return None
    if isinstance(call, PartitionedCall):
        return ignore_register_partition(call.call)
    return call


def ignore_allocation(call: ResourceCall) -> ResourceCall | None:
    """Omit allocation or release, including inside partition wrappers."""
    if isinstance(call, BookkeepingCall) and call.opcode in {"allocate", "free"}:
        return None
    return _inside_partition(ignore_allocation, call)


def generalize_rotation(call: ResourceCall) -> ResourceCall | None:
    """Erase angle or exponent identity, and group S/T with their adjoints."""
    if isinstance(call, RotationCall):
        return replace(call, parameter=None)
    if isinstance(call, PhaseCall):
        return replace(call, adjoint=False)
    return _inside_partition(generalize_rotation, call)


def generalize_control_values(call: ResourceCall) -> ResourceCall | None:
    """Group binary And polarities and forward MultiAnd calls of equal arity."""
    if isinstance(call, ConjunctionCall) and isinstance(call.controls, tuple):
        if len(call.controls) == 2:
            return replace(call, controls=(None, None))
        if not call.adjoint:
            return replace(call, controls=len(call.controls))
    return _inside_partition(generalize_control_values, call)


def generalize_approximate_swap(call: ResourceCall) -> ResourceCall | None:
    """Map approximate controlled swaps to the self-adjoint exact identity."""
    if isinstance(call, ControlledSwapCall) and call.approximate:
        return ControlledSwapCall(call.width)
    return _inside_partition(generalize_approximate_swap, call)
