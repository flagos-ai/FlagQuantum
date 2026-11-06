# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 AutoPartition.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/bookkeeping/auto_partition.py

"""Bind named child ports and pass unused portions through unchanged."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from .._component_trace.symbolic_math import SymbolicInt, smax
from .bitwise import _validate_wires
from .partition import (
    RegisterLayout,
    partition_register,
    partition_width,
    unpartition_register,
)

WirePorts = Mapping[str, tuple[int, ...]]
PartitionGroups = tuple[tuple[str, tuple[tuple[str | None, SymbolicInt], ...]], ...]


@dataclass(frozen=True)
class AutoPartition:
    """Ordered packed groups. A None segment is an unused bystander."""

    groups: PartitionGroups
    child_outputs: tuple[str, ...] | None = None
    left_only: bool = False

    def __post_init__(self) -> None:
        layouts = self._layouts()
        partition_width(
            tuple((outer, partition_width(inner)) for outer, inner in layouts)
        )
        partition_width(tuple(port for _, inner in layouts for port in inner))
        inputs = self._inputs()
        if self.left_only and any(
            name is None for _, group in self.groups for name, _ in group
        ):
            raise ValueError("Cannot use left_only with unused registers")
        outputs = self.output_names
        if len(set(outputs)) != len(outputs) or not set(outputs) <= inputs.keys():
            raise ValueError(
                "Child outputs must be distinct ports from the input layout"
            )
        if not self.left_only and set(outputs) != inputs.keys():
            raise ValueError("Repacking requires all child input ports to be returned")

    def _inputs(self) -> dict[str, SymbolicInt]:
        return {
            name: width
            for _, group in self.groups
            for name, width in group
            if name is not None
        }

    @property
    def output_names(self) -> tuple[str, ...]:
        return (
            tuple(self._inputs()) if self.child_outputs is None else self.child_outputs
        )

    def _layouts(self) -> tuple[tuple[str, tuple[tuple[str, SymbolicInt], ...]], ...]:
        return tuple(
            (
                outer,
                tuple(
                    (name if name is not None else f"_unused{i}_{j}", width)
                    for j, (name, width) in enumerate(group)
                ),
            )
            for i, (outer, group) in enumerate(self.groups)
        )

    def _concrete_layouts(self) -> tuple[tuple[str, RegisterLayout], ...]:
        layouts = self._layouts()
        if any(type(width) is not int for _, layout in layouts for _, width in layout):
            raise ValueError("AutoPartition binding requires concrete widths")
        return cast("tuple[tuple[str, RegisterLayout], ...]", layouts)

    def logical_width(self, child_width: SymbolicInt) -> SymbolicInt:
        """Include every unused bystander live across the child."""
        used: SymbolicInt = sum(self._inputs().values())
        unused: SymbolicInt = sum(
            width for _, group in self.groups for name, width in group if name is None
        )
        if not self._inputs():
            return cast(SymbolicInt, smax(child_width, unused))
        return cast(SymbolicInt, smax(used, child_width)) + unused

    def bind(
        self, packed: WirePorts
    ) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]:
        """Return child inputs and the original unused bindings."""
        layouts = self._concrete_layouts()
        if packed.keys() != {name for name, _ in layouts}:
            raise ValueError("AutoPartition requires exactly the declared packed ports")
        wires = tuple(wire for word in packed.values() for wire in word)
        if wires:
            _validate_wires(wires)
        bound: dict[str, tuple[int, ...]] = {}
        unused: dict[str, tuple[int, ...]] = {}
        for (outer, layout), (_, group) in zip(layouts, self.groups, strict=True):
            parts = partition_register(packed[outer], layout)
            for (name, _), (child, _) in zip(layout, group, strict=True):
                if child is None:
                    unused[name] = parts[name]
                else:
                    bound[name] = parts[name]
        return bound, unused

    def restore(
        self, outputs: WirePorts, unused: WirePorts
    ) -> dict[str, tuple[int, ...]]:
        """Return native child outputs or repack them with the unchanged bystanders."""
        layouts = self._concrete_layouts()
        names = self.output_names
        if outputs.keys() != set(names):
            raise ValueError(
                "AutoPartition child outputs must match the declared names"
            )
        expected_unused = {
            name
            for (_, layout), (_, group) in zip(layouts, self.groups, strict=True)
            for (name, _), (child, _) in zip(layout, group, strict=True)
            if child is None
        }
        if unused.keys() != expected_unused:
            raise ValueError("AutoPartition requires the original unused ports")
        widths = self._inputs()
        if any(len(outputs[name]) != widths[name] for name in names):
            raise ValueError(
                "AutoPartition child output widths must match their inputs"
            )
        wires = tuple(
            wire for word in (*outputs.values(), *unused.values()) for wire in word
        )
        if wires:
            _validate_wires(wires)
        if self.left_only:
            return {name: outputs[name] for name in names}
        bound = dict(outputs) | dict(unused)
        return {
            outer: unpartition_register(
                {name: bound[name] for name, _ in layout}, layout
            )
            for outer, layout in layouts
        }
