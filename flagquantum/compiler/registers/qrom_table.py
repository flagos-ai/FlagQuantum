# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 QROMBase data and shape factories.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/data_loading/qrom_base.py

"""Lookup metadata shared by concrete binding and symbolic resource queries."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from math import prod
from typing import cast

import numpy as np
from numpy.typing import ArrayLike

from ...core.ir import Instruction
from ...core.ir.program.registers import Register, RegisterType
from ...core.ir.program.signatures import Signature
from ...core.value_parameters import is_symbolic_value
from .._component_trace.resource_model import ComponentCost
from .._component_trace.symbolic_math import SymbolicInt, bit_length
from .partition import RegisterLayout, partition_register
from .qrom_resource_model import source_qrom_cost
from .source_qrom import source_qrom_ir, source_qrom_workspace_size
from .symbolic_resource_model import (
    SymbolicResourceFormula,
    source_qrom_resource_formula,
)


@dataclass(frozen=True)
class QromTable:
    """Row-major lookup metadata, independent of wire and workspace ownership.

    Shape-only tables can be bound to data with ``with_data``. Target array
    axes follow selection axes; words and their physical bits are big endian.
    Binding never allocates wires or classical feedback slots.
    """

    shape: tuple[SymbolicInt, ...]
    selection_bitsizes: tuple[SymbolicInt, ...]
    target_bitsizes: tuple[SymbolicInt, ...]
    target_shapes: tuple[tuple[SymbolicInt, ...], ...]
    num_controls: SymbolicInt = 0
    datasets: tuple[tuple[int, ...], ...] | None = None

    def __post_init__(self) -> None:
        if (
            not self.target_bitsizes
            or len(self.target_bitsizes) != len(self.target_shapes)
            or len(self.shape) != len(self.selection_bitsizes)
        ):
            raise ValueError(
                "QROM requires one width per selection axis and target shape"
            )
        for value, minimum in (
            *((n, 1) for n in self.shape),
            *((n, 1) for sh in self.target_shapes for n in sh),
            *((n, 0) for n in self.selection_bitsizes + self.target_bitsizes),
            (self.num_controls, 0),
        ):
            if type(value) is int:
                if value < minimum:
                    raise ValueError(
                        "QROM requires nonnegative widths and positive axes"
                    )
            else:
                from sympy import Expr

                if not isinstance(value, Expr):
                    raise ValueError(
                        "QROM dimensions must be integers or symbolic expressions"
                    )
        if self.datasets is not None and (
            len(self.datasets) != len(self.target_shapes)
            or any(
                len(data) != prod(self.shape + target)
                for data, target in zip(self.datasets, self.target_shapes, strict=True)
            )
        ):
            raise ValueError("QROM datasets must match the declared array shapes")

    def _require_concrete_dimensions(self) -> None:
        dimensions = (
            self.num_controls,
            *self.shape,
            *self.selection_bitsizes,
            *self.target_bitsizes,
            *(n for sh in self.target_shapes for n in sh),
        )
        if any(type(n) is not int for n in dimensions):
            raise ValueError("Physical QROM binding requires Python integer dimensions")

    @classmethod
    def from_shape(
        cls,
        shape: SymbolicInt | tuple[SymbolicInt, ...],
        target_bitsizes: SymbolicInt | tuple[SymbolicInt, ...],
        *,
        target_shapes: tuple[tuple[SymbolicInt, ...], ...] = (),
        selection_bitsizes: tuple[SymbolicInt, ...] = (),
        num_controls: SymbolicInt = 0,
    ) -> "QromTable":
        """Build source shape metadata without materializing any lookup data."""
        axes = shape if isinstance(shape, tuple) else (shape,)
        widths = (
            target_bitsizes
            if isinstance(target_bitsizes, tuple)
            else (target_bitsizes,)
        )
        return cls(
            axes,
            selection_bitsizes or tuple(bit_length(n - 1) for n in axes),
            widths,
            target_shapes or ((),) * len(widths),
            num_controls,
        )

    @classmethod
    def from_data(
        cls,
        *data: ArrayLike,
        target_bitsizes: tuple[int, ...] | None = None,
        target_shapes: tuple[tuple[int, ...], ...] = (),
        num_controls: int = 0,
    ) -> "QromTable":
        """Infer source widths before converting lookup values to native integers.

        Like the source factory, explicit widths must be a tuple. A width
        smaller than its data is retained here and rejected by IR binding.
        """
        if not data:
            raise ValueError("QROM requires at least one dataset")
        arrays = tuple(np.array(d, dtype=int) for d in data)
        widths = target_bitsizes
        if widths is None:
            widths = tuple(
                max(int(np.max(np.asarray(d))).bit_length(), 1) for d in data
            )
        if not isinstance(widths, tuple):
            raise ValueError("QROM data factory requires a tuple of target widths")
        targets = target_shapes or ((),) * len(widths)
        if len(targets) != len(arrays) or len(widths) != len(arrays):
            raise ValueError("QROM requires one target shape and width per dataset")
        shapes = tuple(
            a.shape[: -len(t)] if t else a.shape
            for a, t in zip(arrays, targets, strict=True)
        )
        if any(sh != shapes[0] for sh in shapes) or any(
            t and a.shape[-len(t) :] != t for a, t in zip(arrays, targets, strict=True)
        ):
            raise ValueError(
                "QROM datasets require common selection and declared target axes"
            )
        table = cls.from_shape(
            shapes[0], widths, target_shapes=targets, num_controls=num_controls
        )
        return replace(table, datasets=tuple(tuple(map(int, a.flat)) for a in arrays))

    def with_data(self, *data: ArrayLike) -> "QromTable":
        """Replace values while preserving every declared width and array shape."""
        arrays = tuple(np.array(d, dtype=int) for d in data)
        if len(arrays) != len(self.target_shapes) or any(
            a.shape != self.shape + t
            for a, t in zip(arrays, self.target_shapes, strict=True)
        ):
            raise ValueError("Replacement QROM data must preserve every declared shape")
        if any(
            width < int(np.max(a)).bit_length()
            for width, a in zip(self.target_bitsizes, arrays, strict=True)
        ):
            raise ValueError("Replacement QROM data exceeds its declared target width")
        return replace(self, datasets=tuple(tuple(map(int, a.flat)) for a in arrays))

    def _data(self) -> tuple[tuple[int, ...], ...]:
        if self.datasets is None:
            raise ValueError("QROM data is unavailable; bind it with with_data first")
        return self.datasets

    @property
    def target_layout(self) -> tuple[tuple[SymbolicInt, SymbolicInt], ...]:
        """Return existing compiler (word count, word width) layout pairs."""
        return tuple(
            (prod(shape), bits)
            for shape, bits in zip(
                self.target_shapes, self.target_bitsizes, strict=True
            )
        )

    def _selection_layout(self) -> tuple[tuple[str, SymbolicInt], ...]:
        widths = []
        for extent, width in zip(self.shape, self.selection_bitsizes, strict=True):
            if (
                isinstance(extent, int)
                and isinstance(width, int)
                and extent > 1 << width
            ):
                raise ValueError("QROM selection widths must fit the source axes")
            if not isinstance(extent, int) or extent > 1:
                widths.append(width)
        if len(widths) == 1:
            return (("selection", widths[0]),)
        return tuple((f"selection{i}", width) for i, width in enumerate(widths))

    @property
    def register_layout(self) -> tuple[tuple[str, SymbolicInt], ...]:
        """Flatten source control/selection/target names into FQ partition widths.

        Target array shapes remain in ``target_shapes``. Singleton selection
        axes and Python-integer zero-width targets are omitted as in the source.
        Symbolic integer literals retain the original symbolic dispatch.
        """
        return tuple(
            (register.name, register.total_bits()) for register in self.signature
        )

    @cached_property
    def signature(self) -> Signature:
        """Retain source encoding, side and shape before physical flattening."""
        registers = (
            [Register("control", RegisterType("opaque", self.num_controls))]
            if self.num_controls
            else []
        )
        selected_axes = (
            extent for extent in self.shape if is_symbolic_value(extent) or extent > 1
        )
        for (name, width), extent in zip(
            self._selection_layout(), selected_axes, strict=True
        ):
            registers.append(
                Register(
                    name, RegisterType("uint", width, parameters=(("bound", extent),))
                )
            )
        registers.extend(
            Register(f"target{index}_", RegisterType("opaque", width), shape=shape)
            for index, (width, shape) in enumerate(
                zip(self.target_bitsizes, self.target_shapes, strict=True)
            )
            if is_symbolic_value(width) or width
        )
        return Signature(registers)

    def bind_packed(
        self,
        ports: tuple[int, ...],
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
    ) -> tuple[Instruction, ...]:
        """Partition one caller-owned port tuple and bind the existing QROM body."""
        self._data()
        self._require_concrete_dimensions()
        parts = partition_register(ports, cast(RegisterLayout, self.register_layout))
        selected = iter(parts[name] for name, _ in self._selection_layout())
        selections = tuple(
            next(selected) if extent > 1 else ()
            for extent in cast(tuple[int, ...], self.shape)
        )
        targets = tuple(
            tuple(
                parts.get(f"target{i}_", ())[j * width : (j + 1) * width]
                for j in range(words)
            )
            for i, (words, width) in enumerate(
                cast(tuple[tuple[int, int], ...], self.target_layout)
            )
        )
        return self.bind(
            parts.get("control", ()), selections, targets, workspace, classical_bits
        )

    @property
    def workspace_size(self) -> int:
        data = self._data()
        self._require_concrete_dimensions()
        return source_qrom_workspace_size(
            cast(int, self.num_controls),
            cast(tuple[int, ...], self.selection_bitsizes),
            data,
        )

    def resource_formula(
        self, *, legacy_shims: bool = False
    ) -> SymbolicResourceFormula:
        """Query unknown-data declarations, including symbolic dimensions.

        This query deliberately ignores any concrete values. Use
        ``resource_model`` for data-dependent pruning and physical-width checks.
        """
        return source_qrom_resource_formula(
            self.shape,
            self.target_bitsizes,
            target_shapes=self.target_shapes,
            control_count=self.num_controls,
            selection_widths=self.selection_bitsizes,
            legacy_shims=legacy_shims,
        )

    def resource_model(self, *, legacy_shims: bool = False) -> ComponentCost:
        """Query the existing concrete QROM model without emitting gates."""
        data = self._data()
        self._require_concrete_dimensions()
        return source_qrom_cost(
            cast(int, self.num_controls),
            cast(tuple[int, ...], self.selection_bitsizes),
            cast(tuple[int, ...], self.shape),
            cast(tuple[tuple[int, int], ...], self.target_layout),
            data,
            legacy_shims=legacy_shims,
        )

    def bind(
        self,
        controls: tuple[int, ...],
        selections: tuple[tuple[int, ...], ...],
        targets: tuple[tuple[tuple[int, ...], ...], ...],
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
    ) -> tuple[Instruction, ...]:
        """Bind caller-owned wires and slots to the existing measured XOR tree."""
        data = self._data()
        self._require_concrete_dimensions()
        layout = cast(tuple[tuple[int, int], ...], self.target_layout)
        if (
            len(controls) != self.num_controls
            or tuple(map(len, selections)) != self.selection_bitsizes
            or tuple((len(group), tuple(map(len, group))) for group in targets)
            != tuple((n, (w,) * n) for n, w in layout)
        ):
            raise ValueError("QROM physical ports must match the table metadata")
        return source_qrom_ir(
            controls,
            selections,
            cast(tuple[int, ...], self.shape),
            targets,
            data,
            workspace,
            classical_bits,
        )
