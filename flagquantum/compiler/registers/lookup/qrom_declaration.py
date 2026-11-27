# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 data_loading/qrom_base.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/data_loading/qrom_base.py

"""Preserve source lookup declarations before concrete layout validation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from ....core.ir.program.register_types import (
    bounded_unsigned_register_type,
    opaque_register_type,
)
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ....core.value_math import ssum
from ....core.value_parameters import is_symbolic_value
from ..._array_literals import freeze_array, thaw_array
from ..._component_trace.symbolic_math import SymbolicInt, bit_length

if TYPE_CHECKING:
    from .qrom_table import QromTable


def qrom_selection_registers(
    shape: tuple[SymbolicInt, ...],
    selection_bitsizes: tuple[SymbolicInt, ...],
) -> tuple[Register, ...]:
    """Build source selector ports without touching target or control metadata."""
    selectors = tuple(
        bounded_unsigned_register_type(
            int(width) if isinstance(width, bool) else width, extent
        )
        for extent, width in zip(shape, selection_bitsizes, strict=False)
        if is_symbolic_value(extent) or extent > 1
    )
    return (
        (Register("selection", selectors[0]),)
        if len(selectors) == 1
        else tuple(
            Register(f"selection{i}", dtype) for i, dtype in enumerate(selectors)
        )
    )


def qrom_signature(
    shape: tuple[SymbolicInt, ...],
    selection_bitsizes: tuple[SymbolicInt, ...],
    target_bitsizes: tuple[SymbolicInt, ...],
    target_shapes: tuple[tuple[SymbolicInt, ...], ...],
    num_controls: SymbolicInt,
) -> Signature:
    """Retain the source's zipped axes, omitted words and nominal port types."""
    indices = qrom_selection_registers(shape, selection_bitsizes)
    controls = (
        (
            Register(
                "control",
                opaque_register_type(
                    int(num_controls)
                    if isinstance(num_controls, bool)
                    else num_controls
                ),
            ),
        )
        if num_controls
        else ()
    )
    targets = tuple(
        Register(f"target{i}_", opaque_register_type(width), shape=target)
        for i, (width, target) in enumerate(
            zip(target_bitsizes, target_shapes, strict=False)
        )
        if is_symbolic_value(width) or width
    )
    return Signature((*controls, *indices, *targets))


@dataclass(frozen=True)
class QromDeclaration:
    """Own original array shapes and deferred metadata, without physical storage.

    Array literals preserve raw dtype and shape until a visited source load.
    Integer tuples support arbitrary-width native words without object pickle.
    None denotes unknown data. The stricter materialize operation validates a
    dense integer table; source compilation does not require that projection.
    """

    dataset_shapes: tuple[tuple[SymbolicInt, ...], ...]
    literals: tuple[bytes | tuple[int, ...] | None, ...]
    selection_bitsizes: tuple[SymbolicInt, ...]
    target_bitsizes: tuple[SymbolicInt, ...]
    target_shapes: tuple[tuple[SymbolicInt, ...], ...]
    num_controls: SymbolicInt = 0

    def __post_init__(self) -> None:
        assert all(
            is_symbolic_value(n) or isinstance(n, int) for n in self.selection_bitsizes
        )
        assert all(
            is_symbolic_value(n) or isinstance(n, int) for n in self.target_bitsizes
        )
        assert (
            len(self.target_bitsizes) == len(self.dataset_shapes) == len(self.literals)
        )
        assert len(self.target_bitsizes) == len(self.target_shapes)

    @cached_property
    def data_shape(self) -> tuple[SymbolicInt, ...]:
        """Read the original common prefix, including its empty-prefix sentinel."""
        result: tuple[SymbolicInt, ...] = ()
        for shape, target in zip(self.dataset_shapes, self.target_shapes, strict=False):
            prefix = shape[: -len(target)] if target else shape
            if result == ():
                result = prefix
            if result != prefix:
                raise ValueError("All datasets must have same shape")
        return result

    def has_data(self) -> bool:
        return all(row is not None for row in self.literals)

    def is_symbolic(self) -> bool:
        return not self.has_data() or is_symbolic_value(
            self.num_controls,
            *self.selection_bitsizes,
            *self.target_bitsizes,
            *(extent for target in self.target_shapes for extent in target),
        )

    @property
    def signature(self) -> Signature:
        return qrom_signature(
            self.data_shape,
            self.selection_bitsizes,
            self.target_bitsizes,
            self.target_shapes,
            self.num_controls,
        )

    def selection_axes(self) -> tuple[SymbolicInt, ...]:
        """Project source selector extents for the separate host-value evaluator."""
        # The source callback reads selector ports only. Malformed target or
        # control storage cannot reject an otherwise defined host-value query.
        qrom_selection_registers(self.data_shape, self.selection_bitsizes)
        return tuple(
            extent
            for extent, _ in zip(self.data_shape, self.selection_bitsizes, strict=False)
            if is_symbolic_value(extent) or extent > 1
        )

    def static_qubit_count(self) -> SymbolicInt:
        """Quote the source width expression without requiring loadable data.

        The original can undercount scalar or malformed layouts. This is its
        declared model, not a verified physical lease peak.
        """
        return (
            self.signature.n_qubits()
            + ssum(self.selection_bitsizes)
            - 1
            + self.num_controls
        )

    @classmethod
    def from_data(
        cls,
        *data: ArrayLike,
        target_bitsizes: tuple[SymbolicInt, ...] | None = None,
        target_shapes: tuple[tuple[SymbolicInt, ...], ...] = (),
        num_controls: SymbolicInt = 0,
    ) -> QromDeclaration:
        """Convert array literals without eagerly reading their common prefix."""
        arrays = tuple(np.array(row, dtype=int) for row in data)
        widths = target_bitsizes
        if widths is None:
            widths = tuple(
                max(int(np.max(np.asarray(row))).bit_length(), 1) for row in data
            )
        assert isinstance(widths, tuple)
        targets = target_shapes or ((),) * len(widths)
        selection_len = len(arrays[0].shape) - len(targets[0])
        selection = tuple((n - 1).bit_length() for n in arrays[0].shape[:selection_len])
        return cls(
            tuple(row.shape for row in arrays),
            tuple(tuple(map(int, row.flat)) for row in arrays),
            selection,
            widths,
            targets,
            num_controls,
        )

    @classmethod
    def from_arrays(
        cls,
        *data: ArrayLike,
        selection_bitsizes: tuple[SymbolicInt, ...],
        target_bitsizes: tuple[SymbolicInt, ...],
        target_shapes: tuple[tuple[SymbolicInt, ...], ...] = (),
        num_controls: SymbolicInt = 0,
    ) -> QromDeclaration:
        """Lower raw source arrays without its data-factory int64 coercion.

        The pickle-free literal keeps ignored cells and raw equality without
        prematurely converting any loaded or unused cell to an integer.
        """
        arrays = tuple(np.asarray(row) for row in data)
        return cls(
            tuple(row.shape for row in arrays),
            tuple(freeze_array(row) for row in arrays),
            selection_bitsizes,
            target_bitsizes,
            target_shapes or ((),) * len(target_bitsizes),
            num_controls,
        )

    @classmethod
    def from_shape(
        cls,
        shape: SymbolicInt | tuple[SymbolicInt, ...],
        target_bitsizes: SymbolicInt | tuple[SymbolicInt, ...],
        *,
        target_shapes: tuple[tuple[SymbolicInt, ...], ...] = (),
        selection_bitsizes: tuple[SymbolicInt, ...] = (),
        num_controls: SymbolicInt = 0,
    ) -> QromDeclaration:
        """Retain source unknown arrays without importing a foreign Shaped object."""
        axes = shape if isinstance(shape, tuple) else (shape,)
        widths = (
            target_bitsizes
            if isinstance(target_bitsizes, tuple)
            else (target_bitsizes,)
        )
        targets = target_shapes or ((),) * len(widths)
        selection = selection_bitsizes or tuple(bit_length(n - 1) for n in axes)
        assert len(selection) == len(axes)
        return cls(
            tuple(axes + target for target in targets),
            (None,) * len(targets),
            selection,
            widths,
            targets,
            num_controls,
        )

    def with_data(self, *data: ArrayLike) -> QromDeclaration:
        """Replace literal values with the original array-shape and width checks."""
        arrays = tuple(np.array(row, dtype=int) for row in data)
        assert all(
            row.shape == shape
            for row, shape in zip(arrays, self.dataset_shapes, strict=False)
        )
        assert len(arrays) == len(self.target_bitsizes)
        assert all(
            width >= int(np.max(row)).bit_length()
            for width, row in zip(self.target_bitsizes, arrays, strict=False)
        )
        return replace(
            self,
            dataset_shapes=tuple(row.shape for row in arrays),
            literals=tuple(tuple(map(int, row.flat)) for row in arrays),
        )

    @property
    def data(self) -> tuple[np.ndarray[Any, np.dtype[Any]], ...]:
        """Return fresh compile-time arrays; unknown data is never materialized."""
        if not self.has_data():
            raise ValueError("Data not available for symbolic QROM")
        arrays = []
        for literal, shape in zip(self.literals, self.dataset_shapes, strict=True):
            if isinstance(literal, bytes):
                arrays.append(thaw_array(literal))
            else:
                assert literal is not None
                # Object dtype preserves arbitrary-width native integer words.
                arrays.append(np.asarray(literal, dtype=object).reshape(shape))
        return tuple(arrays)

    @property
    def datasets(self) -> tuple[tuple[int, ...] | None, ...]:
        """Project every cell to an integer only for strict dense-table callers."""
        if not self.has_data():
            return (None,) * len(self.literals)
        return tuple(tuple(map(int, row.flat)) for row in self.data)

    def materialize(self) -> QromTable:
        """Require the existing validated layout before asking for physical work."""
        from .qrom_table import QromTable

        values = None
        if self.has_data():
            values = tuple(row for row in self.datasets if row is not None)
        return QromTable(
            self.data_shape,
            tuple(
                int(width) if isinstance(width, bool) else width
                for width in self.selection_bitsizes
            ),
            self.target_bitsizes,
            self.target_shapes,
            (
                int(self.num_controls)
                if isinstance(self.num_controls, bool)
                else self.num_controls
            ),
            values,
        )
