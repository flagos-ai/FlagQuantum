# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 QROMBase data and concrete shape factories.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/data_loading/qrom_base.py

"""Concrete lookup metadata shared by QROM binding and resource queries."""

from dataclasses import dataclass, replace
from math import prod

import numpy as np
from numpy.typing import ArrayLike

from ...core.ir import Instruction
from .._component_trace.resource_model import ComponentCost
from .qrom_resource_model import source_qrom_cost
from .source_qrom import source_qrom_ir, source_qrom_workspace_size


@dataclass(frozen=True)
class QromTable:
    """Row-major lookup metadata, independent of wire and workspace ownership.

    Shape-only tables can be bound to data with ``with_data``. Target array
    axes follow selection axes; words and their physical bits are big endian.
    Binding never allocates wires or classical feedback slots.
    """

    shape: tuple[int, ...]
    selection_bitsizes: tuple[int, ...]
    target_bitsizes: tuple[int, ...]
    target_shapes: tuple[tuple[int, ...], ...]
    num_controls: int = 0
    datasets: tuple[tuple[int, ...], ...] | None = None

    def __post_init__(self) -> None:
        if (
            not self.target_bitsizes
            or len(self.target_bitsizes) != len(self.target_shapes)
            or len(self.shape) != len(self.selection_bitsizes)
            or type(self.num_controls) is not int
            or self.num_controls < 0
            or any(type(n) is not int or n < 1 for n in self.shape)
            or any(type(n) is not int or n < 0 for n in self.selection_bitsizes)
            or any(type(n) is not int or n < 0 for n in self.target_bitsizes)
            or any(type(n) is not int or n < 1 for sh in self.target_shapes for n in sh)
        ):
            raise ValueError(
                "QROM requires concrete nonnegative widths and positive axes"
            )
        if self.datasets is not None and (
            len(self.datasets) != len(self.target_shapes)
            or any(
                len(data) != prod(self.shape + target)
                for data, target in zip(self.datasets, self.target_shapes, strict=True)
            )
        ):
            raise ValueError("QROM datasets must match the declared array shapes")

    @classmethod
    def from_shape(
        cls,
        shape: int | tuple[int, ...],
        target_bitsizes: int | tuple[int, ...],
        *,
        target_shapes: tuple[tuple[int, ...], ...] = (),
        selection_bitsizes: tuple[int, ...] = (),
        num_controls: int = 0,
    ) -> "QromTable":
        """Build concrete metadata without materializing any lookup data."""
        axes = (shape,) if isinstance(shape, int) else shape
        widths = (
            (target_bitsizes,) if isinstance(target_bitsizes, int) else target_bitsizes
        )
        return cls(
            axes,
            selection_bitsizes or tuple((n - 1).bit_length() for n in axes),
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
    def target_layout(self) -> tuple[tuple[int, int], ...]:
        """Return existing compiler (word count, word width) layout pairs."""
        return tuple(
            (prod(shape), bits)
            for shape, bits in zip(
                self.target_shapes, self.target_bitsizes, strict=True
            )
        )

    @property
    def workspace_size(self) -> int:
        return source_qrom_workspace_size(
            self.num_controls, self.selection_bitsizes, self._data()
        )

    def resource_model(self, *, legacy_shims: bool = False) -> ComponentCost:
        """Query the existing concrete QROM model without emitting gates."""
        return source_qrom_cost(
            self.num_controls,
            self.selection_bitsizes,
            self.shape,
            self.target_layout,
            self._data(),
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
        if (
            len(controls) != self.num_controls
            or tuple(map(len, selections)) != self.selection_bitsizes
            or tuple((len(group), tuple(map(len, group))) for group in targets)
            != tuple((n, (w,) * n) for n, w in self.target_layout)
        ):
            raise ValueError("QROM physical ports must match the table metadata")
        return source_qrom_ir(
            controls,
            selections,
            self.shape,
            targets,
            self._data(),
            workspace,
            classical_bits,
        )
