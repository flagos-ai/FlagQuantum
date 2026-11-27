# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 data_loading/select_swap_qrom.py.

"""Preserve blocked source metadata independently of strict physical layouts."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, fields, replace
from functools import cached_property
from operator import index
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ....core.ir.program.signatures import Signature
from ....core.value_math import prod, ssum
from ....core.value_parameters import is_symbolic_value
from ..._component_trace.qec_gate_cost import QECGateCost, query_qec_gate_cost
from ..._component_trace.resource_generalization import CliffordCall, ControlledSwapCall
from ..._component_trace.symbolic_math import SymbolicInt, ceil
from ..resources.symbolic_resource_model import source_qrom_resource_formula
from .qrom_declaration import QromDeclaration, qrom_selection_registers
from .qrom_declaration_resources import declaration_qrom_gate_cost
from .select_swap_callees import SelectSwapRouteCall, SelectSwapXorCall
from .symbolic_qroam_resource_model import _forward_logs

SelectSwapDeclaredCall = QromDeclaration | SelectSwapRouteCall | SelectSwapXorCall


@dataclass(frozen=True)
class SelectSwapDeclaration:
    """Own source block axes, including trailing axes ignored by zipped shapes.

    The base declaration owns immutable data, signature and host-value semantics.
    This record owns block derivation, padded/batched literals and direct calls.
    Concrete properties implement the existing shared physical bank stages;
    source symbolic and malformed declarations remain separate from execution.
    """

    lookup: QromDeclaration
    logs: tuple[SymbolicInt, ...] | None = None
    dirty: bool = True

    def __post_init__(self) -> None:
        if self.lookup.target_shapes != ((),) * len(self.lookup.target_bitsizes):
            raise ValueError("Select-swap requires scalar dataset targets")
        if self.logs is None:
            object.__setattr__(
                self,
                "logs",
                _forward_logs(
                    self.lookup.data_shape, self.lookup.target_bitsizes, self.dirty
                ),
            )

    @property
    def signature(self) -> Signature:
        return self.lookup.signature

    def is_symbolic(self) -> bool:
        return self.lookup.is_symbolic() or is_symbolic_value(*(self.logs or ()))

    def with_data(self, *data: ArrayLike) -> SelectSwapDeclaration:
        """Retain source replacement checks and the already selected block axes."""
        return replace(self, lookup=self.lookup.with_data(*data))

    def with_log_block_sizes(
        self, logs: tuple[SymbolicInt, ...] | SymbolicInt | None = None
    ) -> "SelectSwapDeclaration":
        if logs is None:
            return self
        values = logs if isinstance(logs, tuple) else (logs,)
        if not is_symbolic_value(*values):
            assert all(
                1 <= 2**log <= extent
                for log, extent in zip(values, self.lookup.data_shape, strict=False)
            )
        return replace(self, logs=values)

    @property
    def source_block_sizes(self) -> tuple[SymbolicInt, ...]:
        return tuple(2**log for log in self.logs or ())

    @property
    def source_batch_shape(self) -> tuple[SymbolicInt, ...]:
        return tuple(
            ceil(n / k)
            for n, k in zip(
                self.lookup.data_shape, self.source_block_sizes, strict=False
            )
        )

    @property
    def source_high_widths(self) -> tuple[SymbolicInt, ...]:
        return tuple(
            width - log
            for width, log in zip(
                self.lookup.selection_bitsizes, self.logs or (), strict=False
            )
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(index(n) for n in self.lookup.data_shape)

    @property
    def log_block_sizes(self) -> tuple[int, ...]:
        return tuple(index(n) for n in self.logs or ())

    @property
    def block_shape(self) -> tuple[int, ...]:
        return tuple(index(n) for n in self.source_block_sizes)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(index(n) for n in self.source_batch_shape)

    def _high_widths(self, widths: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(
            width - log
            for width, log in zip(widths, self.log_block_sizes, strict=False)
        )

    def require_physical_layout(self) -> None:
        """Reject known source graph failures before allocating any bank words.

        Source declarations and model queries do not call this method. In
        particular, reject oversized exponents before constructing 2**log.
        """
        if self.is_symbolic() or index(self.lookup.num_controls) < 0:
            raise ValueError(
                "Source select-swap needs concrete nonnegative physical ports"
            )
        if any(
            isinstance(width, bool) or index(width) < 1
            for width in self.lookup.target_bitsizes
        ):
            raise ValueError("Source select-swap cannot bind omitted target words")
        shape, logs = self.shape, self.log_block_sizes
        if any(log != 0 for log in logs[len(shape) :]) or any(
            extent < 1 or log < 0 or log > extent.bit_length() - 1
            for extent, log in zip(shape, logs, strict=False)
        ):
            raise ValueError(
                "Source select-swap block exponents cannot bind these axes"
            )
        parents = tuple(
            reg for reg in self.signature if reg.name.startswith("selection")
        )
        children = qrom_selection_registers(
            self.source_batch_shape, self.source_high_widths
        )
        routed = any(logs)
        # The frozen builder zips already filtered selector ports with the raw
        # logs. Its return list must still supply every original selector.
        high = (
            tuple(
                index(reg.bitsize) - log
                for reg, log in zip(parents, logs, strict=False)
            )
            if routed
            else tuple(index(reg.bitsize) for reg in parents)
        )
        if len(parents) != len(children) or high != tuple(
            index(reg.bitsize) for reg in children
        ):
            raise ValueError("Source select-swap loses an omitted high selector")
        if routed:
            low = logs[: len(parents)]
            selected = tuple(log for log in logs if log > 0)
            if len(low) != len(parents) or low != selected:
                raise ValueError("Source select-swap loses an omitted low selector")

    @property
    def padded_arrays(self) -> tuple[NDArray[np.generic], ...]:
        """Return source zero padding without converting the original dtype."""
        pad = tuple(
            (0, batch * block - extent)
            for extent, block, batch in zip(
                self.shape, self.block_shape, self.batch_shape, strict=False
            )
        )
        return tuple(np.pad(array, pad) for array in self.lookup.data)

    @property
    def batched_arrays(self) -> tuple[NDArray[np.int_], ...]:
        """Port source assignment into integer batch storage, including truncation."""
        outputs = []
        for padded in self.padded_arrays:
            batched = np.zeros(self.batch_shape + self.block_shape, dtype=int)
            block_slices = tuple(slice(0, k) for k in self.block_shape)
            for batch in np.ndindex(self.batch_shape):
                source = tuple(
                    slice(i * k, (i + 1) * k)
                    for i, k in zip(batch, self.block_shape, strict=False)
                )
                batched[batch + block_slices] = padded[source]
            outputs.append(batched)
        return tuple(outputs)

    @cached_property
    def qrom(self) -> QromDeclaration:
        child = QromDeclaration.from_shape(
            self.source_batch_shape,
            self.lookup.target_bitsizes,
            target_shapes=(self.source_block_sizes,) * len(self.lookup.target_bitsizes),
            selection_bitsizes=self.source_high_widths,
            num_controls=self.lookup.num_controls,
        )
        return child if self.is_symbolic() else child.with_data(*self.batched_arrays)

    def calls(self) -> dict[SelectSwapDeclaredCall, int]:
        copies = 2 if self.dirty else 1
        calls: Counter[SelectSwapDeclaredCall] = Counter({self.qrom: 2})
        for width in self.lookup.target_bitsizes:
            if is_symbolic_value(width) or width:
                calls[SelectSwapXorCall(width)] += copies
        if any(is_symbolic_value(log) or log > 0 for log in self.logs or ()):
            for width in self.lookup.target_bitsizes:
                calls[
                    SelectSwapRouteCall(
                        width, self.logs or (), self.source_block_sizes, False
                    )
                ] += copies
                calls[
                    SelectSwapRouteCall(
                        width, self.logs or (), self.source_block_sizes, True
                    )
                ] += copies
        return dict(calls)

    def static_qubit_count(self) -> SymbolicInt:
        return (
            self.qrom.static_qubit_count()
            + ssum(self.logs or ())
            + ssum(self.lookup.target_bitsizes)
        )

    def gate_cost(self, *, legacy_shims: bool = False) -> QECGateCost:
        """Aggregate existing QROM, XOR and approximate-swap model authorities."""
        if type(legacy_shims) is not bool:
            raise ValueError("Select-swap cost profile must be boolean")
        total = QECGateCost()
        for child, count in self.calls().items():
            if isinstance(child, QromDeclaration):
                if child.has_data():
                    cost = declaration_qrom_gate_cost(child, legacy_shims=legacy_shims)
                else:
                    formula = source_qrom_resource_formula(
                        child.data_shape,
                        child.target_bitsizes,
                        target_shapes=child.target_shapes,
                        control_count=child.num_controls,
                        selection_widths=child.selection_bitsizes,
                        legacy_shims=legacy_shims,
                    )
                    cost = QECGateCost(
                        **{
                            field.name: cast(SymbolicInt, getattr(formula, field.name))
                            for field in fields(QECGateCost)
                        }
                    )
            elif isinstance(child, SelectSwapRouteCall):
                cost = query_qec_gate_cost(
                    ControlledSwapCall(
                        child.width, approximate=True, adjoint=child.adjoint
                    ),
                    legacy_shims=legacy_shims,
                ) * (prod(child.banks) - 1)
            else:
                # Source Xor(QAny) declares one CNOT per stored bit. Its
                # logical count does not imply physical word-width admission.
                cost = (
                    query_qec_gate_cost(
                        CliffordCall("cx", 2), legacy_shims=legacy_shims
                    )
                    * child.width
                )
            total = total.plus(cost, count)
        return total
