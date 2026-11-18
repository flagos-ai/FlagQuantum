# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 phase_estimation/kaiser_window_state.py at
# bcb17bb25360ed68ce3bc084645db3e929799389.

"""Declare exact Kaiser windows and bind their explicit quantized rotation child."""

from collections.abc import Hashable
from dataclasses import dataclass, replace
from functools import cached_property
from typing import TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from ...compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from ...compiler._component_trace.resource_model import ComponentCost, CostCall
from ...compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
)
from ...core.value_math import SymbolicInt, SymbolicReal, ceil, ln, log2, pi, smax
from ...core.value_parameters import is_symbolic_value
from ...simulation.numerics.kaiser_window import kaiser_window_coefficients
from ...simulation.value_dimensions import SymbolicSequenceLength
from ._block_binding import _BoundBlockOperation
from .phase_estimation_window import PhaseEstimationWindow
from .rotation_preparation import RotationPreparation
from .rotation_resource_model import rotation_preparation_cost
from .symbolic_rotation_resource_model import (
    source_rotation_preparation_resource_formula,
)

_Node = TypeVar("_Node", bound=Hashable)


@dataclass(frozen=True)
class KaiserWindowState(PhaseEstimationWindow):
    """Preserve the exact source tensor and its separately declared rotation child.

    The source has no executable decomposition. Explicit make_on binds its
    quantized child with the original bitsize as phase precision. The caller
    supplies a reusable negative phase-gradient eigenstate, clean target and
    scratch. That child's output approximates the exact coefficient tensor.
    """

    bitsize: SymbolicInt
    alpha: SymbolicReal

    @property
    def m_bits(self) -> SymbolicInt:
        return self.bitsize

    @cached_property
    def kaiser_state_coeff(self) -> NDArray[np.float64]:
        """Return the original normalized SciPy periodic coefficients."""
        if not isinstance(self.bitsize, int):
            raise ValueError("Cannot compute coefficients for symbolic Kaiser width")
        return kaiser_window_coefficients(self.bitsize, self.alpha)

    @classmethod
    def from_precision_and_delta(
        cls, precision: SymbolicInt, delta: SymbolicReal
    ) -> "KaiserWindowState":
        """Retain the complete source ln/ln-ln expression and its failure behavior."""
        alpha = 1 / pi(delta) * ln(1 / delta) + ln(ln(1 / delta))
        width = precision + ceil(
            log2(ln(1 / delta + 1 / 2**precision * ln(ln(1 / delta))))
        )
        return cls(width, alpha)

    @property
    def source_preparation_coefficients(
        self,
    ) -> tuple[complex, ...] | SymbolicSequenceLength:
        """Snapshot concrete coefficients or retain the original HasLength shape."""
        if is_symbolic_value(self.bitsize, self.alpha):
            return SymbolicSequenceLength(2**self.bitsize)
        return tuple(complex(value) for value in self.kaiser_state_coeff)

    def _validate_phase_precision(self) -> None:
        if isinstance(self.bitsize, int) and self.bitsize < 2:
            raise ValueError("Source Kaiser child requires at least two phase bits")

    def source_graph(
        self, *, root: _Node, preparation: _Node
    ) -> SymbolicCallGraph[_Node]:
        """Declare exactly one original preparation child under caller-owned keys."""
        self._validate_phase_precision()
        return symbolic_call_graph(root, {root: {preparation: 1}})

    def rotation_preparation(self) -> RotationPreparation:
        """Bind concrete source coefficients at the original phase precision."""
        self._validate_phase_precision()
        coefficients = self.source_preparation_coefficients
        if isinstance(coefficients, SymbolicSequenceLength) or not isinstance(
            self.bitsize, int
        ):
            raise ValueError(
                "Kaiser rotation binding requires concrete coefficients and width"
            )
        return RotationPreparation(coefficients, self.bitsize)

    def source_shape_resource_formula(self) -> SymbolicResourceFormula:
        """Query the original shape-only child when coefficients are unresolved."""
        self._validate_phase_precision()
        if not isinstance(self.source_preparation_coefficients, SymbolicSequenceLength):
            raise ValueError("Use the concrete model for known Kaiser coefficients")
        child = source_rotation_preparation_resource_formula(
            2**self.bitsize, self.bitsize
        )
        return replace(
            child,
            logical_qubits=cast(SymbolicInt, smax(self.bitsize, child.logical_qubits)),
        )

    def source_resource_model(self) -> ComponentCost:
        """Query the accepted concrete rotation model independently of emission."""
        preparation = self.rotation_preparation()
        child = rotation_preparation_cost(preparation)
        return ComponentCost(
            "KaiserWindowState", preparation.phase_width, calls=(CostCall(child),)
        )

    def make_on(
        self,
        *,
        qpe_reg: tuple[int, ...],
        phase_gradient: tuple[int, ...],
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
        uncompute: bool = False,
    ) -> _BoundBlockOperation:
        """Append the declared child with explicit gradient and owned work.

        The gradient begins in its negative-exponent eigenstate and is preserved.
        Work begins and ends clean under the child's measured lookup contract.
        """
        gates = self.rotation_preparation().emit(
            (), qpe_reg, phase_gradient, workspace, classical_bits, uncompute=uncompute
        )
        return _BoundBlockOperation(
            gates, max((*qpe_reg, *phase_gradient, *workspace)) + 1
        )
