# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 guided Hamiltonian and qubitization QPE declarations.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/hamiltonian_simulation

"""Shape and call declarations for guided Hamiltonian phase estimation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

from ...compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from ...core.ir.program.register_types import (
    fixed_point_register_type,
    opaque_register_type,
)
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt, SymbolicReal, ceil, log2
from ...core.value_parameters import is_symbolic_value


@dataclass(frozen=True)
class GuidedPhaseEstimationModel:
    """Declare the guide, Kaiser window and qubitized walk for a block encoding."""

    system_width: SymbolicInt
    walk_ancilla_width: SymbolicInt
    guide_selection_width: SymbolicInt
    guide_ancilla_width: SymbolicInt
    precision: SymbolicReal
    failure_probability: SymbolicReal
    resource_width: SymbolicInt = 0

    def __post_init__(self) -> None:
        if self.resource_width != 0:
            raise ValueError("Block encoding resource registers are unsupported")
        if self.system_width != self.guide_selection_width:
            raise ValueError("Block encoding and guide system widths must match")

    @cached_property
    def _failure_log(self) -> SymbolicReal:
        inverse = 1 / self.failure_probability
        if is_symbolic_value(inverse):
            from sympy.codegen.cfunctions import log

            return log(inverse)
        return math.log(inverse)

    @cached_property
    def window_alpha(self) -> SymbolicReal:
        if is_symbolic_value(self.failure_probability):
            from sympy import pi

            return self._failure_log / pi
        return self._failure_log / math.pi

    @cached_property
    def phase_width(self) -> SymbolicInt:
        return ceil(log2((1 / self.precision) * self._failure_log))

    @cached_property
    def signature(self) -> Signature:
        return Signature.build_from_dtypes(
            phase_estimate=fixed_point_register_type(
                self.phase_width, self.phase_width
            ),
            system=opaque_register_type(self.system_width),
            walk_ancilla=opaque_register_type(self.walk_ancilla_width),
            guide_ancilla=opaque_register_type(self.guide_ancilla_width),
        )

    def source_call_graph(
        self, *, max_depth: int | None = None
    ) -> SymbolicCallGraph[str]:
        """Keep the guided root and qubitization QPE as distinct declarations."""
        root = "GuidedHamiltonianPhaseEstimation"
        return symbolic_call_graph(root, self._calls(), max_depth=max_depth)

    def _calls(self) -> dict[str, dict[str, SymbolicInt]]:
        return {
            "GuidedHamiltonianPhaseEstimation": {
                "BlackBoxPrepare": 1,
                "QubitizationQPE": 1,
            },
            "QubitizationQPE": {
                "KaiserWindowState": 1,
                "ControlledWalk": 1,
                "ZeroControlledReflection": 2 * (self.phase_width - 1),
                "Walk": 2**self.phase_width - 2,
                "QFTWithReverse†": 1,
            },
        }


@dataclass(frozen=True)
class GuidedHamiltonianModel:
    """Declare the guided phase estimate and amplitude amplification schedule."""

    system_width: SymbolicInt
    walk_ancilla_width: SymbolicInt
    guide_selection_width: SymbolicInt
    guide_ancilla_width: SymbolicInt
    block_scale: SymbolicReal
    threshold: SymbolicReal
    relative_gap: SymbolicReal
    overlap: SymbolicReal
    resource_width: SymbolicInt = 0

    def __post_init__(self) -> None:
        # The original constructor evaluates the phase signature eagerly.
        _ = self.signature

    @cached_property
    def phase_estimation(self) -> GuidedPhaseEstimationModel:
        return GuidedPhaseEstimationModel(
            self.system_width,
            self.walk_ancilla_width,
            self.guide_selection_width,
            self.guide_ancilla_width,
            self.relative_gap * self.threshold / self.block_scale,
            self.overlap**3,
            self.resource_width,
        )

    @property
    def signature(self) -> Signature:
        return self.phase_estimation.signature

    @cached_property
    def amplification_rounds(self) -> SymbolicInt:
        return ceil(1 / self.overlap)

    def source_call_graph(
        self, *, max_depth: int | None = None
    ) -> SymbolicCallGraph[str]:
        """Retain the two negative-phase reflections and both QPE directions."""
        root = "GuidedHamiltonian"
        phase = "GuidedHamiltonianPhaseEstimation"
        rounds = self.amplification_rounds
        calls = self.phase_estimation._calls()
        calls[root] = {
            phase: 1 + rounds,
            "ReflectGuideAncilla(-1)": rounds,
            f"{phase}†": rounds,
            "ReflectAll(-1)": rounds,
        }
        # A negative-control reflection is self-adjoint. The inverse QFT
        # becomes the forward, bit-reversed textbook QFT in the inverse QPE.
        calls[f"{phase}†"] = {"BlackBoxPrepare†": 1, "QubitizationQPE†": 1}
        qpe = calls["QubitizationQPE"]
        calls["QubitizationQPE†"] = {
            "KaiserWindowState†": qpe["KaiserWindowState"],
            "ControlledWalk†": qpe["ControlledWalk"],
            "ZeroControlledReflection": qpe["ZeroControlledReflection"],
            "Walk†": qpe["Walk"],
            "QFTWithReverse": qpe["QFTWithReverse†"],
        }
        return symbolic_call_graph(root, calls, max_depth=max_depth)
