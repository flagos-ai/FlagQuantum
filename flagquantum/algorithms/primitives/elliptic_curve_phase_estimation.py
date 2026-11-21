# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ECPhaseEstimateR.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/cryptography/ecc/ec_phase_estimate_r.py

"""Windowed point phase estimation and source-level scalar declarations."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from functools import cached_property

from ...circuit import Circuit
from ...compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from ...core.ir import CircuitIR, Instruction
from ...core.ir.program.register_types import unsigned_register_type
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt
from ...dynamic import DynamicCircuit
from .elliptic_curve_point import ECPoint
from .elliptic_curve_window_addition import EllipticCurveWindowAddition
from .measure_qft import MeasureQFT


def elliptic_curve_phase_signature(width: SymbolicInt) -> Signature:
    """Declare the two unsigned through words of the original phase operation."""
    word = unsigned_register_type(width)
    return Signature((Register("x", word), Register("y", word)))


def elliptic_curve_phase_source_graph(
    width: SymbolicInt, window: SymbolicInt = 1
) -> SymbolicCallGraph[str]:
    """Declare the source grouped counts without constructing physical wires."""
    root = "ECPhaseEstimateR"
    addition = "ECAddR" if window == 1 else "ECWindowAddR"
    return symbolic_call_graph(
        root, {root: {addition: width // window, "MeasureQFT": 1}}
    )


def phase_addition_points(
    width: int, window: int, point: ECPoint
) -> tuple[ECPoint, ...]:
    """Return the source powers of R in their order of application."""
    if (
        type(width) is not int
        or type(window) is not int
        or not 1 <= window <= width
        or width % window
    ):
        raise ValueError(
            "Phase estimation requires a positive window dividing the width"
        )
    return tuple((1 << exponent) * point for exponent in range(0, width, window))


@dataclass(frozen=True)
class _BoundEllipticCurvePhaseEstimation:
    wires: tuple[int, ...]
    instructions: tuple[Instruction, ...]

    def append_to(self, circuit: Circuit) -> Circuit:
        if not isinstance(circuit, DynamicCircuit):
            raise ValueError(
                "Elliptic-curve phase estimation requires a DynamicCircuit"
            )
        local = CircuitIR(
            len(self.wires), self.instructions, metadata={"batch_size": circuit.bsz}
        ).validate()
        return circuit.compose(local, qubits=self.wires)


@dataclass(frozen=True)
class EllipticCurvePhaseEstimation:
    """Bind the executable windowed point phase-estimation circuit.

    The coordinate words hold Montgomery residues. Caller-owned phase wires
    start zero, are prepared with H, and are measured after the source's
    forward QFT in their original order. Keep the result slots distinct
    from arithmetic slots. Reused scratch requires every point addition
    in the superposition to satisfy the clean source domain.
    """

    width: int
    point: ECPoint
    window: int
    multiplication_window: int = 1

    @cached_property
    def addition_points(self) -> tuple[ECPoint, ...]:
        return phase_addition_points(self.width, self.window, self.point)

    @property
    def source_signature(self) -> Signature:
        return elliptic_curve_phase_signature(self.width)

    def source_callees(self) -> dict[Hashable, SymbolicInt]:
        """Retain the representative point and window parameters in model calls."""
        addition: Hashable = (
            ("ECAddR", self.width, self.point)
            if self.window == 1
            else (
                "ECWindowAddR",
                self.width,
                self.point,
                self.window,
                self.multiplication_window,
            )
        )
        return {addition: self.width // self.window, ("MeasureQFT", self.width): 1}

    def source_call_graph(self) -> SymbolicCallGraph[Hashable]:
        return symbolic_call_graph(self, {self: self.source_callees()})

    @cached_property
    def _windows(self) -> tuple[EllipticCurveWindowAddition, ...]:
        if self.window == 1:
            raise ValueError("The locked scalar ECAddR has no gate decomposition")
        return tuple(
            EllipticCurveWindowAddition(
                self.width, point, self.window, self.multiplication_window
            )
            for point in self.addition_points
        )

    def workspace_requirements(self) -> tuple[int, int]:
        """Count shared clean arithmetic wires and reusable measurement slots."""
        requirements = tuple(add.workspace_requirements() for add in self._windows)
        return max(q for q, _ in requirements), max(c for _, c in requirements)

    def make_on(
        self,
        *,
        x: tuple[int, ...],
        y: tuple[int, ...],
        phase: tuple[int, ...],
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
        result_bits: tuple[int, ...],
    ) -> _BoundEllipticCurvePhaseEstimation:
        n = self.width
        q, c = self.workspace_requirements()
        if any(len(word) != n for word in (x, y, phase, result_bits)):
            raise ValueError(
                "Phase estimation requires equal coordinate, phase and result widths"
            )
        if len(workspace) != q or len(classical_bits) != c:
            raise ValueError("Phase estimation requires exact workspace sizes")
        wires = (*x, *y, *phase, *workspace)
        for values in (wires, (*classical_bits, *result_bits)):
            if len(set(values)) != len(values) or any(
                type(v) is not int or v < 0 for v in values
            ):
                raise ValueError(
                    "Phase estimation requires distinct nonnegative wires and slots"
                )
        x = tuple(range(n))
        y = tuple(range(n, 2 * n))
        phase = tuple(range(2 * n, 3 * n))
        work = tuple(range(3 * n, len(wires)))
        circuit = DynamicCircuit(len(wires))
        for wire in phase:
            circuit.h(wire)
        for i, add in enumerate(self._windows):
            add_q, add_c = add.workspace_requirements()
            controls = phase[i * self.window : (i + 1) * self.window]
            circuit.append(
                add.make_on(
                    ctrl=controls,
                    x=x,
                    y=y,
                    workspace=work[:add_q],
                    classical_bits=classical_bits[:add_c],
                )
            )
        circuit.append(MeasureQFT(n).make_on(x=phase, result_bits=result_bits))
        return _BoundEllipticCurvePhaseEstimation(wires, circuit.to_ir().instructions)
