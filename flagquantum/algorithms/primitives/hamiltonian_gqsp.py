# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 HamiltonianSimulationByGQSP.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/hamiltonian_simulation/hamiltonian_simulation_by_gqsp.py

"""Compose PREPARE, generalized signal processing and inverse PREPARE."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from math import isfinite

from ...circuit import Circuit
from ...compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from ...compiler.registers.bits.bitwise import _validate_wires
from ...compiler.registers.layout.partition import RegisterLayout, partition_register
from ...core.classical_control import is_classical_control_instruction
from ...core.ir import CircuitIR, Instruction
from ...core.value_math import SymbolicInt, SymbolicReal
from ...dynamic import DynamicCircuit
from ...simulation.numerics.hamiltonian_qsp_coefficients import (
    hamiltonian_qsp_coefficients,
)
from ...simulation.numerics.jacobi_anger_degree import degree_jacobi_anger_approximation
from .black_box_prepare import BlackBoxPrepare
from .generalized_qsp import GeneralizedQSP, UnitaryEmitter


def hamiltonian_gqsp_source_graph(
    alpha: SymbolicReal, time: SymbolicReal, precision: SymbolicReal
) -> SymbolicCallGraph[str]:
    """Declare the source's five immediate costs for a concrete or symbolic shape."""
    degree = degree_jacobi_anger_approximation(time * alpha, precision=precision / 2)
    root = "HamiltonianSimulationByGQSP"
    return symbolic_call_graph(
        root,
        {
            root: {
                "Always(PREPARE)": 1,
                "Always(PREPARE†)": 1,
                "ZeroControlledWalk": degree,
                "PositiveControlledWalk†": degree,
                "SU2Rotation": 2 * degree + 1,
            }
        },
    )


@dataclass(frozen=True)
class HamiltonianGQSP:
    """Apply source-signed evolution with a named, coherent walk implementation.

    The walk owns the ordered selection and system ports and accepts both
    signal-control polarities. The caller provides zero-initialized outer
    PREPARE junk and workspace. Outer junk is live across GQSP and is
    separate from the workspace restored by each walker invocation.
    """

    walk_layout: RegisterLayout
    walk: UnitaryEmitter
    prepare: BlackBoxPrepare
    time: float
    precision: float

    def __post_init__(self) -> None:
        scale = self.alpha
        if (
            not isfinite(self.time * scale)
            or not isfinite(self.precision)
            or self.precision <= 0
        ):
            raise ValueError(
                "GQSP evolution requires finite scaled time and positive finite precision"
            )
        walk_widths = dict(self.walk_layout)
        if any(
            walk_widths.get(name) != width
            for name, width in self.prepare.selection_layout
        ):
            raise ValueError("PREPARE selection ports must match named walk ports")

    @property
    def alpha(self) -> float:
        scale = self.prepare.normalization
        if scale is None:
            raise ValueError("Hamiltonian walk requires PREPARE normalization")
        return scale

    @cached_property
    def degree(self) -> int:
        degree: SymbolicInt = degree_jacobi_anger_approximation(
            self.time * self.alpha, precision=self.precision / 2
        )
        assert isinstance(degree, int)
        return degree

    @cached_property
    def coefficients(self) -> tuple[complex, ...]:
        values = hamiltonian_qsp_coefficients(
            self.alpha, self.time, self.precision, degree=self.degree
        )
        assert isinstance(values, tuple)
        return values

    @cached_property
    def gqsp(self) -> GeneralizedQSP:
        from scipy.fft import fft

        return GeneralizedQSP.from_qsp_polynomial(
            self.walk_layout,
            self.walk,
            self.coefficients,
            negative_power=self.degree,
            precision=self.precision / 2,
            verify=True,
            verify_precision=1e-4,
            fft=fft,
        )

    def make_on(
        self,
        *,
        signal: int,
        target: tuple[int, ...],
        prepare_junk: tuple[int, ...] = (),
        prepare_workspace: tuple[int, ...] = (),
        walk_workspace: tuple[int, ...] = (),
        prepare_classical_bits: tuple[int, ...] = (),
    ) -> _BoundHamiltonianGQSP:
        """Bind big-endian walk ports and compose the existing three-stage path."""
        logical = (signal, *target, *prepare_junk)
        _validate_wires((*logical, *prepare_workspace))
        _validate_wires((*logical, *walk_workspace))
        ports = partition_register(target, self.walk_layout)
        selection = tuple(
            wire for name, _ in self.prepare.selection_layout for wire in ports[name]
        )
        prepare = self.prepare.emit(
            selection, prepare_junk, prepare_workspace, prepare_classical_bits
        )
        walk = self.gqsp.emit(signal, target, walk_workspace)
        unprepare = self.prepare.emit(
            selection,
            prepare_junk,
            prepare_workspace,
            prepare_classical_bits,
            adjoint=True,
        )
        return _BoundHamiltonianGQSP(
            (*prepare, *walk, *unprepare),
            max((*logical, *prepare_workspace, *walk_workspace)) + 1,
        )


@dataclass(frozen=True)
class _BoundHamiltonianGQSP:
    instructions: tuple[Instruction, ...]
    required_qubits: int

    def append_to(self, circuit: Circuit) -> Circuit:
        if circuit.n_qubits < self.required_qubits:
            raise ValueError("Hamiltonian GQSP binding exceeds circuit width")
        if not isinstance(circuit, DynamicCircuit) and any(
            is_classical_control_instruction(gate) for gate in self.instructions
        ):
            raise ValueError("Measured Hamiltonian GQSP requires a DynamicCircuit")
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits,
                self.instructions,
                metadata={"batch_size": circuit.bsz},
            ).validate()
        )
