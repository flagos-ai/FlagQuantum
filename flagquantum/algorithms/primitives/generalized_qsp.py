# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 GeneralizedQSP.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/qsp/generalized_qsp.py

"""Bind finite complementary polynomials and named unitary ports for append."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property

from ...circuit import Circuit
from ...compiler.registers.bits.bitwise import _validate_wires
from ...compiler.registers.layout.partition import RegisterLayout, partition_register
from ...compiler.registers.phase.generalized_qsp import generalized_qsp_ir
from ...core.ir import CircuitIR, Instruction
from ...simulation.numerics.qsp_phases import qsp_phase_factors

UnitaryEmitter = Callable[
    [int, Mapping[str, tuple[int, ...]], tuple[int, ...], int | None, bool],
    tuple[Instruction, ...],
]


@dataclass(frozen=True)
class GeneralizedQSP:
    """Compose the source phase recursion and signal sequence on named U ports.

    Coefficients are in ascending degree order. The caller supplies a
    complementary pair and a coherent unitary emitter, including both control
    polarities and its adjoint. Workspace belongs to that emitter and must be
    restored after every invocation. This internal operation uses continuous
    rotations; it does not perform synthesis or certify complementarity.
    """

    unitary_layout: RegisterLayout
    unitary: UnitaryEmitter
    p: Sequence[complex]
    q: Sequence[complex]
    negative_power: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "p", tuple(self.p))
        object.__setattr__(self, "q", tuple(self.q))
        if len(self.p) < 2 or len(self.q) < 2:
            raise ValueError("GQSP Polynomial must have degree at least 1")

    @cached_property
    def _phases(self) -> tuple[tuple[float, ...], tuple[float, ...], float]:
        theta, phi, lambd = qsp_phase_factors(self.p, self.q)
        return tuple(map(float, theta)), tuple(map(float, phi)), lambd

    def emit(
        self,
        signal: int,
        target: tuple[int, ...],
        workspace: tuple[int, ...] = (),
    ) -> tuple[Instruction, ...]:
        """Bind ordered big-endian ports and lower through the existing Compiler."""
        if "signal" in dict(self.unitary_layout):
            raise ValueError("GQSP reserves the signal port name")
        ports = partition_register(target, self.unitary_layout)
        _validate_wires((signal, *target, *workspace))

        def bound(value: int | None, adjoint: bool) -> tuple[Instruction, ...]:
            return self.unitary(signal, ports.copy(), workspace, value, adjoint)

        theta, phi, lambd = self._phases
        return generalized_qsp_ir(
            signal,
            (*target, *workspace),
            theta,
            phi,
            lambd,
            unitary=bound,
            negative_power=self.negative_power,
        )

    def make_on(
        self,
        *,
        signal: int,
        target: tuple[int, ...],
        workspace: tuple[int, ...] = (),
    ) -> "_BoundGeneralizedQSP":
        """Bind the operation for the existing circuit.append protocol."""
        return _BoundGeneralizedQSP(self, signal, target, workspace)


@dataclass(frozen=True)
class _BoundGeneralizedQSP:
    operation: GeneralizedQSP
    signal: int
    target: tuple[int, ...]
    workspace: tuple[int, ...]

    def append_to(self, circuit: Circuit) -> Circuit:
        """Append on the same physical binding, preserving the circuit batch."""
        if any(
            wire >= circuit.n_qubits
            for wire in (self.signal, *self.target, *self.workspace)
        ):
            raise ValueError("GQSP binding exceeds circuit width")
        gates = self.operation.emit(self.signal, self.target, self.workspace)
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits, gates, metadata={"batch_size": circuit.bsz}
            ).validate()
        )
