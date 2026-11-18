# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 GeneralizedQSP.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/qsp/generalized_qsp.py

"""Declare QSP metadata and bind known coefficients on named unitary ports."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING

from ...circuit import Circuit
from ...compiler._component_trace.resource_generalization import (
    RotationCall,
    ZeroCostCall,
)
from ...compiler.registers.bits.bitwise import _validate_wires
from ...compiler.registers.layout.partition import RegisterLayout, partition_register
from ...compiler.registers.phase.generalized_qsp import generalized_qsp_ir
from ...compiler.registers.phase.su2_rotation import su2_rotation_calls
from ...core.ir import CircuitIR, Instruction
from ...core.ir.program.register_types import opaque_register_type, quantum_bit_type
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt, SymbolicReal
from ...core.value_parameters import is_symbolic_value
from ...simulation.numerics.qsp_complement import qsp_complementary_polynomial
from ...simulation.numerics.qsp_grid import assert_is_qsp_polynomial
from ...simulation.numerics.qsp_phases import qsp_phase_factors
from ...simulation.value_dimensions import SymbolicArrayShape, value_length

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ...simulation.numerics.qsp_grid import _NumericArray

UnitaryEmitter = Callable[
    [int, Mapping[str, tuple[int, ...]], tuple[int, ...], int | None, bool],
    tuple[Instruction, ...],
]


def source_gqsp_signature(unitary_signature: Signature) -> Signature:
    """Prepend the signal to the complete unchanged unitary register descriptors."""
    return Signature((Register("signal", quantum_bit_type()), *unitary_signature))


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
    p: Sequence[complex] | SymbolicArrayShape
    q: Sequence[complex] | SymbolicArrayShape
    negative_power: SymbolicInt = 0
    precision: SymbolicReal = field(default=1e-11, kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.p, SymbolicArrayShape):
            object.__setattr__(self, "p", tuple(self.p))
        if not isinstance(self.q, SymbolicArrayShape):
            object.__setattr__(self, "q", tuple(self.q))
        for polynomial in (self.p, self.q):
            degree = value_length(polynomial) - 1
            if not is_symbolic_value(degree) and degree <= 0:
                raise ValueError("GQSP Polynomial must have degree at least 1")

    @property
    def signature(self) -> Signature:
        """Describe scalar binding ports with the existing opaque layout types."""
        return source_gqsp_signature(
            Signature(
                Register(name, opaque_register_type(width))
                for name, width in self.unitary_layout
            )
        )

    def is_symbolic(self) -> bool:
        """Retain source classification of coefficients and Laurent power."""
        return is_symbolic_value(self.p, self.q, self.negative_power)

    @property
    def precision_per_rotation(self) -> SymbolicReal:
        """Retain source coefficient-count plus one synthesis allocation."""
        return self.precision / (value_length(self.p) + 1)

    @classmethod
    def from_qsp_polynomial(
        cls,
        unitary_layout: RegisterLayout,
        unitary: UnitaryEmitter,
        p: Sequence[complex] | SymbolicArrayShape,
        *,
        negative_power: SymbolicInt = 0,
        precision: SymbolicReal = 0,
        verify: bool = False,
        verify_precision: float = 1e-7,
        fft: Callable[[_NumericArray], NDArray[np.complex128]] | None = None,
    ) -> GeneralizedQSP:
        """Apply the source grid-check, complement, then construction sequence.

        Verification requires an explicit FFT provider. Its finite-grid bound
        check is not a whole-circle certificate. Preserve complement failures
        and the source factory's zero precision default. Shape-only input retains
        the direct constructor precision default and skips numerical verification.
        """
        if isinstance(p, SymbolicArrayShape) or is_symbolic_value(p):
            return cls(unitary_layout, unitary, p, p, negative_power)
        if verify:
            if fft is None:
                raise ValueError("GQSP verification requires an FFT provider")
            assert_is_qsp_polynomial(p, fft=fft)
        q = qsp_complementary_polynomial(
            p, verify=verify, verify_precision=verify_precision
        )
        return cls(
            unitary_layout,
            unitary,
            p,
            tuple(q),
            negative_power,
            precision=precision,
        )

    @cached_property
    def _phases(self) -> tuple[tuple[float, ...], tuple[float, ...], float]:
        if isinstance(self.p, SymbolicArrayShape) or isinstance(
            self.q, SymbolicArrayShape
        ):
            raise ValueError("Cannot compute phases for shape-only GQSP polynomials")
        theta, phi, lambd = qsp_phase_factors(self.p, self.q)
        return tuple(map(float, theta)), tuple(map(float, phi)), lambd

    @cached_property
    def signal_rotation_calls(
        self,
    ) -> tuple[tuple[ZeroCostCall, RotationCall, RotationCall, RotationCall], ...]:
        """Declare original SU2 child angles and precision without synthesizing."""
        theta, phi, lambd = self._phases
        return tuple(
            su2_rotation_calls(
                t,
                p,
                lambd if i == 0 else 0,
                precision=self.precision_per_rotation,
            )
            for i, (t, p) in enumerate(zip(theta, phi, strict=True))
        )

    def emit(
        self,
        signal: int,
        target: tuple[int, ...],
        workspace: tuple[int, ...] = (),
    ) -> tuple[Instruction, ...]:
        """Bind ordered big-endian ports and lower through the existing Compiler."""
        if self.is_symbolic():
            raise ValueError("Cannot emit symbolic GQSP coefficients or Laurent power")
        if type(self.negative_power) is not int:
            raise ValueError("GQSP negative_power must be a nonnegative integer")
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
