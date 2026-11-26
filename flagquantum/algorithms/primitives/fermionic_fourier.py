# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 TwoBitFFFT.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/qft/two_bit_ffft.py

"""Two-qubit fermionic Fourier operations through shared Compiler owners."""

from ...compiler._component_trace.resource_generalization import (
    CliffordCall,
    PhaseCall,
    RotationCall,
)
from ...compiler._component_trace.resource_model import ComponentCost
from ...compiler._component_trace.symbolic_math import SymbolicReal
from ...compiler.registers.phase import fermionic_fourier as _fourier
from ...compiler.registers.resources import fermionic_fourier_model as _model
from ...core.ir import Instruction
from ...core.ir.program import ProgramRecord
from ...core.ir.program.signatures import Signature


def source_two_bit_fermionic_fourier_signature() -> Signature:
    """Describe the two through bit ports in the source order."""
    return _fourier.source_two_bit_fermionic_fourier_signature()


def source_two_bit_fermionic_fourier_calls(
    k: int, n: int, *, epsilon: SymbolicReal = 1e-10
) -> dict[RotationCall | PhaseCall | CliffordCall, int]:
    """Keep the source precision and combined T/adjoint-T declaration."""
    return _model.source_two_bit_fermionic_fourier_calls(k, n, epsilon=epsilon)


def source_two_bit_fermionic_fourier_cost(k: int, n: int) -> ComponentCost:
    """Price the source calls separately from measured physical primitives."""
    return _model.source_two_bit_fermionic_fourier_cost(k, n)


def two_bit_fermionic_fourier_ir(
    x: int, y: int, k: int, n: int, *, adjoint: bool = False
) -> tuple[Instruction, ...]:
    """Bind the shared literal gate sequence or its exact inverse."""
    return _fourier.two_bit_fermionic_fourier_ir(x, y, k, n, adjoint=adjoint)


def two_bit_fermionic_fourier_program(
    k: int, n: int, *, adjoint: bool = False
) -> ProgramRecord:
    """Use shared typed calls without changing caller-owned work or phases."""
    return _fourier.two_bit_fermionic_fourier_program(k, n, adjoint=adjoint)
