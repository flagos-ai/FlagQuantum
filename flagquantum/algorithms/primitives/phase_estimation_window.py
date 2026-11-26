# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 phase_estimation/qpe_window_state.py at
# bcb17bb25360ed68ce3bc084645db3e929799389.

"""Declare phase window output storage and bind rectangular preparation."""

from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property

from ...compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from ...compiler.registers.bits.bitwise import on_each_ir
from ...compiler.registers.phase.window_component import (
    rectangular_window_program as _rectangular_window_program,
)
from ...core.ir import Instruction
from ...core.ir.program import ProgramRecord
from ...core.ir.program.register_types import (
    fixed_point_register_type,
)
from ...core.ir.program.registers import Register, RegisterSide, RegisterType
from ...core.value_math import SymbolicInt, SymbolicReal, ceil, log2, pi
from .black_box_prepare import BlackBoxPrepare, PackedPrepareDeclaration
from .prepare_oracle import PrepareOracle


class PhaseEstimationWindow(PrepareOracle):
    """Share the source's output-only fractional phase register declaration."""

    @property
    @abstractmethod
    def m_bits(self) -> SymbolicInt:
        """Return the phase register width, retaining symbolic dimensions."""

    @cached_property
    def m_qdtype(self) -> RegisterType:
        return fixed_point_register_type(self.m_bits, self.m_bits)

    @cached_property
    def m_register(self) -> Register:
        return Register("qpe_reg", self.m_qdtype, side=RegisterSide.RIGHT)

    @property
    def selection_registers(self) -> tuple[Register, ...]:
        return (self.m_register,)


@dataclass(frozen=True)
class RectangularWindowState(PhaseEstimationWindow):
    """Prepare uniform phase amplitudes on caller-supplied clean output wires."""

    bitsize: SymbolicInt

    @property
    def m_bits(self) -> SymbolicInt:
        return self.bitsize

    @classmethod
    def from_precision_and_delta(
        cls, precision: SymbolicInt, delta: SymbolicReal
    ) -> "RectangularWindowState":
        """Preserve the source precision-bit and failure-probability formula."""
        return cls(precision + ceil(log2(2 + 1 / (2 * delta))))

    @classmethod
    def from_standard_deviation_eps(cls, eps: SymbolicReal) -> "RectangularWindowState":
        """Preserve the source standard-deviation window-width formula."""
        return cls(ceil(2 * log2(pi(eps) / eps)))

    def bind(self) -> BlackBoxPrepare:
        """Bind concrete positive-width output storage through named PREPARE.

        Examples:
            >>> state = RectangularWindowState(3).bind()
            >>> op = state.make_on(selection=(0, 1, 2), junk=())
            >>> circuit.append(op)
        """
        if self.bitsize == 0:
            raise ValueError("Source rectangular window has no zero-width tensor body")
        return PackedPrepareDeclaration(self).bind(_emit_rectangular)

    def source_graph(self) -> SymbolicCallGraph[str]:
        """Keep allocation and repeated-H declarations distinct from bound gates."""
        return symbolic_call_graph(
            "RectangularWindowState",
            {
                "RectangularWindowState": {"Allocate(phase)": 1, "OnEach(H)": 1},
                "OnEach(H)": {"H": self.bitsize},
            },
        )


def _emit_rectangular(
    ports: Mapping[str, tuple[int, ...]],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    adjoint: bool,
) -> tuple[Instruction, ...]:
    return on_each_ir(Instruction("h", (0,)), ports["qpe_reg"])


def rectangular_window_program(width: SymbolicInt) -> ProgramRecord:
    """Build the shared allocated repeated-H phase window Program."""
    return _rectangular_window_program(width)
