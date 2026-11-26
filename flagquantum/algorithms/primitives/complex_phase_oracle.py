# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ComplexPhaseOracle.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mean_estimation/complex_phase_oracle.py

"""Encode a value, apply its signed ArcTan phase and erase the encoding."""

from dataclasses import dataclass, fields
from typing import cast

from ...circuit import Circuit, _CircuitOperation
from ...compiler._component_trace.qec_gate_cost import QECGateCost
from ...compiler.registers.fixed_point.signed_phase import signed_fractional_phase_ir
from ...compiler.registers.fixed_point.trigonometric_resource_model import (
    ARCTANGENT_PRODUCT_CALLS,
    source_arctangent_t_count,
)
from ...compiler.registers.lookup.arctan_lookup import (
    arctan_lookup_ir,
    arctan_lookup_requirements,
)
from ...compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
)
from ...core.ir import CircuitIR
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt, smax
from ._block_binding import _BoundBlockOperation, _capture_block, _validate_block_work


@dataclass(frozen=True)
class ComplexPhaseOracle:
    """Apply the encoded unsigned integer's quantized complex phase.

    The encoder's control and selection registers are public through ports.
    The encoded value, sign, fractional magnitude and lookup scratch are
    caller-owned clean workspace, restored on every measurement outcome.
    Target registers are concatenated in declaration order and interpreted
    as one big-endian unsigned word.
    """

    control_registers: tuple[Register, ...]
    selection_registers: tuple[Register, ...]
    target_registers: tuple[Register, ...]
    arctan_bitsize: int = 32

    @property
    def signature(self) -> Signature:
        return Signature((*self.control_registers, *self.selection_registers))

    def wire_symbols(self) -> tuple[str, ...]:
        controls = sum(int(reg.total_bits()) for reg in self.control_registers)
        selection = sum(int(reg.total_bits()) for reg in self.selection_registers)
        return (*(("@",) * controls), *(("ROTy",) * selection))

    def source_resource_model(
        self,
        encoder: SymbolicResourceFormula,
        encoder_adjoint: SymbolicResourceFormula,
    ) -> SymbolicResourceFormula:
        """Estimate the locked opaque ArcTan and signed Z-power calls."""
        precision = self.arctan_bitsize
        if type(precision) is not int or precision < 1:
            raise ValueError("Phase oracle resource model needs positive precision")
        public = self.signature.n_qubits()
        encoded = Signature(self.target_registers).n_qubits()
        phase_registers = precision + 1
        ports = public + encoded
        for child in (encoder, encoder_adjoint):
            if (
                isinstance(ports, int)
                and isinstance(child.logical_qubits, int)
                and child.logical_qubits < ports
            ):
                raise ValueError(
                    "Encoder estimates must include their public and target ports"
                )
        # At the first two weights the positive S and T wrappers are
        # atomic rotations. Later pairs both expand through controlled ZPow.
        second = int(precision > 1)
        tail = max(0, precision - 2)
        conjunctions = 1 + second + 2 * tail
        gates = (
            QECGateCost(
                t=2 * source_arctangent_t_count(precision, ARCTANGENT_PRODUCT_CALLS)
                + second,
                and_bloq=conjunctions,
                clifford=4 + 3 * second + 4 * tail,
                rotation=conjunctions,
                measurement=conjunctions,
            )
            .plus(encoder, 1)
            .plus(encoder_adjoint, 1)
        )
        peak = cast(
            SymbolicInt,
            smax(
                encoder.logical_qubits,
                # Source scheduling frees phase outputs before the final decoder.
                encoder_adjoint.logical_qubits,
                public + 4 * precision,
                public + encoded + phase_registers + 1,
            ),
        )
        return SymbolicResourceFormula(
            logical_qubits=peak,
            **{field.name: getattr(gates, field.name) for field in fields(gates)},
        )

    def _encoded_width(self) -> int:
        width = Signature(self.target_registers).n_qubits()
        if not isinstance(width, int):
            raise ValueError("Phase oracle encoder targets need a concrete width")
        return width

    def workspace_requirements(self, *, controlled: bool = False) -> tuple[int, int]:
        """Return all clean wires and reusable measurement slots for lookup."""
        if type(controlled) is not bool:
            raise ValueError("Phase oracle control flag must be boolean")
        encoded = self._encoded_width()
        scratch, slots = arctan_lookup_requirements(encoded, self.arctan_bitsize)
        return encoded + 1 + self.arctan_bitsize + max(scratch, int(controlled)), max(
            slots, int(controlled)
        )

    def make_on(
        self,
        encoder: _CircuitOperation,
        *,
        through: tuple[int, ...],
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
        adjoint: bool = False,
        phase_control: int | None = None,
        control_value: int = 1,
    ) -> _BoundBlockOperation:
        """Capture the coherent encoder and bind its exact inverse and work.

        The encoder may touch its public and encoded target wires. Supply
        these in register order. The internal ArcTan lookup uses a dynamic
        circuit when it needs measurement feedback. An outer phase control
        leaves compute and erase unconditional, so the inactive output is identity.
        """
        controlled = phase_control is not None
        size, slots = self.workspace_requirements(controlled=controlled)
        if len(through) != self.signature.n_qubits():
            raise ValueError("Phase oracle binding must match its public signature")
        if len(workspace) != size or len(classical_bits) != slots:
            raise ValueError("Phase oracle binding must supply its exact workspace")
        if type(adjoint) is not bool:
            raise ValueError("Phase oracle direction must be boolean")
        public = through if phase_control is None else (*through, phase_control)
        _validate_block_work(public, workspace, classical_bits)
        encoded_size = self._encoded_width()
        encoded = workspace[:encoded_size]
        sign = workspace[encoded_size]
        end = encoded_size + 1 + self.arctan_bitsize
        magnitude = workspace[encoded_size + 1 : end]
        lookup_work, lookup_slots = arctan_lookup_requirements(
            encoded_size, self.arctan_bitsize
        )
        lookup = arctan_lookup_ir(
            encoded,
            sign,
            magnitude,
            workspace[end : end + lookup_work],
            classical_bits[:lookup_slots],
        )
        required = max((*public, *workspace)) + 1
        forward = _capture_block(encoder, required, {*through, *encoded})
        inverse = Circuit(required).compose(CircuitIR(required, forward)).adjoint()
        phase = signed_fractional_phase_ir(
            sign,
            magnitude,
            adjoint=adjoint,
            control=phase_control,
            control_value=control_value,
            work=workspace[end] if controlled else None,
            classical_bit=classical_bits[0] if controlled else None,
        )
        return _BoundBlockOperation(
            (*forward, *lookup, *phase, *lookup, *inverse.to_ir().instructions),
            required,
        )
