# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Source-aligned binding adapted from Qualtran v0.7.0 SelectSparse.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/sparse/select_bloq.py

"""SELECT for the sparse second-quantized chemistry Hamiltonian."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import pi

from ....circuit import Circuit
from ....compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from ....compiler.registers.bits.bitwise import _validate_wires
from ....compiler.registers.layout.partition import RegisterLayout, unpartition_register
from ....compiler.registers.lookup.bounded_majorana_select import (
    bounded_majorana_ir,
    bounded_majorana_workspace,
    source_bounded_majorana_cost,
)
from ....compiler.registers.primitives.global_phase import scalar_phase_ir
from ....core.ir import CircuitIR, Instruction
from ....core.ir.program.register_types import (
    bounded_unsigned_register_type,
    opaque_register_type,
    quantum_bit_type,
)
from ....core.ir.program.registers import Register
from ....core.ir.program.signatures import Signature
from ....errors import ValidationError
from .selected_majorana_fermion import SelectedMajoranaFermion


@dataclass(frozen=True)
class SelectSparse:
    """Bind sparse chemistry SELECT with no external control register.

    The four spatial orbital registers are big-endian. The spin is the least
    significant bit of each spin-orbital address. The one-body flag selects
    the additional p/q pair and contributes its S phase.
    """

    p: tuple[int, ...]
    q: tuple[int, ...]
    r: tuple[int, ...]
    s: tuple[int, ...]
    alpha: int
    beta: int
    flag_1b: int
    sys: tuple[int, ...]

    @classmethod
    def make_on(
        cls,
        *,
        p: Sequence[int],
        q: Sequence[int],
        r: Sequence[int],
        s: Sequence[int],
        alpha: int,
        beta: int,
        flag_1b: int,
        sys: Sequence[int],
    ) -> SelectSparse:
        """Bind SELECT to FlagQuantum qubit indices."""
        return cls(
            tuple(p), tuple(q), tuple(r), tuple(s), alpha, beta, flag_1b, tuple(sys)
        )

    @staticmethod
    def source_requirements(num_spin_orb: int) -> tuple[int, int]:
        """Return reusable clean wires and classical slots for the source path."""
        if type(num_spin_orb) is int and num_spin_orb == 2:
            return 0, 0
        work = bounded_majorana_workspace(num_spin_orb, controlled=True)
        return work, work - 1

    @staticmethod
    def source_signature(
        num_spin_orb: int, *, control_value: int | None = None
    ) -> Signature:
        """Declare through ports including the optional pass-through control."""
        if type(num_spin_orb) is not int or num_spin_orb < 2 or num_spin_orb % 2:
            raise ValueError(
                "Sparse SELECT requires a positive even number of spin orbitals"
            )
        if control_value is not None and (
            type(control_value) is not int or control_value not in (0, 1)
        ):
            raise ValueError("Sparse SELECT control value must be zero, one or None")
        spatial = num_spin_orb // 2
        orbital = bounded_unsigned_register_type((spatial - 1).bit_length(), spatial)
        bit = bounded_unsigned_register_type(1)
        control = (
            () if control_value is None else (Register("control", quantum_bit_type()),)
        )
        return Signature(
            (
                *control,
                *(Register(name, orbital) for name in ("p", "q", "r", "s")),
                *(Register(name, bit) for name in ("alpha", "beta", "flag_1b")),
                Register("sys", opaque_register_type(num_spin_orb)),
            )
        )

    @classmethod
    def source_layout(
        cls, num_spin_orb: int, *, control_value: int | None = None
    ) -> RegisterLayout:
        """Flatten the source signature in binding order."""
        signature = cls.source_signature(num_spin_orb, control_value=control_value)
        return tuple((reg.name, int(reg.total_bits())) for reg in signature)

    @classmethod
    def source_resource_model(
        cls, num_spin_orb: int, *, control_value: int | None = None
    ) -> ComponentCost:
        """Count the locked SELECT graph without assigning physical wires."""
        signature = cls.source_signature(num_spin_orb, control_value=control_value)
        width = int(signature.n_qubits())
        active = num_spin_orb + (num_spin_orb - 1).bit_length()
        calls = tuple(
            CostCall(
                source_bounded_majorana_cost(
                    num_spin_orb, pauli, controlled=controlled
                ),
                bystander_qubits=width - active - int(controlled),
            )
            for controlled in (True, False)
            for pauli in ("Y", "X")
        )
        return ComponentCost("SelectSparse", width, GateCost(clifford=1), calls)

    @classmethod
    def make_source_on(
        cls,
        num_spin_orb: int,
        ports: Mapping[str, tuple[int, ...]],
        *,
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
        control_value: int | None = None,
    ) -> _BoundSourceSelectSparse:
        """Bind the locked bounded stream for spatial addresses in range.

        The optional external control is a through port in the pinned
        source for either polarity. The one-body flag controls the p/q pair.
        """
        logical = unpartition_register(
            ports, cls.source_layout(num_spin_orb, control_value=control_value)
        )
        work, slots = cls.source_requirements(num_spin_orb)
        if len(workspace) != work or len(classical_bits) != slots:
            raise ValueError("Sparse SELECT requires matching work wires and slots")
        _validate_wires((*logical, *workspace))
        if classical_bits:
            _validate_wires(classical_bits)
        select = cls(
            ports["p"],
            ports["q"],
            ports["r"],
            ports["s"],
            ports["alpha"][0],
            ports["beta"][0],
            ports["flag_1b"][0],
            ports["sys"],
        )
        return _BoundSourceSelectSparse(
            select, logical, workspace, classical_bits, control_value
        )

    def _majoranas(self) -> tuple[SelectedMajoranaFermion, ...]:
        return tuple(
            SelectedMajoranaFermion.make_on(
                selection=(*orbital, spin),
                target=self.sys,
                control=control,
                target_gate=pauli,
            )
            for orbital, spin, control, pauli in (
                (self.p, self.alpha, (self.flag_1b,), "Y"),
                (self.q, self.alpha, (self.flag_1b,), "X"),
                (self.r, self.beta, (), "Y"),
                (self.s, self.beta, (), "X"),
            )
        )

    def append_to(self, circuit: Circuit) -> Circuit:
        """Append the phase and four Majorana instructions."""
        if not self.sys or len(self.sys) % 2:
            raise ValidationError(
                "Sparse SELECT needs a positive even number of spin orbitals."
            )
        width = (len(self.sys) // 2 - 1).bit_length()
        if any(len(register) != width for register in (self.p, self.q, self.r, self.s)):
            raise ValidationError(
                "Sparse SELECT orbital registers have the wrong width."
            )
        qubits = (
            *self.p,
            *self.q,
            *self.r,
            *self.s,
            self.alpha,
            self.beta,
            self.flag_1b,
            *self.sys,
        )
        if any(
            type(qubit) is not int or not 0 <= qubit < circuit.n_qubits
            for qubit in qubits
        ):
            raise ValidationError("Sparse SELECT qubits must be valid circuit indices.")
        if len(set(qubits)) != len(qubits):
            raise ValidationError("Sparse SELECT registers must use disjoint qubits.")

        circuit.gate("s", self.flag_1b)
        for majorana in self._majoranas():
            circuit.append(majorana)
        return circuit


@dataclass(frozen=True)
class _BoundSourceSelectSparse:
    select: SelectSparse
    logical: tuple[int, ...]
    workspace: tuple[int, ...]
    classical_bits: tuple[int, ...]
    control_value: int | None

    def emit(self, *, adjoint: bool = False) -> tuple[Instruction, ...]:
        """Reverse the self-adjoint Majorana calls and conjugate S if requested."""
        if type(adjoint) is not bool:
            raise ValueError("Sparse SELECT direction must be boolean")
        if len(self.select.sys) == 2:
            return self._emit_single_spatial_orbital(adjoint)
        phase = Instruction("sdg" if adjoint else "s", (self.select.flag_1b,))
        gates = [] if adjoint else [phase]
        majoranas = self.select._majoranas()
        for majorana in reversed(majoranas) if adjoint else majoranas:
            control = majorana.control[0] if majorana.control else None
            work = bounded_majorana_workspace(
                len(majorana.target), controlled=control is not None
            )
            gates.extend(
                bounded_majorana_ir(
                    majorana.selection,
                    majorana.target,
                    control,
                    majorana.target_gate,
                    self.workspace[:work],
                    self.classical_bits[: work - 1],
                )
            )
        if adjoint:
            gates.append(phase)
        return tuple(gates)

    def _emit_single_spatial_orbital(self, adjoint: bool) -> tuple[Instruction, ...]:
        # X Y on a common spin orbital is i Z; the Jordan–Wigner prefixes cancel.
        # The controlled p/q pair and the one-body S phase combine to controlled -Z.
        alpha, beta, flag = self.select.alpha, self.select.beta, self.select.flag_1b
        first, second = self.select.sys
        route = Instruction("cswap", (alpha, first, second))
        return (
            *scalar_phase_ir(flag, -pi / 2 if adjoint else pi / 2),
            Instruction("z", (first,)),
            Instruction("cz", (beta, first)),
            Instruction("cz", (beta, second)),
            route,
            Instruction("cz", (flag, first)),
            route,
            Instruction("z", (flag,)),
        )

    def resource_model(self) -> ComponentCost:
        """Count source Majorana calls with their inactive ports live."""
        return self.select.source_resource_model(
            len(self.select.sys), control_value=self.control_value
        )

    def append_to(self, circuit: Circuit) -> Circuit:
        if max((*self.logical, *self.workspace)) >= circuit.n_qubits:
            raise ValidationError("Sparse SELECT binding exceeds circuit width")
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits, self.emit(), metadata={"batch_size": circuit.bsz}
            ).validate()
        )
