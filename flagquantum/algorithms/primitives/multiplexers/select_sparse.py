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
from ....compiler.registers.control.control_predicate import ControlRegister
from ....compiler.registers.control.control_predicate_resource_model import (
    control_predicate_cost,
)
from ....compiler.registers.control.controlled_component import controlled_component_ir
from ....compiler.registers.control.controlled_component_resource_model import (
    controlled_component_cost,
)
from ....compiler.registers.layout.partition import RegisterLayout, unpartition_register
from ....compiler.registers.lookup.bounded_majorana_select import (
    bounded_majorana_ir,
    bounded_majorana_workspace,
    source_bounded_majorana_cost,
)
from ....compiler.registers.lookup.sparse_chemistry_select_component import (
    SPARSE_SELECT_STAGES,
    sparse_select_signature,
)
from ....compiler.registers.primitives.global_phase import (
    global_phase_ir,
    scalar_phase_ir,
)
from ....core.ir import CircuitIR, Instruction
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
        return sparse_select_signature(num_spin_orb, control_value=control_value)

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
        orbitals = {"p": self.p, "q": self.q, "r": self.r, "s": self.s}
        spins = {"alpha": self.alpha, "beta": self.beta}
        return tuple(
            SelectedMajoranaFermion.make_on(
                selection=(*orbitals[orbital], spins[spin]),
                target=self.sys,
                control=(self.flag_1b,) if flagged else (),
                target_gate=pauli,
            )
            for orbital, spin, pauli, flagged in SPARSE_SELECT_STAGES
        )

    @staticmethod
    def controlled_requirements(num_spin_orb: int) -> tuple[int, int]:
        """Count conjunction work and controlled Majorana scratch and slots."""
        SelectSparse.source_signature(num_spin_orb)
        majorana = bounded_majorana_workspace(
            num_spin_orb,
            controlled=True,
            selection_width=1 if num_spin_orb == 2 else None,
        )
        return majorana + 1, majorana

    @classmethod
    def make_controlled_on(
        cls,
        num_spin_orb: int,
        ports: Mapping[str, tuple[int, ...]],
        *,
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
        control_value: int = 1,
        adjoint: bool = False,
    ) -> _BoundControlledSelectSparse:
        """Bind a native external control over the full sparse SELECT."""
        if control_value is None or type(adjoint) is not bool:
            raise ValueError(
                "Controlled sparse SELECT requires a polarity and direction"
            )
        work, slots = cls.controlled_requirements(num_spin_orb)
        if len(workspace) != work or len(classical_bits) != slots:
            raise ValueError(
                "Controlled sparse SELECT requires matching work and slots"
            )
        source_work, source_slots = cls.source_requirements(num_spin_orb)
        bound = cls.make_source_on(
            num_spin_orb,
            ports,
            workspace=workspace[:source_work],
            classical_bits=classical_bits[:source_slots],
            control_value=control_value,
        )
        _validate_wires((*bound.logical, *workspace))
        _validate_wires(classical_bits)
        return _BoundControlledSelectSparse(
            bound.select,
            bound.logical,
            workspace,
            classical_bits,
            ports["control"][0],
            control_value,
            adjoint,
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


@dataclass(frozen=True)
class _BoundControlledSelectSparse:
    select: SelectSparse
    logical: tuple[int, ...]
    workspace: tuple[int, ...]
    classical_bits: tuple[int, ...]
    control: int
    control_value: int
    adjoint: bool

    def resource_model(self) -> ComponentCost:
        """Compose native control predicates with the Majorana source models."""
        width = len(self.logical)
        phase = ComponentCost(
            "ControlledS",
            2,
            GateCost(clifford=1),
            (
                CostCall(control_predicate_cost((1, 1))),
                CostCall(control_predicate_cost((1, 1), uncompute=True)),
            ),
        )
        phase_call = CostCall(phase, bystander_qubits=width - 2)
        calls = [] if self.adjoint else [phase_call]
        majoranas = self.select._majoranas()
        size = len(self.select.sys)
        address = (size - 1).bit_length()
        for majorana in reversed(majoranas) if self.adjoint else majoranas:
            child = source_bounded_majorana_cost(
                size,
                majorana.target_gate,
                controlled=True,
                selection_width=1 if size == 2 else None,
            )
            control_count = 1 + bool(majorana.control)
            if majorana.control:
                child = controlled_component_cost(
                    child,
                    (ControlRegister(1, (1, 1)),),
                    target_width=size + address,
                )
            calls.append(
                CostCall(child, bystander_qubits=width - size - address - control_count)
            )
        if self.adjoint:
            calls.append(phase_call)
        return ComponentCost(
            "NativeControlledSelectSparse",
            width,
            GateCost(clifford=2 * (self.control_value == 0)),
            tuple(calls),
        )

    def emit(self) -> tuple[Instruction, ...]:
        predicate = self.workspace[:1]
        bits = self.classical_bits[:1]
        phase = global_phase_ir(
            -0.5 if self.adjoint else 0.5,
            controls=(self.control, self.select.flag_1b),
            values=(1, 1),
            workspace=predicate,
            classical_bits=bits,
        )
        gates = [] if self.adjoint else list(phase)
        majoranas = self.select._majoranas()
        for majorana in reversed(majoranas) if self.adjoint else majoranas:

            def emit(
                control: int, child: SelectedMajoranaFermion = majorana
            ) -> tuple[Instruction, ...]:
                return bounded_majorana_ir(
                    child.selection,
                    child.target,
                    control,
                    child.target_gate,
                    self.workspace[1:],
                    self.classical_bits[1:],
                    selection_width=1 if len(child.target) == 2 else None,
                )

            if majorana.control:
                gates.extend(
                    controlled_component_ir(
                        ((self.control, majorana.control[0]),),
                        (ControlRegister(1, (1, 1)),),
                        emit,
                        predicate,
                        bits,
                    )
                )
            else:
                gates.extend(emit(self.control))
        if self.adjoint:
            gates.extend(phase)
        flips = (Instruction("x", (self.control,)),) if self.control_value == 0 else ()
        return (*flips, *gates, *flips)

    def append_to(self, circuit: Circuit) -> Circuit:
        if max((*self.logical, *self.workspace)) >= circuit.n_qubits:
            raise ValidationError(
                "Controlled sparse SELECT binding exceeds circuit width"
            )
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits, self.emit(), metadata={"batch_size": circuit.bsz}
            ).validate()
        )
