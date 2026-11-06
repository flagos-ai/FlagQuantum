"""Executable single-factorization SELECT from arXiv:2011.03494, Eq. A4 and Fig. 13."""

from __future__ import annotations

from dataclasses import dataclass

from ....circuit import Circuit
from ....compiler.registers.bitwise import _validate_wires
from ....compiler.registers.bounded_majorana_select import (
    bounded_majorana_ir,
    bounded_majorana_workspace,
)
from ....compiler.registers.multi_control import multi_and_ir
from ....core.classical_control import validate_classical_control
from ....core.ir import CircuitIR, Instruction
from ....dynamic import DynamicCircuit
from ....errors import ValidationError


@dataclass(frozen=True)
class SelectSingleFactorization:
    """Apply the selected Jordan–Wigner pair on the successful branch.

    Spatial addresses p and q are big-endian and must select populated orbitals.
    System wires are ordered by orbital, then spin. Both success flags must be
    one. Pass the optional nonzero-factor flag for the second SELECT in the
    full encoding. An optional coefficient sign applies a phase on the same
    active branch. All caller-owned work starts and returns to zero.
    """

    p: tuple[int, ...]
    q: tuple[int, ...]
    spin: int
    succ_pq: int
    succ_l: int
    system: tuple[int, ...]
    workspace: tuple[int, ...]
    l_ne_zero: int | None = None
    coefficient_sign: int | None = None

    @staticmethod
    def workspace_size(spin_orbitals: int, *, additional_control: bool = False) -> int:
        """Return clean predicate wires followed by reusable Majorana work."""
        predicate_size = 1 + int(additional_control)
        if type(spin_orbitals) is int and spin_orbitals == 2:
            return predicate_size
        return predicate_size + bounded_majorana_workspace(
            spin_orbitals, controlled=True
        )

    def append_to(self, circuit: Circuit) -> Circuit:
        """Append to a DynamicCircuit, reserving feedback slots after existing ones."""
        size = len(self.system)
        needed = self.workspace_size(
            size, additional_control=self.l_ne_zero is not None
        )
        address_width = (size // 2 - 1).bit_length()
        if len(self.p) != address_width or len(self.q) != address_width:
            raise ValidationError("Single-factorization SELECT address width mismatch")
        if len(self.workspace) != needed:
            raise ValidationError(
                f"Single-factorization SELECT requires {needed} work wires"
            )
        wires = (
            *self.p,
            *self.q,
            self.spin,
            self.succ_pq,
            self.succ_l,
            *(q for q in (self.l_ne_zero, self.coefficient_sign) if q is not None),
            *self.system,
            *self.workspace,
        )
        _validate_wires(wires)
        if max(wires) >= circuit.n_qubits:
            raise ValidationError(
                "Single-factorization SELECT binding exceeds circuit width"
            )
        if not isinstance(circuit, DynamicCircuit):
            raise ValidationError(
                "Single-factorization SELECT requires a DynamicCircuit"
            )
        slot = validate_classical_control(
            circuit.to_ir().instructions, n_qubits=circuit.n_qubits
        )
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits,
                self._emit(slot),
                metadata={"batch_size": circuit.bsz},
            ).validate()
        )

    def _emit(self, slot: int) -> tuple[Instruction, ...]:
        controls: tuple[int, ...] = (self.succ_pq, self.succ_l)
        if self.l_ne_zero is not None:
            controls = (*controls, self.l_ne_zero)
        count = len(controls) - 1
        predicate_work, majorana_work = self.workspace[:count], self.workspace[count:]
        predicate = predicate_work[-1]
        polarity = (1,) * len(controls)
        selected: tuple[Instruction, ...]
        if len(self.system) == 2:
            # The only spatial pair is diagonal, so spin chooses which -Z to apply.
            swap = Instruction("cswap", (self.spin, *self.system))
            selected = (
                swap,
                Instruction("cz", (predicate, self.system[0])),
                swap,
                Instruction("z", (predicate,)),
            )
        else:
            bits = tuple(range(slot + count, slot + count + len(majorana_work) - 1))
            selected = (
                *bounded_majorana_ir(
                    (*self.p, self.spin),
                    self.system,
                    predicate,
                    "Y",
                    majorana_work,
                    bits,
                ),
                *bounded_majorana_ir(
                    (*self.q, self.spin),
                    self.system,
                    predicate,
                    "X",
                    majorana_work,
                    bits,
                ),
                Instruction("s", (predicate,)),
            )
        sign = (
            ()
            if self.coefficient_sign is None
            else (Instruction("cz", (predicate, self.coefficient_sign)),)
        )
        return (
            *multi_and_ir(controls, polarity, predicate_work),
            *sign,
            *selected,
            *multi_and_ir(
                controls,
                polarity,
                predicate_work,
                classical_bits=tuple(range(slot, slot + count)),
                uncompute=True,
            ),
        )
