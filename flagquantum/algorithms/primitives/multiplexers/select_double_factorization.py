"""Programmed DF number selection from Appendix C step 4 of arXiv:2011.03494."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ....circuit import Circuit
from ....compiler.registers.arithmetic.addition import add_adjoint_ir, add_ir
from ....compiler.registers.bits.bitwise import _validate_wires, xor_ir
from ....compiler.registers.control.multi_control import multi_and_ir
from ....compiler.registers.layout.partition import RegisterLayout, unpartition_register
from ....compiler.registers.lookup.qroam_bank_binding import (
    bind_clean_banks,
    clean_bank_output_layout,
)
from ....compiler.registers.lookup.source_qroam_clean import (
    QroamCleanTable,
    source_qroam_clean_ir,
    source_qroam_clean_uncompute_ir,
)
from ....compiler.registers.phase.phase_gradient import phase_gradient_state_ir
from ....compiler.registers.phase.programmed_givens import programmed_givens_ir
from ....compiler.registers.primitives.controlled_swap import controlled_swap_ir
from ....core.classical_control import validate_classical_control
from ....core.ir import CircuitIR, Instruction
from ....dynamic import DynamicCircuit


@dataclass(frozen=True)
class SelectDoubleFactorization:
    """Apply a signed number reflection in the selected spin and orbital basis.

    The system is spin-major. Each angle row gives one unsigned binary
    fraction of a turn per adjacent pair, ordered by increasing left orbital.
    The lookup uses a fresh address, offset + post-alias p. Its entire bank,
    the gradient and caller-owned scratch start and end in zero.
    """

    orbitals: int
    angle_bits: int
    table: QroamCleanTable

    @classmethod
    def from_angle_words(
        cls,
        orbitals: int,
        rows: Sequence[Sequence[int]],
        *,
        angle_bits: int = 8,
        log_block_size: int = 0,
    ) -> "SelectDoubleFactorization":
        """Build one QROAM bank per adjacent orbital pair."""
        if type(orbitals) is not int or orbitals < 1 or len(rows) < orbitals:
            raise ValueError(
                "DF rotations require at least one row per spatial orbital"
            )
        if type(angle_bits) is not int or angle_bits < 1:
            raise ValueError("DF rotation precision must be positive")
        if any(
            len(row) != orbitals - 1
            or any(
                type(word) is not int or not 0 <= word < 1 << angle_bits for word in row
            )
            for row in rows
        ):
            raise ValueError(
                "DF rotation rows must have one valid word per adjacent pair"
            )
        columns = (
            tuple(zip(*rows, strict=True)) if orbitals > 1 else ((0,) * len(rows),)
        )
        if orbitals == 1 and log_block_size != 0:
            raise ValueError("A single orbital has no rotation lookup to block")
        return cls(
            orbitals,
            angle_bits,
            QroamCleanTable((len(rows),), (log_block_size,), columns),
        )

    @property
    def _targets(self) -> RegisterLayout:
        return tuple((f"angle_{i}", self.angle_bits) for i in range(self.orbitals - 1))

    @property
    def _outputs(self) -> RegisterLayout:
        return clean_bank_output_layout(
            self._targets, self.table.block_shape[0], junk_prefix="junk_"
        )

    def layout(self, *, additional_control: bool = False) -> RegisterLayout:
        """Name the signal controls, system and clean lookup registers."""
        address = (self.table.shape[0] - 1).bit_length() if self.orbitals > 1 else 0
        return (
            ("offset", address),
            ("p", (self.orbitals - 1).bit_length()),
            ("spin", 1),
            ("succ_l", 1),
            ("succ_p", 1),
            ("theta", 1),
            *((("l_ne_zero", 1),) if additional_control else ()),
            ("system", 2 * self.orbitals),
            ("address", address),
            ("gradient", self.angle_bits if self.orbitals > 1 else 0),
            *self._outputs,
        )

    def workspace_size(self, *, additional_control: bool = False) -> int:
        """Return clean work wires and the size of the reusable slot pool."""
        predicate = 1 + int(additional_control)
        if self.orbitals == 1:
            return predicate
        address = (self.table.shape[0] - 1).bit_length()
        return max(
            predicate,
            address - 1,
            self.angle_bits - 1,
            self.table.workspace_size(0, (address,)),
        )

    def make_on(
        self, ports: Mapping[str, tuple[int, ...]], *, workspace: tuple[int, ...]
    ) -> "_BoundDoubleFactorizationSelect":
        """Bind the involution for circuit.append on a DynamicCircuit."""
        extra = "l_ne_zero" in ports
        wires = unpartition_register(ports, self.layout(additional_control=extra))
        _validate_wires((*wires, *workspace))
        if len(workspace) != self.workspace_size(additional_control=extra):
            raise ValueError("DF SELECT requires matching clean work")
        return _BoundDoubleFactorizationSelect(
            self, dict(ports), workspace, max((*wires, *workspace)) + 1
        )

    def _emit(
        self,
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        slots: tuple[int, ...],
    ) -> tuple[Instruction, ...]:
        system = ports["system"][: self.orbitals]
        other_spin = ports["system"][self.orbitals :]
        swap = controlled_swap_ir(ports["spin"][0], system, other_spin)
        controls = (*ports["succ_l"], *ports["succ_p"], *ports.get("l_ne_zero", ()))
        values = (1,) * len(controls)
        predicate = work[: len(controls) - 1]
        phase = (
            *multi_and_ir(controls, values, predicate),
            Instruction("cz", (predicate[-1], system[0])),
            Instruction("cz", (predicate[-1], ports["theta"][0])),
            *multi_and_ir(
                controls,
                values,
                predicate,
                classical_bits=slots[: len(predicate)],
                uncompute=True,
            ),
        )
        if self.orbitals == 1:
            return (*swap, *phase, *swap)

        address = ports["address"]
        copy_offset = xor_ir(ports["offset"], address)
        carry = len(address) - 1
        offset = add_ir(ports["p"], address, work[:carry], slots[:carry])
        unoffset = add_adjoint_ir(ports["p"], address, work[:carry], slots[:carry])
        banks = bind_clean_banks(
            {name: ports[name] for name, _ in self._outputs},
            self._targets,
            self.table.block_shape[0],
            junk_prefix="junk_",
        )
        lookup_work = self.table.workspace_size(0, (len(address),))
        lookup_args = (
            self.table,
            (),
            (address,),
            banks,
            work[:lookup_work],
            slots[:lookup_work],
        )
        lookup = source_qroam_clean_ir(*lookup_args)
        erase = source_qroam_clean_uncompute_ir(*lookup_args)
        gradient = ports["gradient"]
        carry = self.angle_bits - 1

        def rotations(adjoint: bool) -> tuple[Instruction, ...]:
            pairs = range(self.orbitals - 1)
            return tuple(
                gate
                for i in (pairs if adjoint else reversed(pairs))
                for gate in programmed_givens_ir(
                    system[i],
                    system[i + 1],
                    ports[f"angle_{i}"],
                    gradient,
                    work[:carry],
                    slots[:carry],
                    adjoint=adjoint,
                )
            )

        return (
            *copy_offset,
            *offset,
            *lookup,
            *swap,
            *phase_gradient_state_ir(gradient, exponent=-1),
            *rotations(False),
            *phase,
            *rotations(True),
            *phase_gradient_state_ir(gradient, inverse=True, exponent=-1),
            *swap,
            *erase,
            *unoffset,
            *copy_offset,
        )


@dataclass(frozen=True)
class _BoundDoubleFactorizationSelect:
    select: SelectDoubleFactorization
    ports: Mapping[str, tuple[int, ...]]
    workspace: tuple[int, ...]
    required_qubits: int

    def append_to(self, circuit: Circuit) -> Circuit:
        if circuit.n_qubits < self.required_qubits or not isinstance(
            circuit, DynamicCircuit
        ):
            raise ValueError("DF SELECT needs a DynamicCircuit with sufficient width")
        start = validate_classical_control(
            circuit.to_ir().instructions, n_qubits=circuit.n_qubits
        )
        slots = tuple(range(start, start + len(self.workspace)))
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits,
                self.select._emit(self.ports, self.workspace, slots),
                metadata={"batch_size": circuit.bsz},
            ).validate()
        )
