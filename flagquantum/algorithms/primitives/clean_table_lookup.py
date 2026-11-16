"""Append a clean-register table lookup or erase its retained banks."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import prod

from ...circuit import Circuit
from ...compiler.registers.lookup.qroam_bank_binding import bind_clean_banks
from ...compiler.registers.lookup.qroam_clean_erasure import (
    qroam_clean_measurement_erase_ir,
)
from ...compiler.registers.lookup.qroam_clean_metadata import qroam_clean_signature
from ...compiler.registers.lookup.source_qroam_clean import (
    QroamCleanTable,
    source_qroam_clean_ir,
)
from ...core.ir import CircuitIR, Instruction
from ...core.ir.program.signatures import Signature
from ...dynamic import DynamicCircuit


@dataclass(frozen=True)
class _BoundCleanTableLookup:
    wires: tuple[int, ...]
    instructions: tuple[Instruction, ...]

    def append_to(self, circuit: Circuit) -> Circuit:
        local = CircuitIR(
            len(self.wires), self.instructions, metadata={"batch_size": circuit.bsz}
        ).validate()
        if any(g.name == "measure" for g in self.instructions) and not isinstance(
            circuit, DynamicCircuit
        ):
            raise ValueError("Measured table lookup requires a DynamicCircuit")
        return circuit.compose(local, qubits=self.wires)


@dataclass(frozen=True)
class CleanTableLookup:
    """Bind named targets and retained junk for a concrete blocked table."""

    table: QroamCleanTable
    target_widths: tuple[int, ...]
    control_count: int = 0

    def __post_init__(self) -> None:
        if len(self.target_widths) != len(self.table.datasets) or any(
            type(width) is not int or width < 1 for width in self.target_widths
        ):
            raise ValueError(
                "Clean table lookup requires one positive target width per dataset"
            )
        if type(self.control_count) is not int or self.control_count < 0:
            raise ValueError("Table lookup control count must be a nonnegative integer")

    @property
    def selection_widths(self) -> tuple[int, ...]:
        return tuple((extent - 1).bit_length() for extent in self.table.shape)

    @property
    def source_signature(self) -> Signature:
        return qroam_clean_signature(
            self.table.shape,
            self.target_widths,
            self.table.log_block_sizes,
            selection_widths=self.selection_widths,
            controls=self.control_count,
        )

    def workspace_size(self) -> int:
        return self.table.workspace_size(self.control_count, self.selection_widths)

    def erase_result_count(self) -> int:
        return prod(self.table.block_shape) * sum(self.target_widths)

    def make_on(
        self,
        *,
        controls: tuple[int, ...],
        selections: tuple[tuple[int, ...], ...],
        outputs: Mapping[str, tuple[int, ...]],
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
        bank_bits: tuple[int, ...] | None = None,
    ) -> _BoundCleanTableLookup:
        """Load zero banks, or erase matching banks using the supplied readouts.

        The outputs use source target and junk names. Controls and selections
        are ordered big-endian. Bank readouts select the native measured
        erase path and follow dataset, block cell, and wire order.
        """
        if (
            len(controls) != self.control_count
            or tuple(map(len, selections)) != self.selection_widths
        ):
            raise ValueError(
                "Table lookup controls and selections must match the signature"
            )
        targets = tuple(
            (f"target{i}_", width) for i, width in enumerate(self.target_widths)
        )
        banks = bind_clean_banks(
            outputs, targets, prod(self.table.block_shape), junk_prefix="junk_"
        )
        words = (*selections, *(word for bank in banks for word in bank))
        wires = (*controls, *(wire for word in words for wire in word), *workspace)
        if len(set(wires)) != len(wires) or any(
            type(wire) is not int or wire < 0 for wire in wires
        ):
            raise ValueError("Table lookup requires distinct nonnegative wires")
        indices = {wire: i for i, wire in enumerate(wires)}

        def local(word: tuple[int, ...]) -> tuple[int, ...]:
            return tuple(indices[w] for w in word)

        args = (
            self.table,
            local(controls),
            tuple(local(word) for word in selections),
            tuple(tuple(local(word) for word in bank) for bank in banks),
            local(workspace),
            classical_bits,
        )
        instructions = (
            source_qroam_clean_ir(*args)
            if bank_bits is None
            else qroam_clean_measurement_erase_ir(*args, bank_bits)
        )
        return _BoundCleanTableLookup(wires, instructions)
