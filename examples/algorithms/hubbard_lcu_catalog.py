"""Bind the named and packed Hubbard oracles from the LCU tutorial."""

from collections.abc import Mapping
from dataclasses import dataclass, replace

from flagquantum.algorithms.primitives.black_box_prepare import (
    BlackBoxPrepare,
    PackedPrepareDeclaration,
)
from flagquantum.algorithms.primitives.hubbard_prepare import PrepareHubbard
from flagquantum.algorithms.primitives.hubbard_select import SelectHubbard
from flagquantum.algorithms.primitives.lcu_block_encoding import (
    LCUBlockEncoding,
    source_lcu_block_signature,
)
from flagquantum.algorithms.primitives.prepare_oracle import PrepareOracle
from flagquantum.algorithms.primitives.select_block_encoding import (
    SelectBlockEncoding,
    source_select_block_signature,
)
from flagquantum.compiler._component_trace.resource_comparison import (
    count_emitted_instructions,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost, CostCall
from flagquantum.compiler.registers.layout.partition import partition_register
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program.register_types import opaque_register_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.dynamic import DynamicCircuit


@dataclass(frozen=True)
class HubbardLCUExample:
    """Use the original Hubbard parameters with named or packed oracle ports."""

    side: int = 3
    packed: bool = False

    @property
    def preparation(self) -> PrepareHubbard:
        return PrepareHubbard(self.side, t=1.0, u=4.0)

    @property
    def selection(self) -> SelectHubbard:
        return SelectHubbard(self.side)

    def _prepare_declaration(self) -> PrepareOracle | PackedPrepareDeclaration:
        prepare = self.preparation
        return PackedPrepareDeclaration(prepare) if self.packed else prepare

    def _target_registers(self) -> tuple[Register, ...]:
        name, width = self.selection.layout[-1]
        return (
            Register("system" if self.packed else name, opaque_register_type(width)),
        )

    def select_signature(self) -> Signature:
        """Declare the original named or packed SELECT ports."""
        return source_select_block_signature(
            self._prepare_declaration().selection_registers, self._target_registers()
        )

    def lcu_signature(self, *, controlled: bool = False) -> Signature:
        """Declare the original named or packed LCU ports, including through junk."""
        prepare = self._prepare_declaration()
        return source_lcu_block_signature(
            prepare.selection_registers,
            prepare.junk_registers,
            self._target_registers(),
            controlled=controlled,
        )

    @property
    def prepare_oracle(self) -> BlackBoxPrepare:
        child = self.preparation
        selection = child.selection_layout
        declaration = PackedPrepareDeclaration(child)

        def emit(
            ports: Mapping[str, tuple[int, ...]],
            work: tuple[int, ...],
            slots: tuple[int, ...],
            inverse: bool,
        ) -> tuple[Instruction, ...]:
            if (len(work), len(slots)) != child.requirements():
                raise ValueError(
                    "Hubbard preparation needs its declared work and slots"
                )
            named = (
                {
                    **partition_register(ports["selection"], selection),
                    "temp": ports["junk"],
                }
                if self.packed
                else ports
            )
            return child._emit(named, work, slots, inverse)

        if not self.packed:
            return declaration.bind(emit)
        return BlackBoxPrepare(
            (("selection", sum(width for _, width in selection)),),
            (("junk", 2),),
            emit,
            child.l1_norm_of_coeffs,
        )

    def emit_select(
        self,
        control: int | None,
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        slots: tuple[int, ...],
        inverse: bool,
    ) -> tuple[Instruction, ...]:
        """Bind a self-adjoint SELECT using the prescribed control polarity."""
        child = (
            self.selection
            if control is None
            else SelectHubbard(self.side, controlled=True, control_value=control)
        )
        if (len(work), len(slots)) != child.requirements():
            raise ValueError("Hubbard SELECT needs its declared work and slots")
        if self.packed:
            named = partition_register(
                ports["selection"], self.preparation.selection_layout
            )
            named["target"] = ports["system"]
            if control is not None:
                named["control"] = ports["control"]
        else:
            named = dict(ports)
        return child._emit(named, work, slots)

    @property
    def select_block(self) -> SelectBlockEncoding:
        selected = self.selection
        prepare = self.prepare_oracle
        target = (
            (("system", selected.layout[-1][1]),)
            if self.packed
            else selected.layout[-1:]
        )
        return SelectBlockEncoding(
            prepare.selection_layout,
            target,
            lambda ports, work, slots, inverse: self.emit_select(
                None, ports, work, slots, inverse
            ),
            prepare,
        )

    def lcu_block(self, control: int | None = None) -> LCUBlockEncoding:
        """Return the blank-signal block with optional specialized SELECT."""
        return LCUBlockEncoding(
            self.select_block,
            control,
            self.emit_select if control is not None else None,
        )

    def prepare_model(self) -> ComponentCost:
        child = self.preparation.source_resource_model()
        return (
            ComponentCost("Packed PREPARE", child.qubits, calls=(CostCall(child),))
            if self.packed
            else child
        )

    def select_model(self, *, control: int | None = None) -> ComponentCost:
        child = SelectHubbard(
            self.side,
            controlled=control is not None,
            control_value=1 if control is None else control,
        ).source_resource_model()
        return (
            ComponentCost("Packed SELECT", child.qubits, calls=(CostCall(child),))
            if self.packed
            else child
        )

    def lcu_model(self, control: int | None = None) -> ComponentCost:
        prepare = self.prepare_model()
        inverse = replace(prepare, name=prepare.name + " adjoint")
        return self.lcu_block(control).resource_model(
            self.select_model(control=control), prepare, inverse
        )


def hubbard_lcu_circuit(
    example: HubbardLCUExample, *, control: int | None = None, adjoint: bool = False
) -> DynamicCircuit:
    """Append the example using sequentially reused clean work and result slots."""
    block = example.lcu_block(control)
    controls = (0,) if control is not None else ()
    start = len(controls)
    ancilla_end = start + block.ancilla_bitsize
    junk_end = ancilla_end + block.resource_bitsize
    through = junk_end + block.system_bitsize
    prepare_work, prepare_slots = example.preparation.requirements()
    select_work, select_slots = SelectHubbard(
        example.side, controlled=control is not None
    ).requirements()
    work = tuple(range(through, through + max(prepare_work, select_work)))
    slots = tuple(range(max(prepare_slots, select_slots)))
    circuit = DynamicCircuit(through + len(work))
    circuit.append(
        block.make_on(
            selection=tuple(range(start, ancilla_end)),
            junk=tuple(range(ancilla_end, junk_end)),
            target=tuple(range(junk_end, through)),
            control=controls,
            prepare_workspace=work[:prepare_work],
            prepare_classical_bits=slots[:prepare_slots],
            select_workspace=work[:select_work],
            select_classical_bits=slots[:select_slots],
            adjoint=adjoint,
        )
    )
    return circuit


def hubbard_packed_prepare_circuit(
    example: HubbardLCUExample, *, adjoint: bool = False
) -> DynamicCircuit:
    """Append the preparation stage on selection, junk and clean work in that order."""
    packed = example.prepare_oracle
    split = packed.selection_width
    boundary = split + packed.junk_width
    work, bits = example.preparation.requirements()
    circuit = DynamicCircuit(boundary + work)
    circuit.append(
        packed.make_on(
            selection=tuple(range(split)),
            junk=tuple(range(split, boundary)),
            workspace=tuple(range(boundary, boundary + work)),
            classical_bits=tuple(range(bits)),
            adjoint=adjoint,
        )
    )
    return circuit


if __name__ == "__main__":
    circuit = hubbard_packed_prepare_circuit(HubbardLCUExample(2, packed=True))
    counts = count_emitted_instructions(circuit.to_ir())
    assert counts.t_count == 47 and counts.unconditional["measure"] == 3
    print(f"Packed 2×2 Hubbard PREPARE on {counts.declared_qubits} wires")
    print(f"{counts.t_count} T/T† and {counts.unconditional['measure']} measurements")
