# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 black_box_select.py.

"""Packed, uncontrolled Hubbard SELECT fixture from the locked source."""

from flagquantum.algorithms.primitives.hubbard_select import SelectHubbard
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.layout.auto_partition import AutoPartition
from flagquantum.compiler.registers.layout.select_partition import packed_select_layout
from flagquantum.core.ir.program.register_types import opaque_register_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature


class PackedHubbardSelectExample:
    """Bind the two packed words without adding a physical permutation."""

    def _packing(self, select: SelectHubbard) -> tuple[Signature, AutoPartition]:
        child = tuple(
            Register(name, opaque_register_type(width)) for name, width in select.layout
        )
        return packed_select_layout(child[:-1], child[-1:])

    @property
    def signature(self) -> Signature:
        return self._packing(SelectHubbard(2))[0]

    @property
    def qubit_count(self) -> int:
        select = SelectHubbard(2)
        return sum(width for _, width in select.layout) + select.requirements()[0]

    def append_to(self, circuit: Circuit) -> Circuit:
        select = SelectHubbard(2)
        signature, layout = self._packing(select)
        split = int(signature[0].bitsize)
        end = split + int(signature[1].bitsize)
        ports, _ = layout.bind(
            {"selection": tuple(range(split)), "system": tuple(range(split, end))}
        )
        return circuit.append(
            select.make_on(ports, workspace=tuple(range(end, self.qubit_count)))
        )


def main() -> None:
    """Report primitive counts from the packed append entry."""
    from collections import Counter

    from flagquantum.dynamic import DynamicCircuit

    op = PackedHubbardSelectExample()
    circuit = DynamicCircuit(op.qubit_count)
    circuit.append(op)
    counts = Counter(gate.name for gate in circuit.to_ir().instructions)
    print(
        f"{op.qubit_count} wires, {counts['t'] + counts['tdg']} T/T†, {counts['measure']} measurements"
    )


if __name__ == "__main__":
    main()
