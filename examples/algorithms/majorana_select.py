"""Compile the three-target Selected Majorana example with bounded traversal."""

from collections import Counter

from flagquantum.algorithms.primitives.multiplexers.selected_majorana_fermion import (
    SelectedMajoranaFermion,
)
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.drawer.text_drawer import draw_text
from flagquantum.dynamic import DynamicCircuit


def main() -> None:
    """Show the original three-target X example and distinguish its counts."""
    op = SelectedMajoranaFermion.make_bounded_on(
        control=(0,),
        selection=(1, 2),
        target=(3, 4, 5),
        target_gate="X",
        workspace=(6, 7, 8),
        classical_bits=(0, 1),
    )
    circuit = DynamicCircuit(9)
    circuit.append(op)
    ir = circuit.to_ir()
    print(draw_text(ir, show_all_qubits=True))
    counts = Counter(gate.name for gate in ir.instructions)
    print(
        f"Emitted {counts['t'] + counts['tdg']} T/T† and {counts['measure']} measurements"
    )
    print("Source model:", estimate_component_resources(op.resource_model()))


if __name__ == "__main__":
    main()
