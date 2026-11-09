"""Apply one of four components through circuit.append(op).

Run python -m examples.algorithms.indexed_component_dispatch.
"""

from collections import Counter

import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler._component_trace.resource_comparison import (
    count_emitted_instructions,
)
from flagquantum.compiler.registers.control.indexed_component import (
    indexed_component_ir,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


def main() -> None:
    labels = ("x", "z", "x", "z")
    gates = indexed_component_ir(
        ((1, 2),),
        (4,),
        lambda index, predicate: (Instruction("c" + labels[index], (predicate, 3)),),
        controls=(0,),
        workspace=(4, 5),
        classical_bits=(0, 1),
    )
    amplitudes = torch.complex(torch.arange(1, 17).double(), torch.arange(16).double())
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(64, dtype=torch.complex128)
    initial[::4] = amplitudes
    expected = torch.zeros_like(initial)
    for word in range(16):
        enable, selection, target = (word >> 3) & 1, (word >> 1) & 3, word & 1
        output, phase = word, 1
        if enable and labels[selection] == "x":
            output ^= 1
        elif enable and target:
            phase = -1
        expected[output << 2] = phase * amplitudes[word]
    circuit = DynamicCircuit(6, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(6, gates))
    result = run_dynamic(circuit, shots=16, seed=31, strategy="batched")
    torch.testing.assert_close(
        result.final_states,
        expected.expand_as(result.final_states),
        atol=2e-13,
        rtol=2e-13,
    )
    emitted = count_emitted_instructions(CircuitIR(6, gates))
    print("Emitted primitive counts", dict(Counter(gate.name for gate in gates)))
    print("Emitted T/T-dagger instructions", emitted.t_count)
    print("Parent clean work wires", 2)
    print("Complex amplitudes, original inputs and zero work verified")


if __name__ == "__main__":
    main()
