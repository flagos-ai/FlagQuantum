"""Apply a measured register increment with circuit.append(op).

Run python -m examples.algorithms.measured_register_increment.
"""

from collections import Counter

import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.kickmix_increment import kickmix_increment_ir
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


def main() -> None:
    gates = kickmix_increment_ir(
        (0, 1, 2, 3), control=4, clean=(5, 6), measurement_bit=0
    )
    amplitudes = torch.complex(torch.arange(1, 33).double(), torch.arange(32).double())
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(128, dtype=torch.complex128)
    initial[::4] = amplitudes
    expected = torch.zeros_like(initial)
    for word in range(32):
        value, control = word >> 1, word & 1
        expected[((((value + control) % 16) << 1) | control) << 2] = amplitudes[word]
    circuit = DynamicCircuit(7, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(7, gates))
    result = run_dynamic(circuit, shots=16, seed=19, strategy="batched")
    torch.testing.assert_close(
        result.final_states,
        expected.expand_as(result.final_states),
        atol=2e-14,
        rtol=2e-14,
    )
    artifact = lower_reversible_phase(CircuitIR(7, gates))
    print(
        "Emitted native primitives",
        dict(Counter(op.opcode for op in artifact.operations)),
    )
    print("Clean quantum carry wires", 2)
    print("Reused classical measurement slots", 1)
    print("Complex amplitudes, preserved control and zero carries verified")


if __name__ == "__main__":
    main()
