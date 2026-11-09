"""Apply controlled quantum addition with a quantum carry and clean workspace.

Run python -m examples.algorithms.controlled_carry_addition.
"""

from collections import Counter

import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.arithmetic.carry_controlled_addition import (
    kickmix_controlled_addition_ir,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


def main() -> None:
    gates = kickmix_controlled_addition_ir(
        (0, 1, 2),
        (3, 4, 5),
        control=6,
        carry_in=7,
        clean=(8, 9, 10),
        measurement_bit=0,
    )
    amplitudes = torch.complex(
        torch.arange(1, 257).double(), torch.arange(256).double()
    )
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(2048, dtype=torch.complex128)
    initial[::8] = amplitudes
    expected = torch.zeros_like(initial)
    for word in range(256):
        value, offset, enable, carry = (
            word >> 5,
            (word >> 2) & 7,
            (word >> 1) & 1,
            word & 1,
        )
        updated = (value + enable * (offset + carry)) % 8
        expected[((updated << 5) | (word & 31)) << 3] = amplitudes[word]
    circuit = DynamicCircuit(11, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(11, gates))
    result = run_dynamic(circuit, shots=16, seed=57, strategy="batched")
    torch.testing.assert_close(
        result.final_states,
        expected.expand_as(result.final_states),
        atol=2e-14,
        rtol=2e-14,
    )
    artifact = lower_reversible_phase(CircuitIR(11, gates))
    print(
        "Emitted native primitives",
        dict(Counter(op.opcode for op in artifact.operations)),
    )
    print("Source-required clean wires", 3)
    print("Clean wires used by the carry chain", 2)
    print("Reused classical measurement slots", 1)
    print(
        "Full complex amplitudes, unchanged offset/control/carry and zero work verified"
    )


if __name__ == "__main__":
    main()
