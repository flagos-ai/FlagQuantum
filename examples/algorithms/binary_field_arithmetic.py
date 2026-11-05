"""Run polynomial-basis field operations through circuit.append(op).

Run: python -m examples.algorithms.binary_field_arithmetic
No reference packages are needed to execute this example.
"""

from collections import Counter
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler.registers.field_erasure import gf2_product_erasure
from flagquantum.compiler.registers.field_inverse import (
    gf2_inverse_ir,
    gf2_inverse_workspace_size,
)
from flagquantum.compiler.registers.field_linear import (
    gf2_constant_multiplication,
    gf2_monomial_multiplication,
)
from flagquantum.compiler.registers.field_multiplication import (
    bind_field_outputs,
    gf2_multiplication,
)
from flagquantum.compiler.registers.polynomial import binary_polynomial_product_ir
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class FieldOperation:
    """Bind an internal register decomposition to the existing append protocol."""

    n_qubits: int
    instructions: tuple[Instruction, ...]

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        return circuit.compose(CircuitIR(self.n_qubits, self.instructions))


def main() -> None:
    """Compute 3 * 5 = 4 in GF(8), modulo x^3 + x + 1."""
    x, y, target = (0, 1, 2), (3, 4, 5), (6, 7, 8)
    plan = gf2_multiplication(x, y, target, (), modulus=0b1011)
    instructions = bind_field_outputs(plan, (x, y, target))
    state = torch.zeros(512, dtype=torch.complex128)
    state[(3 << 6) | (5 << 3)] = 1
    circuit = fq.Circuit(9, inputs=state, dtype=torch.complex128)
    result = circuit.append(FieldOperation(9, instructions)).state().flatten()
    assert int(result.abs().argmax()) == (3 << 6) | (5 << 3) | 4
    print("GF(8): 3 * 5 = 4; source registers preserved")
    print("Emitted gates:", dict(Counter(g.name for g in instructions)))
    print("Output binding SWAPs:", len(instructions) - len(plan.instructions))

    work_count = gf2_inverse_workspace_size(3, modulus=0b1011)
    count = 6 + work_count
    gates = gf2_inverse_ir(x, (3, 4, 5), tuple(range(6, count)), modulus=0b1011)
    state = torch.zeros(1 << count, dtype=torch.complex128)
    state[3 << (3 + work_count)] = 1
    result = (
        fq.Circuit(count, inputs=state, dtype=torch.complex128)
        .append(FieldOperation(count, gates))
        .state()
        .flatten()
    )
    assert int(result.abs().argmax()) == ((3 << 3) | 6) << work_count
    print(f"GF(8): inverse(3) = 6; all {work_count} work wires returned to zero")
    print("Emitted clean-inverse gates:", dict(Counter(g.name for g in gates)))

    erase = gf2_product_erasure(x, y, plan.outputs[2], (0, 1, 2), modulus=0b1011)
    compute_erase = (*plan.instructions, *erase.lower(parity_bit=3))
    state = torch.zeros(512, dtype=torch.complex128)
    state[(3 << 6) | (5 << 3)] = 1
    circuit = DynamicCircuit(9, inputs=state, dtype=torch.complex128).append(
        FieldOperation(9, compute_erase)
    )
    trajectories = run_dynamic(circuit, shots=16, seed=29, strategy="trajectory")
    torch.testing.assert_close(
        trajectories.final_states.reshape(16, -1),
        state.expand(16, -1),
        rtol=0,
        atol=1e-12,
    )
    print("GF(8): compute and measured product erasure preserve all 16 trajectories")
    print("Erasure logical CZ calls:", len(erase.corrections))
    print(
        "Emitted erasure gates:",
        dict(Counter(g.name for g in erase.lower(parity_bit=3))),
    )

    scale = gf2_constant_multiplication(x, 5, modulus=0b1011)
    state = torch.zeros(8, dtype=torch.complex128)
    state[3] = 1
    result = fq.Circuit(3, inputs=state, dtype=torch.complex128).append(
        FieldOperation(3, bind_field_outputs(scale, (x,)))
    )
    assert int(result.state().abs().argmax()) == 4
    print("GF(8): in-place constant multiplication 3 * 5 = 4; no work wires")

    forward = gf2_monomial_multiplication(x, 2, modulus=0b1011)
    backward = gf2_monomial_multiplication(forward.outputs[0], -2, modulus=0b1011)
    assert backward.outputs == (x,)
    # Propagate logical wire maps between components; no intermediate SWAPs.
    gates = (*forward.instructions, *backward.instructions)
    restored = fq.Circuit(3, inputs=state, dtype=torch.complex128).append(
        FieldOperation(3, gates)
    )
    torch.testing.assert_close(restored.state().flatten(), state, rtol=0, atol=1e-12)
    print("GF(8): x^2 followed by x^-2 restores the input without binding SWAPs")

    # Polynomial multiplication has no field modulus; accumulate into nonzero h.
    gates = binary_polynomial_product_ir((0, 1), (2, 3), (4, 5, 6), ())
    state = torch.zeros(128, dtype=torch.complex128)
    state[(3 << 5) | (2 << 3) | 1] = 1
    circuit = fq.Circuit(7, inputs=state, dtype=torch.complex128).append(
        FieldOperation(7, gates)
    )
    # (x+1)*x = x^2+x = 6; h=1 becomes 1 XOR 6 = 7.
    assert int(circuit.state().abs().argmax()) == (3 << 5) | (2 << 3) | 7
    restored = circuit.append(FieldOperation(7, gates)).state().flatten()
    torch.testing.assert_close(restored, state, rtol=0, atol=1e-12)
    print("Binary polynomials: h=1 XOR (3*2) = 7; second accumulation restores h")
    print("Polynomial emitted gates:", dict(Counter(g.name for g in gates)))


if __name__ == "__main__":
    main()
