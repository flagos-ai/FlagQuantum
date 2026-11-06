"""Use the literal controlled AddK inverse and forward restoration via append.

Run: python -m examples.algorithms.register_inverse_addition
Expected: every two-bit target/control/spectator amplitude correct, three work
wires zero and forward addition restoring the input for both control polarities.
"""

from collections import Counter
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler.registers.controlled_addition import (
    controlled_add_constant_adjoint_ir,
    controlled_add_constant_ir,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class ConstantAddition:
    """Bind coherent control, two-bit target, constant/carry work and spectator."""

    value: int
    control_value: int
    adjoint: bool = False

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        emit = (
            controlled_add_constant_adjoint_ir
            if self.adjoint
            else controlled_add_constant_ir
        )
        gates = emit(
            0, (1, 2), self.value, (3, 4), (5,), (0,), control_value=self.control_value
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    data = torch.arange(16, dtype=torch.int64)
    control, target, spectator = data >> 3, (data >> 1) & 3, data & 1
    coeff = data.to(torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    state = torch.zeros(128, dtype=torch.complex128)
    state[((data >> 1) << 4) | spectator] = amps
    for cv in (0, 1):
        for value in (1, -1):
            mapped = (control << 2) | ((target - (control == cv) * value) % 4)
            expected = torch.zeros_like(state)
            expected[(mapped << 4) | spectator] = amps
            circuit = DynamicCircuit(7, inputs=state, dtype=torch.complex128).append(
                ConstantAddition(value, cv, adjoint=True)
            )
            actual = run_dynamic(circuit, shots=32, seed=41, strategy="trajectory")
            torch.testing.assert_close(
                actual.final_states.reshape(32, -1),
                expected.expand(32, -1),
                rtol=0,
                atol=1e-12,
            )
            counts = Counter(g.name for g in circuit.to_ir().instructions)
            restored = run_dynamic(
                circuit.append(ConstantAddition(value, cv)),
                shots=32,
                seed=41,
                strategy="trajectory",
            )
            torch.testing.assert_close(
                restored.final_states.reshape(32, -1),
                state.expand(32, -1),
                rtol=0,
                atol=1e-12,
            )
            print(
                f"control={cv}, k={value}: exact complex inverse, spectator preserved, three work wires zero; forward restores input"
            )
            print("Measured inverse gates:", dict(counts))
    print(
        "Independent source model per inverse: 4 T, 1 measurement; three quantum work wires."
    )


if __name__ == "__main__":
    main()
