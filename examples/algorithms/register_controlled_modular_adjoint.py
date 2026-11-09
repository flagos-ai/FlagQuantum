"""Append controlled modular addition and its original exact adjoint.

Run: python -m examples.algorithms.register_controlled_modular_adjoint
"""

from collections import Counter
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler.registers.modular.controlled_modular_addition import (
    controlled_modular_add_ir,
)
from flagquantum.compiler.registers.modular.controlled_modular_addition_adjoint import (
    controlled_modular_add_adjoint_ir,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class ControlledModularAddition:
    """Bind coherent two-bit residues modulo three and explicit clean work."""

    inverse: bool = False
    control_value: int = 1

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        emit = (
            controlled_modular_add_adjoint_ir
            if self.inverse
            else controlled_modular_add_ir
        )
        gates = emit(
            0,
            (1, 2),
            (3, 4),
            3,
            tuple(range(5, 13)),
            tuple(range(4)),
            control_value=self.control_value,
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    labels = [(c, x, y) for c in (0, 1) for x in range(4) for y in range(3)]
    v = torch.arange(len(labels), dtype=torch.float64)
    amps = torch.complex(v + 1, 3 * v - 5)
    amps /= amps.norm()
    state = torch.zeros(1 << 13, dtype=torch.complex128)
    for (c, x, y), amp in zip(labels, amps, strict=True):
        state[((c << 4) | (x << 2) | y) << 8] = amp
    for cv in (0, 1):
        for inverse in (False, True):
            expected = torch.zeros_like(state)
            for (c, x, y), amp in zip(labels, amps, strict=True):
                z = (y + (-x if inverse else x)) % 3 if c == cv else y
                expected[((c << 4) | (x << 2) | z) << 8] = amp
            op = ControlledModularAddition(inverse, cv)
            circuit = DynamicCircuit(13, inputs=state, dtype=torch.complex128).append(
                op
            )
            result = run_dynamic(circuit, shots=8, seed=41, strategy="trajectory")
            torch.testing.assert_close(
                result.final_states.reshape(8, -1),
                expected.expand(8, -1),
                rtol=0,
                atol=1e-12,
            )
            counts = Counter(g.name for g in circuit.to_ir().instructions)
            restored = run_dynamic(
                circuit.append(ControlledModularAddition(not inverse, cv)),
                shots=8,
                seed=41,
                strategy="trajectory",
            )
            torch.testing.assert_close(
                restored.final_states.reshape(8, -1),
                state.expand(8, -1),
                rtol=0,
                atol=1e-12,
            )
            print(
                "inverse",
                inverse,
                "control value",
                cv,
                ": coherent amplitudes correct; work zero; input restored",
            )
            print("Measured primitives:", dict(counts))
    print(
        "Independent source model per direction: 44 T, eleven measurements, thirteen total qubits."
    )


if __name__ == "__main__":
    main()
