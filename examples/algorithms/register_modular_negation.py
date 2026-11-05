"""Coherent modular negation with native append and both control polarities.

Run: python -m examples.algorithms.register_modular_negation
"""

from collections import Counter
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler.registers.controlled_modular_negation import (
    controlled_modular_negate_adjoint_ir,
    controlled_modular_negate_ir,
)
from flagquantum.compiler.registers.modular_negation import (
    modular_negate_adjoint_ir,
    modular_negate_ir,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class ModularNegation:
    """Bind a three-bit residue modulo five and explicit zero work wires."""

    control_value: int | None = None
    inverse: bool = False

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        target = (1, 2, 3)
        emit = modular_negate_adjoint_ir if self.inverse else modular_negate_ir
        emit_controlled = (
            controlled_modular_negate_adjoint_ir
            if self.inverse
            else controlled_modular_negate_ir
        )
        gates = (
            emit(target, 5, tuple(range(5, 11)), (0, 1))
            if self.control_value is None
            else emit_controlled(
                0,
                target,
                5,
                tuple(range(4, 11)),
                (0, 1),
                control_value=self.control_value,
            )
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    labels = [(c, x) for c in (0, 1) for x in range(5)]
    coeff = torch.arange(len(labels), dtype=torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    state = torch.zeros(1 << 11, dtype=torch.complex128)
    for (c, x), amp in zip(labels, amps, strict=True):
        state[((c << 3) | x) << 7] = amp
    for cv in (None, 0, 1):
        expected = torch.zeros_like(state)
        for (c, x), amp in zip(labels, amps, strict=True):
            y = (-x) % 5 if cv is None or c == cv else x
            expected[((c << 3) | y) << 7] = amp
        for inverse in (False, True):
            op = ModularNegation(cv, inverse)
            circuit = DynamicCircuit(11, inputs=state, dtype=torch.complex128)
            circuit.append(op)
            result = run_dynamic(circuit, shots=8, seed=41, strategy="trajectory")
            torch.testing.assert_close(
                result.final_states.reshape(8, -1),
                expected.expand(8, -1),
                rtol=0,
                atol=1e-12,
            )
            counts = Counter(g.name for g in circuit.to_ir().instructions)
            circuit.append(ModularNegation(cv, not inverse))
            restored = run_dynamic(circuit, shots=8, seed=41, strategy="trajectory")
            torch.testing.assert_close(
                restored.final_states.reshape(8, -1),
                state.expand(8, -1),
                rtol=0,
                atol=1e-12,
            )
            print(
                "control value",
                cv,
                "inverse",
                inverse,
                ": coherent amplitudes correct; work zero; exact inverse restoration verified",
            )
            print("Measured primitive gates:", dict(counts))
    print(
        "Independent source model: ordinary 24 T/6 measurements; controlled 28 T/7 measurements."
    )
    print(
        "Work: ordinary six wires; controlled seven wires; two reused classical slots."
    )


if __name__ == "__main__":
    main()
