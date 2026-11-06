"""Run original controlled modular constant/subtraction adjoints through append.

Run: python -m examples.algorithms.register_modular_constant_adjoint
"""

from collections import Counter
from dataclasses import dataclass
from typing import Literal

import torch

import flagquantum as fq
from flagquantum.compiler.registers.controlled_modular_subtraction import (
    controlled_modular_subtract_adjoint_ir,
    controlled_modular_subtract_ir,
)
from flagquantum.compiler.registers.modular_constant_addition import (
    controlled_modular_add_constant_adjoint_ir,
    controlled_modular_add_constant_ir,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class ModularAdjoint:
    """Bind two-bit example operands, clean work and original component direction."""

    operation: Literal["constant", "subtraction"]
    inverse: bool = True
    control_value: int = 1

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        if self.operation == "constant":
            if self.control_value != 1:
                raise ValueError("Source CModAddK has positive control")
            emit_constant = (
                controlled_modular_add_constant_adjoint_ir
                if self.inverse
                else controlled_modular_add_constant_ir
            )
            gates = emit_constant(0, (1, 2), 1, 3, tuple(range(3, 13)), tuple(range(4)))
        else:
            emit = (
                controlled_modular_subtract_adjoint_ir
                if self.inverse
                else controlled_modular_subtract_ir
            )
            gates = emit(
                0,
                (1, 2),
                (3, 4),
                2,
                tuple(range(5, 13)),
                tuple(range(4)),
                control_value=self.control_value,
            )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    operations: tuple[Literal["constant", "subtraction"], ...] = (
        "constant",
        "subtraction",
    )
    for operation in operations:
        control_values = (1,) if operation == "constant" else (0, 1)
        for cv in control_values:
            if operation == "constant":
                labels = [(c, 0, y) for c in (0, 1) for y in range(3)]
                modulus, work_width = 3, 10
            else:
                labels = [(c, x, y) for c in (0, 1) for x in range(2) for y in range(2)]
                modulus, work_width = 2, 8
            coeff = torch.arange(len(labels), dtype=torch.float64)
            amps = torch.complex(coeff + 1, 3 * coeff - 5)
            amps /= amps.norm()
            state = torch.zeros(1 << 13, dtype=torch.complex128)
            expected = torch.zeros_like(state)
            for (c, x, y), amp in zip(labels, amps, strict=True):
                left = c if operation == "constant" else (c << 2) | x
                delta = -1 if operation == "constant" else x
                z = (y + delta) % modulus if c == cv else y
                state[((left << 2) | y) << work_width] = amp
                expected[((left << 2) | z) << work_width] = amp
            # This is the ordinary user append contract; all expansion belongs
            # to compiler/registers and execution stays in the existing runtime.
            op = ModularAdjoint(operation, True, cv)
            circuit = DynamicCircuit(13, inputs=state, dtype=torch.complex128)
            circuit.append(op)
            result = run_dynamic(circuit, shots=8, seed=73, strategy="trajectory")
            torch.testing.assert_close(
                result.final_states.reshape(8, -1),
                expected.expand(8, -1),
                rtol=0,
                atol=1e-12,
            )
            counts = Counter(g.name for g in circuit.to_ir().instructions)
            circuit.append(ModularAdjoint(operation, False, cv))
            restored = run_dynamic(
                circuit,
                shots=8,
                seed=73,
                strategy="trajectory",
            )
            torch.testing.assert_close(
                restored.final_states.reshape(8, -1),
                state.expand(8, -1),
                rtol=0,
                atol=1e-12,
            )
            print(
                operation,
                "cv",
                cv,
                ": complex amplitudes correct; quantum work zero; restored",
            )
            print(
                "Measured primitives:",
                dict(counts),
            )
            print(
                "Independent source model T / measurements:",
                (44, 11) if operation == "constant" else (52, 13),
            )


if __name__ == "__main__":
    main()
