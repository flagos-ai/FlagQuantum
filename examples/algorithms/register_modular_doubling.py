"""Append source-exact modular doubling and consume its explicit output ports.

Run: python -m examples.algorithms.register_modular_doubling
"""

from collections import Counter
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum.compiler.registers.modular.modular_doubling import (
    modular_double_outputs,
    modular_double_requirements,
)
from flagquantum.compiler.registers.modular.modular_doubling_component import (
    modular_doubling_program,
)
from flagquantum.compiler.registers.modular.modular_halving import (
    modular_half_ir,
    modular_half_outputs,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program.register_types import unsigned_register_type
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


@dataclass(frozen=True)
class ModularDoubling:
    """Bind a residue word and caller-owned clean work."""

    target: tuple[int, ...]
    workspace: tuple[int, ...]
    inverse: bool = False
    modulus: int = 5

    @property
    def outputs(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return (modular_half_outputs if self.inverse else modular_double_outputs)(
            self.target, self.workspace
        )

    def append_to(self, circuit: fq.Circuit) -> fq.Circuit:
        width = len(self.target)
        _, slots = modular_double_requirements(width)
        if self.inverse:
            gates = modular_half_ir(
                self.target, self.modulus, self.workspace, tuple(range(slots))
            )
        else:
            expected_output = self.outputs[0]
            work = (self.workspace[1], self.workspace[0], *self.workspace[2:])
            bound = bind_register_program(
                modular_doubling_program(unsigned_register_type(width), self.modulus),
                (self.target,),
                clean=work,
                input_ownership=((self.target[0], False),),
                allow_workspace_outputs=True,
            )
            assert bound.outputs == (expected_output,)
            gates = bound.circuit.instructions
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def main() -> None:
    target, work = (0, 1, 2), tuple(range(3, 14))
    x = torch.arange(5, dtype=torch.float64)
    amps = torch.complex(x + 1, 3 * x - 5)
    amps /= amps.norm()
    state = torch.zeros(1 << 14, dtype=torch.complex128)
    for i, a in enumerate(amps):
        state[i << 11] = a
    for inverse in (False, True):
        op = ModularDoubling(target, work, inverse)
        out, clean = op.outputs
        expected = torch.zeros_like(state)
        for i, a in enumerate(amps):
            y = (i + 5 * (i & 1)) // 2 if inverse else (2 * i) % 5
            index = sum(
                ((y >> (2 - j)) & 1) << (13 - wire) for j, wire in enumerate(out)
            )
            expected[index] = a
        circuit = DynamicCircuit(14, inputs=state, dtype=torch.complex128)
        circuit.append(op)
        result = run_dynamic(circuit, shots=8, seed=41, strategy="trajectory")
        torch.testing.assert_close(
            result.final_states.reshape(8, -1),
            expected.expand(8, -1),
            rtol=0,
            atol=1e-12,
        )
        counts = Counter(g.name for g in circuit.to_ir().instructions)
        reverse = ModularDoubling(out, clean, not inverse)
        assert reverse.outputs == (target, work)
        circuit.append(reverse)
        restored = run_dynamic(circuit, shots=8, seed=41, strategy="trajectory")
        torch.testing.assert_close(
            restored.final_states.reshape(8, -1),
            state.expand(8, -1),
            rtol=0,
            atol=1e-12,
        )
        print(
            "inverse",
            inverse,
            ": coherent amplitudes correct; returned work zero; ports and input restored",
        )
        print("Output ports:", out, "clean work:", clean)
        print("Measured primitive gates:", dict(counts))
    print(
        "Independent source model: 28 T, seven measurements, fourteen total qubits per operation."
    )


if __name__ == "__main__":
    main()
