"""Use non-self-adjoint indexed children, then undo them with circuit.append(op).

Run python -m examples.algorithms.indexed_component_round_trip.
"""

from collections import Counter

import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.indexed_component import indexed_component_ir
from flagquantum.compiler.registers.phase_family import phase_family_ir
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


def main() -> None:
    gates: list[Instruction] = []
    for adjoint in (False, True):

        def emit(
            index: int, predicate: int, reverse: bool = adjoint
        ) -> tuple[Instruction, ...]:
            return phase_family_ir(
                "s" if index == 0 else "t",
                2,
                controls=(predicate,),
                values=(1,),
                adjoint=reverse,
            )

        gates.extend(
            indexed_component_ir(
                ((1,),),
                (2,),
                emit,
                controls=(0,),
                workspace=(3,),
                classical_bits=(0,),
                adjoint=adjoint,
            )
        )
    amplitudes = torch.complex(torch.arange(1, 9).double(), torch.arange(8).double())
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(16, dtype=torch.complex128)
    initial[::2] = amplitudes
    circuit = DynamicCircuit(4, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(4, tuple(gates)))
    result = run_dynamic(circuit, shots=16, seed=37, strategy="batched")
    torch.testing.assert_close(
        result.final_states,
        initial.expand_as(result.final_states),
        atol=2e-13,
        rtol=2e-13,
    )
    counts = Counter(gate.name for gate in gates)
    print("Emitted T/T-dagger instructions", counts["t"] + counts["tdg"])
    print("Unsynthesized controlled phase instructions", counts["cphase"])
    print("Measured erasures", counts["measure"])
    print("Parent clean quantum wires", 1)
    print("Complex input restored and parent work cleared")


if __name__ == "__main__":
    main()
