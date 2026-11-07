"""Accumulate measured table parity with no quantum scratch via circuit.append(op).

Run python -m examples.algorithms.guarded_classical_parity.
"""

from itertools import product

import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.artifacts.reversible_phase import (
    ReversiblePhaseOperation as Op,
)
from flagquantum.compiler.artifacts.reversible_phase import ReversiblePhaseProgram
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


def main() -> None:
    program = ReversiblePhaseProgram(
        1,
        4,
        (
            Op("bit_store0", (), result_bit=3),
            Op("hmr", (0,), result_bit=0),
            Op("push_condition", (), condition_bit=0),
            Op("bit_invert", (), result_bit=3, condition_bit=1),
            Op("bit_invert", (), result_bit=3, condition_bit=2),
            Op("pop_condition", ()),
        ),
        input_bits=(1, 2),
    )
    for a, b in product((0, 1), repeat=2):
        inputs = (
            ClassicalXor((), 1, bool(a)).instruction(),
            ClassicalXor((), 2, bool(b)).instruction(),
        )
        op = FieldOperation(1, (*inputs, *bind_reversible_phase_execution(program)))
        circuit = DynamicCircuit(1, dtype=torch.complex128)
        circuit.append(op)
        result = run_dynamic(circuit, shots=64, seed=83, strategy="batched")
        measured = result.classical_bits[:, 0]
        assert torch.all(result.classical_bits[:, 3] == measured * (a ^ b))
        assert result.statistics["measurement_count"] == 64
        assert bool(torch.all(result.final_states[:, 1].abs() < 1e-12))
    print("All measured parity masks verified; one data qubit, no quantum scratch")


if __name__ == "__main__":
    main()
