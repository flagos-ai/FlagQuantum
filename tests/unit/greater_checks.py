"""Independent coherent-state oracle shared by both greater-than paths."""

import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic


def check_greater_amplitudes(
    m: int, n: int, q: int, c: int, gates: tuple[Instruction, ...]
) -> None:
    data = torch.arange(1 << (m + n + 1), dtype=torch.int64)
    pair = data >> 1
    z = data & 1
    predicate = (pair >> n) > (pair & ((1 << n) - 1))
    mapped = (pair << 1) | (z ^ predicate)
    coeff = data.to(torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    n_qubits = m + n + 1 + q
    state = torch.zeros(1 << n_qubits, dtype=torch.complex128)
    state[data << q] = amps
    expected = torch.zeros_like(state)
    expected[mapped << q] = amps
    circuit = DynamicCircuit(n_qubits, inputs=state, dtype=torch.complex128).append(
        FieldOperation(n_qubits, gates)
    )
    result = run_dynamic(circuit, shots=256, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(256, -1),
        expected.expand(256, -1),
        rtol=0,
        atol=1e-12,
    )
    assert (
        len(set(map(tuple, result.classical_bits[:, 40 : 40 + c].tolist()))) == 1 << c
    )
    restored = run_dynamic(
        circuit.append(FieldOperation(n_qubits, gates)),
        shots=8,
        seed=41,
        strategy="trajectory",
    )
    torch.testing.assert_close(
        restored.final_states.reshape(8, -1), state.expand(8, -1), rtol=0, atol=1e-12
    )
