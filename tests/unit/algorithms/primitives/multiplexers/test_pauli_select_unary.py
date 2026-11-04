"""Check the measurement-assisted SELECT channel on unused addresses."""

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms.primitives.multiplexers import SelectPauliLCU
from flagquantum.algorithms.primitives.multiplexers.pauli_select import (
    lower_select_pauli_lcu_to_unary_iteration,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("words", [("-Y",), ("X", "-Y", "Z")])
def test_partial_table_preserves_unused_addresses_and_coherence(
    words: tuple[str, ...],
) -> None:
    state = torch.randn(
        8, dtype=torch.complex128, generator=torch.Generator().manual_seed(19)
    )
    state /= torch.linalg.vector_norm(state)
    circuit = fq.Circuit(3, dtype=state.dtype, inputs=state)
    op = SelectPauliLCU(2, 1, words).make_on(selection=(0, 1), target=(2,))
    circuit.append(op)
    unary = lower_select_pauli_lcu_to_unary_iteration(circuit.to_ir().instructions[0])
    initial = torch.zeros((8, 2), dtype=state.dtype)
    initial[:, 0] = state
    dynamic = type(unary)(4, dtype=state.dtype, inputs=initial.flatten()).compose(unary)
    expected = torch.zeros_like(initial)
    expected[:, 0] = circuit.state().flatten()
    vector = expected.flatten()
    torch.testing.assert_close(
        dynamic.density_matrix().reshape(16, 16),
        torch.outer(vector, vector.conj()),
        atol=1e-12,
        rtol=0,
    )
