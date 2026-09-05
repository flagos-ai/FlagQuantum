from __future__ import annotations

import pytest
import torch

from flagquantum.simulation.noisy_statevector import (
    apply_amplitude_damping_batched,
    apply_kraus_batched,
    expectation_z,
)

pytestmark = pytest.mark.unit


def _generators(count: int) -> list[torch.Generator]:
    return [torch.Generator().manual_seed(index) for index in range(count)]


def test_generic_kraus_kernel_samples_and_normalizes_a_certain_branch() -> None:
    state = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]], dtype=torch.complex64)
    zero = torch.zeros((2, 2), dtype=torch.complex64)
    x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex64)

    evolved = apply_kraus_batched(state, (zero, x), (0,), 1, _generators(2))

    torch.testing.assert_close(
        evolved, torch.tensor([[[0, 1]], [[0, 1]]], dtype=torch.complex64)
    )
    torch.testing.assert_close(expectation_z(evolved, 1), -torch.ones((2, 1, 1)))


def test_amplitude_damping_fast_path_handles_a_certain_jump() -> None:
    state = torch.tensor([[[0.0, 1.0]], [[0.0, 1.0]]], dtype=torch.complex64)
    operators = (
        torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.complex64),
        torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.complex64),
    )

    evolved = apply_amplitude_damping_batched(state, operators, 0, 1, _generators(2))

    torch.testing.assert_close(
        evolved, torch.tensor([[[1, 0]], [[1, 0]]], dtype=torch.complex64)
    )
    torch.testing.assert_close(expectation_z(evolved, 1), torch.ones((2, 1, 1)))
