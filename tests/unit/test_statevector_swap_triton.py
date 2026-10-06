"""Correctness and refusal boundaries for the SV-014 SWAP-sequence kernel."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from flagquantum.kernels.triton.statevector_swap import (
    apply_complex64_local_swap_sequence,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu, pytest.mark.triton]


def _reference(
    state: torch.Tensor,
    swaps: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    output = state
    n_qubits = state.shape[1].bit_length() - 1
    for first, second in swaps:
        output = (
            output.reshape((state.shape[0],) + (2,) * n_qubits)
            .transpose(first + 1, second + 1)
            .reshape(state.shape)
        )
    return output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "swaps",
    (
        ((0, 1), (2, 3)),
        ((0, 11), (1, 10), (2, 9), (3, 8)),
        ((7, 2), (11, 0), (4, 6), (1, 10), (3, 9)),
        tuple((index, 11 - index) for index in range(8)),
    ),
)
def test_local_swap_sequence_matches_layout_reference(
    swaps: tuple[tuple[int, int], ...],
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(261_014)
    state = torch.randn(
        3,
        1 << 12,
        generator=generator,
        device="cuda",
        dtype=torch.complex64,
    )

    actual = apply_complex64_local_swap_sequence(state, swaps=swaps)
    expected = _reference(state, swaps)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_local_swap_sequence_rejects_non_cuda_state() -> None:
    with pytest.raises(ValueError, match="contiguous CUDA complex64"):
        apply_complex64_local_swap_sequence(
            torch.randn(1, 16, dtype=torch.complex64),
            swaps=((0, 1), (2, 3)),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_local_swap_sequence_rejects_unsupported_contracts() -> None:
    state = torch.randn(2, 16, device="cuda", dtype=torch.complex64)

    with pytest.raises(ValueError, match="two through eight"):
        apply_complex64_local_swap_sequence(state, swaps=((0, 1),))
    with pytest.raises(ValueError, match="two through eight"):
        apply_complex64_local_swap_sequence(state, swaps=((0, 1),) * 9)
    with pytest.raises(ValueError, match="distinct local"):
        apply_complex64_local_swap_sequence(state, swaps=((0, 1), (2, 2)))
    with pytest.raises(ValueError, match="distinct local"):
        apply_complex64_local_swap_sequence(state, swaps=((0, 1), (2, 4)))
    with pytest.raises(ValueError, match="forward-only"):
        apply_complex64_local_swap_sequence(
            state.requires_grad_(),
            swaps=((0, 1), (2, 3)),
        )
    with pytest.raises(ValueError, match="output must be distinct"):
        apply_complex64_local_swap_sequence(
            state.detach(),
            swaps=((0, 1), (2, 3)),
            output=state.detach(),
        )
