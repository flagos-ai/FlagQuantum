"""Exact CPU Toffoli execution preserves amplitudes, ownership and gradients."""

from itertools import permutations

import pytest
import torch

import flagquantum as fq
from flagquantum.simulation.matrices import TOFFOLI_MATRIX
from flagquantum.simulation.statevector.operations import (
    _apply_fixed_permutation,
    _apply_matrix_layout,
)
from flagquantum.simulation.statevector.reversible_cpu import _apply_toffoli_cpu

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("wires", list(permutations(range(4), 3)))
@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_cpu_toffoli_matches_dense_on_every_basis_state(
    wires: tuple[int, int, int], dtype: torch.dtype
) -> None:
    state = torch.eye(16, dtype=dtype)
    actual = _apply_fixed_permutation(state, "ccx", wires, 4)
    expected = _apply_matrix_layout(state, TOFFOLI_MATRIX.to(dtype), wires, 4)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, torch.eye(16, dtype=dtype), rtol=0, atol=0)
    assert actual.data_ptr() != state.data_ptr()


@pytest.mark.parametrize("wires", ((0, 2, 4), (4, 2, 0), (3, 0, 2)))
@pytest.mark.parametrize("layout", ("contiguous", "strided", "conjugate"))
def test_cpu_toffoli_matches_dense_complex_gradient_and_inverse(
    wires: tuple[int, int, int], layout: str
) -> None:
    generator = torch.Generator().manual_seed(1253)
    source = torch.randn(3, 64, dtype=torch.complex128, generator=generator)
    state = source[:, :32].clone() if layout == "contiguous" else source[:, ::2]
    if layout == "conjugate":
        state = state.conj()
    state = state.detach().requires_grad_()
    before = state.detach().clone()
    reference = before.clone().requires_grad_()
    actual = _apply_fixed_permutation(state, "ccx", wires, 5)
    expected = _apply_matrix_layout(reference, TOFFOLI_MATRIX, wires, 5)
    weights = torch.randn(3, 32, dtype=torch.complex128, generator=generator)
    actual_gradient = torch.autograd.grad((actual * weights).real.sum(), state)[0]
    expected_gradient = torch.autograd.grad((expected * weights).real.sum(), reference)[
        0
    ]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
    torch.testing.assert_close(state, before, rtol=0, atol=0)
    torch.testing.assert_close(
        _apply_fixed_permutation(actual, "ccx", wires, 5), before, rtol=0, atol=0
    )


def test_cpu_public_toffoli_uses_permutation_without_dense_matmul(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = torch.arange(32, dtype=torch.float64).to(torch.complex128)
    circuit = fq.Circuit(5, dtype=torch.complex128, inputs=initial).ccx(4, 1, 2)
    expected = _apply_matrix_layout(initial[None], TOFFOLI_MATRIX, (4, 1, 2), 5)

    def reject_bmm(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("CPU Toffoli must not construct a dense multiplication")

    monkeypatch.setattr(torch, "bmm", reject_bmm)
    torch.testing.assert_close(circuit.state(), expected, rtol=0, atol=0)
    torch.testing.assert_close(initial, torch.arange(32).to(torch.complex128))


@pytest.mark.parametrize("requires_grad", (False, True))
def test_public_toffoli_chain_reuses_only_owned_inference_storage(
    monkeypatch: pytest.MonkeyPatch, requires_grad: bool
) -> None:
    from flagquantum.simulation.statevector import operations

    original = _apply_toffoli_cpu
    calls: list[tuple[bool, bool]] = []

    def observe(
        state: torch.Tensor,
        qubits: tuple[int, ...],
        n_qubits: int,
        *,
        inplace: bool = False,
    ) -> torch.Tensor:
        result = original(state, qubits, n_qubits, inplace=inplace)
        calls.append((inplace, state.data_ptr() == result.data_ptr()))
        return result

    monkeypatch.setattr(operations, "_apply_toffoli_cpu", observe)
    generator = torch.Generator().manual_seed(1253)
    initial = torch.randn(32, dtype=torch.complex128, generator=generator)
    initial.requires_grad_(requires_grad)
    before = initial.detach().clone()
    reference = before.clone().requires_grad_(requires_grad)
    actual = (
        fq.Circuit(5, dtype=torch.complex128, inputs=initial)
        .ccx(0, 2, 4)
        .ccx(3, 1, 2)
        .state()
    )
    expected = _apply_matrix_layout(reference[None], TOFFOLI_MATRIX, (0, 2, 4), 5)
    expected = _apply_matrix_layout(expected, TOFFOLI_MATRIX, (3, 1, 2), 5)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(initial, before, rtol=0, atol=0)
    assert calls == [(False, False), (True, not requires_grad)]
    if requires_grad:
        weights = torch.randn(1, 32, dtype=torch.complex128, generator=generator)
        actual_gradient = torch.autograd.grad((actual * weights).real.sum(), initial)[0]
        expected_gradient = torch.autograd.grad(
            (expected * weights).real.sum(), reference
        )[0]
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_public_explicit_matrix_keeps_its_action_instead_of_the_fixed_cpu_gate(
    monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype
) -> None:
    from flagquantum.simulation.statevector import operations

    phases = torch.arange(8, dtype=torch.float64) * 0.19
    matrix = torch.diag(torch.exp(1j * phases)).to(dtype)
    initial = torch.arange(8, dtype=torch.float64).to(dtype)
    wires = (2, 0, 1)
    ir = fq.CircuitIR(
        3,
        (fq.Instruction("ccx", wires, matrix=matrix),),
        dtype=str(dtype).removeprefix("torch."),
    )
    circuit = fq.Circuit.from_ir(ir, inputs=initial)
    expected = _apply_matrix_layout(initial[None], matrix, wires, 3)

    def reject_fixed_gate(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("An explicit matrix must retain its own numerical action")

    monkeypatch.setattr(operations, "_apply_toffoli_cpu", reject_fixed_gate)
    torch.testing.assert_close(circuit.state(), expected, rtol=0, atol=0)
    torch.testing.assert_close(initial, torch.arange(8).to(dtype), rtol=0, atol=0)


@pytest.mark.parametrize("wires", ((0, 8, 16), (8, 16, 0), (16, 0, 8)))
@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
@pytest.mark.parametrize("layout", ("contiguous", "transposed"))
@pytest.mark.parametrize("requires_grad", (False, True))
def test_cpu_toffoli_preserves_large_batched_layout_and_gradient(
    wires: tuple[int, int, int],
    dtype: torch.dtype,
    layout: str,
    requires_grad: bool,
) -> None:
    generator = torch.Generator().manual_seed(57319)
    width = 1 << 17
    state = (
        (
            torch.randn(3, width, dtype=dtype, generator=generator)
            if layout == "contiguous"
            else torch.randn(width, 3, dtype=dtype, generator=generator).T
        )
        .detach()
        .requires_grad_(requires_grad)
    )
    before = state.detach().clone()
    reference = before.clone().requires_grad_(requires_grad)
    # The private call explicitly admits owned storage; the public chain tests
    # independently protect caller inputs at the execution boundary.
    actual = _apply_toffoli_cpu(state, wires, 17, inplace=True)
    expected = _apply_matrix_layout(reference, TOFFOLI_MATRIX.to(dtype), wires, 17)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert (actual.data_ptr() == state.data_ptr()) is (not requires_grad)
    if requires_grad:
        torch.testing.assert_close(state, before, rtol=0, atol=0)
        weights = torch.randn(actual.shape, dtype=dtype, generator=generator)
        actual_gradient = torch.autograd.grad((actual * weights).real.sum(), state)[0]
        reference_gradient = torch.autograd.grad(
            (expected * weights).real.sum(), reference
        )[0]
        torch.testing.assert_close(actual_gradient, reference_gradient, rtol=0, atol=0)
    else:
        torch.testing.assert_close(state, expected, rtol=0, atol=0)
