"""Independent encoded-product references and the labelled user workflow.

Numerical models and their physical conventions remain caller-owned.
"""

from __future__ import annotations

import math

import pytest
import torch

import flagquantum as fq
import flagquantum.lindblad as fql

pytestmark = pytest.mark.unit


def test_invariant_number_sector_matches_full_space_and_analytic_gradient() -> None:
    dtype = torch.complex128
    labels = ((0, 1), (1, 0))
    embedding = torch.eye(4, dtype=dtype)[:, [2 * a + b for a, b in labels]]
    h = torch.zeros((4, 4), dtype=dtype)
    h[1, 2] = h[2, 1] = 1
    complement = torch.eye(4, dtype=dtype) - embedding @ embedding.mH
    assert torch.count_nonzero(complement @ h @ embedding) == 0
    theta = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    reduced_h = embedding.mH @ h @ embedding
    options = fq.ExecutionOptions(require_gradients=True)
    full = fql.run(
        theta * h,
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.3, 1.0],
        dimensions=(2, 2),
        method="krylov-exponential",
        return_density_matrices=True,
        options=options,
    )
    plan = fql.plan(
        theta * reduced_h,
        [1.0, 0.0],
        [0.0, 0.3, 1.0],
        dimensions=(2,),
        method="krylov-exponential",
        return_density_matrices=True,
        options=options,
    )
    reduced = fql.run(plan)
    for full_rho, small_rho in zip(
        full.density_matrices, reduced.density_matrices, strict=True
    ):
        torch.testing.assert_close(
            embedding @ small_rho @ embedding.mH, full_rho, atol=1e-11, rtol=1e-11
        )
    probability = reduced.populations[-1, labels.index((1, 0))]
    (gradient,) = torch.autograd.grad(probability, theta)
    assert probability.item() == pytest.approx(math.sin(0.4) ** 2, abs=1e-11)
    assert gradient.item() == pytest.approx(math.sin(0.8), abs=1e-10)
    # Current plans name the encoded product space, not the physical sector.
    assert plan.dimensions == (2,)
    assert "basis_labels" not in plan.to_dict()["request"]


def test_basis_permutation_requires_permuting_states_and_operators_together() -> None:
    dtype = torch.complex128
    h = torch.tensor([[0.2, 0.4 + 0.1j], [0.4 - 0.1j, -0.2]], dtype=dtype)
    permutation = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)
    initial = torch.tensor([1.0, 0.0], dtype=dtype)
    original = fql.run(
        h,
        initial,
        [0.0, 0.3, 1.0],
        dimensions=(2,),
        method="krylov-exponential",
        return_density_matrices=True,
    )
    reordered = fql.run(
        permutation @ h @ permutation.mH,
        permutation @ initial,
        [0.0, 0.3, 1.0],
        dimensions=(2,),
        method="krylov-exponential",
        return_density_matrices=True,
    )
    for rho, permuted in zip(
        original.density_matrices, reordered.density_matrices, strict=True
    ):
        torch.testing.assert_close(
            permutation @ rho @ permutation.mH, permuted, atol=1e-11, rtol=1e-11
        )


def test_projection_of_a_non_invariant_sector_would_erase_physical_leakage() -> None:
    dtype = torch.complex128
    x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)
    h = torch.kron(x, torch.eye(2, dtype=dtype))
    embedding = torch.eye(4, dtype=dtype)[:, [1, 2]]
    complement = torch.eye(4, dtype=dtype) - embedding @ embedding.mH
    assert torch.max(torch.abs(complement @ h @ embedding)).item() == 1
    assert torch.count_nonzero(embedding.mH @ h @ embedding) == 0
    full = fql.run(
        0.4 * h,
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0],
        dimensions=(2, 2),
        method="krylov-exponential",
    )
    assert full.populations[-1, 3].item() == pytest.approx(
        math.sin(0.4) ** 2, abs=1e-11
    )
    assert full.populations[-1, 1].item() < 1


def test_declared_ising_convention_and_encoded_density_reference() -> None:
    dtype = torch.complex128
    identity = torch.eye(2, dtype=dtype)
    f = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=dtype) / math.sqrt(2)
    r = torch.diag(torch.exp(torch.tensor([-1j, 3j], dtype=dtype) * math.pi / 8))
    b = f.mH @ r @ f
    for matrix in (f, r, b):
        torch.testing.assert_close(matrix.mH @ matrix, identity, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(r @ b @ r, b @ r @ b, atol=1e-14, rtol=1e-14)
    h = (
        math.pi / 4 * torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)
        - math.pi / 8 * identity
    )
    torch.testing.assert_close(torch.matrix_exp(-1j * h), b, atol=1e-14, rtol=1e-14)
    result = fql.run(
        h,
        [1.0, 0.0],
        [0.0, 1.0, 2.0],
        dimensions=(2,),
        method="krylov-exponential",
        return_density_matrices=True,
    )
    for index, matrix in ((1, b), (2, b @ b)):
        state = matrix[:, 0]
        expected = torch.outer(state, state.conj())
        torch.testing.assert_close(
            result.density_matrices[index], expected, atol=1e-11, rtol=1e-11
        )
    assert result.populations[1, 1].item() == pytest.approx(0.5, abs=1e-11)
    assert result.populations[2, 1].item() == pytest.approx(1.0, abs=1e-11)


def test_reference_example_reports_current_api_limits(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from examples.constrained_basis_reference import main

    main()
    assert capsys.readouterr().out == (
        "One-excitation transfer: 0.50000000\n"
        "Control derivative: 1.00000000\n"
        "Fusion-channel probability: 0.50000000\n"
        "Braid relation verified: True\n"
        "Plan retains declared basis labels: True\n"
    )
