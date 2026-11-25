"""Differentiable real orbital rotations for static spatial integrals."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ...core.algebra.pauli import scalar
from ..fermion.numerics import static_tensor


def _real_integral(value: torch.Tensor, name: str, *, ndim: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if (
        value.dtype == torch.bool
        or value.ndim != ndim
        or not torch.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be finite and have {ndim} dimensions")
    if value.is_complex() and value.numel() and float(value.imag.abs().max()) > 1e-12:
        raise ValueError(f"{name} must be real")
    return value.real.to(dtype=torch.float64)


def rotate_spatial_integrals(
    one_body: torch.Tensor,
    two_body: torch.Tensor,
    generators: torch.Tensor,
    parameters: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate real chemist integrals while preserving gradients through parameters.

    The supplied fixed generators are real antisymmetric matrices. Their linear
    combination is exponentiated, ``U = exp(sum_k parameters[k] * generators[k])``.
    Column orbitals transform as ``C_new = C U``, giving ``h_new = U.T h U``
    and ``(pq|rs)_new = U_ap U_bq U_cr U_ds (ab|cd)``. Returned tensors remain
    connected to ``parameters`` and intentionally are not MolecularProblem
    snapshots.

    Args:
        one_body: Static real symmetric spatial integrals ``h[p, q]``.
        two_body: Static real chemist integrals ``(pq|rs)``.
        generators: Static real antisymmetric matrices with shape ``(k, n, n)``.
        parameters: Finite real CPU tensor with shape ``(k,)``; may require grad.

    Returns:
        Differentiable rotated one- and two-body integral tensors.

    Raises:
        TypeError: An input is not a tensor or a static input has boolean entries.
        ValueError: Shapes, finiteness, realness, symmetries or generator
            antisymmetry do not match this real spatial-integral contract.
    """
    one = static_tensor(one_body, "one_body", ndim=2)
    two = static_tensor(two_body, "two_body", ndim=4)
    basis = static_tensor(generators, "generators", ndim=3)
    n = one.shape[0]
    if n < 1 or one.shape != (n, n) or two.shape != (n, n, n, n):
        raise ValueError("integral dimensions must describe the same nonempty space")
    for name, tensor in (("one_body", one), ("two_body", two), ("generators", basis)):
        if tensor.numel() and float(tensor.imag.abs().max()) > 1e-12:
            raise ValueError(f"{name} must be real")
    one_real, two_real, basis_real = one.real, two.real, basis.real
    if not torch.allclose(one_real, one_real.T, atol=1e-10, rtol=1e-10):
        raise ValueError("one_body must be symmetric")
    for permutation in ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)):
        if not torch.allclose(
            two_real, two_real.permute(permutation), atol=1e-10, rtol=1e-10
        ):
            raise ValueError("two_body must obey real chemist-integral symmetry")
    if basis_real.shape[1:] != (n, n):
        raise ValueError("generators must have shape (k, n_orbitals, n_orbitals)")
    if not torch.allclose(
        basis_real + basis_real.transpose(1, 2),
        torch.zeros_like(basis_real),
        atol=1e-12,
        rtol=1e-12,
    ):
        raise ValueError("orbital generators must be antisymmetric")
    if not isinstance(parameters, torch.Tensor):
        raise TypeError("parameters must be a torch.Tensor")
    if (
        parameters.device.type != "cpu"
        or not parameters.is_floating_point()
        or parameters.ndim != 1
        or parameters.shape != (len(basis_real),)
        or not torch.isfinite(parameters).all()
    ):
        raise ValueError("parameters must be a finite real CPU tensor with shape (k,)")
    coefficients = parameters.to(dtype=torch.float64)
    fixed_generators = basis_real.to(dtype=torch.float64)
    generator = torch.einsum("k,kij->ij", coefficients, fixed_generators)
    unitary = torch.linalg.matrix_exp(generator)
    h = one_real.to(dtype=torch.float64)
    eri = two_real.to(dtype=torch.float64)
    rotated_one = unitary.T @ h @ unitary
    # Contract one axis at a time to bound intermediates to O(n^4), even
    # without an optional einsum contraction-path optimizer.
    rotated_two = torch.einsum("ap,abcd->pbcd", unitary, eri)
    rotated_two = torch.einsum("bq,pbcd->pqcd", unitary, rotated_two)
    rotated_two = torch.einsum("cr,pqcd->pqrd", unitary, rotated_two)
    rotated_two = torch.einsum("ds,pqrd->pqrs", unitary, rotated_two)
    return rotated_one, rotated_two


def closed_shell_determinant_energy(
    one_body: torch.Tensor,
    two_body: torch.Tensor,
    occupied_orbitals: Sequence[int],
    *,
    constant: float = 0.0,
) -> torch.Tensor:
    """Return the differentiable energy of an explicit closed-shell determinant.

    Integrals use the real chemist convention ``(pq|rs)``. This contraction is
    a scalar objective for orbital-parameter gradients, not an eigensolver.
    """
    one = _real_integral(one_body, "one_body", ndim=2)
    two = _real_integral(two_body, "two_body", ndim=4)
    n = one.shape[0]
    if n < 1 or one.shape != (n, n) or two.shape != (n, n, n, n):
        raise ValueError("integral dimensions must describe the same nonempty space")
    occupied = tuple(occupied_orbitals)
    if (
        not occupied
        or any(type(index) is not int or index < 0 or index >= n for index in occupied)
        or len(set(occupied)) != len(occupied)
    ):
        raise ValueError(
            "occupied orbitals must be distinct indices in the integral space"
        )
    offset = scalar(constant)
    if offset.imag:
        raise ValueError("constant must be real")
    indices = torch.tensor(occupied, dtype=torch.long)
    coulomb = two[
        indices[:, None], indices[:, None], indices[None, :], indices[None, :]
    ]
    exchange = two[
        indices[:, None], indices[None, :], indices[None, :], indices[:, None]
    ]
    return (
        2 * one[indices, indices].sum() + (2 * coulomb - exchange).sum() + offset.real
    )
