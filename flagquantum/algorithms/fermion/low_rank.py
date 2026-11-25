# Adapted in part from OpenFermion, Apache-2.0. See third_party/openfermion.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Static low-rank factors of real spatial chemist integrals."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...core.algebra.pauli import count, scalar
from .numerics import static_tensor


@dataclass(frozen=True, slots=True)
class LowRankFactors:
    """H2 = correction + 1/2 sum_l eigenvalues[l] Q_l**2.

    ``Q_l = sum_pq,s factors[l,p,q] a†_(p,s) a_(q,s)``. Correction is a
    spatial one-body matrix, repeated on both spin sectors. discarded_bound
    bounds the operator norm of omitted squared factors by the triangle
    inequality, in hartree if integrals use hartree. It does not bound Trotter
    error. residual_norm is the Frobenius integral reconstruction residual.
    Tensor fields are owned static snapshots and must not be mutated in place.
    """

    eigenvalues: torch.Tensor
    factors: torch.Tensor
    correction: torch.Tensor
    discarded_bound: float
    residual_norm: float

    def __post_init__(self) -> None:
        values = static_tensor(self.eigenvalues, "eigenvalues", ndim=1)
        factors = static_tensor(self.factors, "factors", ndim=3)
        correction = static_tensor(self.correction, "correction", ndim=2)
        n = correction.shape[0]
        if n < 1 or correction.shape != (n, n) or factors.shape != (len(values), n, n):
            raise ValueError(
                "low-rank factor dimensions must match a nonempty orbital space"
            )
        for name, tensor in (
            ("eigenvalues", values),
            ("factors", factors),
            ("correction", correction),
        ):
            if tensor.numel() and float(tensor.imag.abs().max()) > 1e-12:
                raise ValueError(f"{name} must be real")
            object.__setattr__(self, name, tensor.real.contiguous())
        if not torch.allclose(
            factors, factors.transpose(1, 2), atol=1e-10, rtol=1e-10
        ) or not torch.allclose(correction, correction.T, atol=1e-10, rtol=1e-10):
            raise ValueError("density factors and correction must be symmetric")
        for name in ("discarded_bound", "residual_norm"):
            value = scalar(getattr(self, name))
            if value.imag or value.real < 0:
                raise ValueError(f"{name} must be finite, real and nonnegative")
            object.__setattr__(self, name, value.real)

    @property
    def n_orbitals(self) -> int:
        return int(self.correction.shape[0])


def decompose_two_body(
    two_body: torch.Tensor, *, rank: int | None = None
) -> LowRankFactors:
    """Decompose ``(pq|rs)`` into real symmetric density factors.

    An explicit rank retains the largest conservative norm weights, including
    negative eigenvalues when present. The default retains all eigenpairs;
    tiny roundoff modes may have zero symmetric factors. This is a static
    preprocessing boundary, not a gradient through an eigendecomposition.
    """
    tensor = static_tensor(two_body, "two_body", ndim=4)
    n = tensor.shape[0]
    if n < 1 or tensor.shape != (n, n, n, n):
        raise ValueError("two_body must be a nonempty four-index spatial tensor")
    if float(tensor.imag.abs().max()) > 1e-12:
        raise ValueError("low-rank integrals require real orbitals")
    real = tensor.real.contiguous()
    for permutation in ((1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)):
        if not torch.allclose(real, real.permute(permutation), atol=1e-10, rtol=1e-10):
            raise ValueError("two_body must obey real chemist-integral symmetry")
    eigenvalues, vectors = torch.linalg.eigh(real.reshape(n * n, n * n))
    factors = vectors.T.reshape(n * n, n, n)
    factors = (factors + factors.transpose(1, 2)) / 2
    weights = 0.5 * eigenvalues.abs() * (2 * factors.abs().sum(dim=(1, 2))).square()
    ordering = torch.argsort(weights, descending=True, stable=True)
    eigenvalues, factors, weights = (
        eigenvalues[ordering],
        factors[ordering],
        weights[ordering],
    )
    retained = n * n if rank is None else count(rank, "rank")
    if retained > n * n:
        raise ValueError("rank must not exceed the spatial pair dimension")
    values, kept = eigenvalues[:retained].clone(), factors[:retained].clone()
    reconstruction = torch.einsum("l,lpq,lrs->pqrs", values, kept, kept)
    return LowRankFactors(
        values,
        kept,
        -0.5 * torch.einsum("pqqs->ps", real),
        float(weights[retained:].sum()),
        float(torch.linalg.vector_norm(real - reconstruction)),
    )
