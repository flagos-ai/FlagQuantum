"""Matrix-free action of a matrix exponential through a Krylov subspace.

An exponential step advances a time-independent linear ODE by ``exp(h L) v``
instead of by a combination of evaluated derivatives. The exponential of ``L``
is never formed: an Arnoldi basis of dimension ``m`` turns the action into the
exponential of an ``m x m`` matrix, so the cost follows the Krylov dimension
rather than the ``(4**n)**2`` entries of ``L``.

The residual this module reports is measured, not estimated. For the Arnoldi
relation ``A V = V H + h v e_m^T`` with ``A = h L`` and ``V e_1 = v / |v|``,
the computed advance ``u = |v| V exp(H) e_1`` deviates from the ODE it
approximates by exactly ``|v| h v_{m+1} (e_m^T exp(H) e_1)``: the derivative of
the computed advance along the step is ``|v| V H exp(H) e_1``, so the defect is
computed from the same quantities that produced the advance rather than from a
separate convergence indicator. A caller compares that defect, not the size of
the last Arnoldi vector, against its tolerance.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

__all__ = ("exponential_action",)

_COEFFICIENT_DTYPE = torch.complex128


def exponential_action(
    apply: Callable[[torch.Tensor], torch.Tensor],
    scale: float,
    vector: torch.Tensor,
    *,
    tolerance: float,
    max_iterations: int,
    restart: int,
) -> tuple[torch.Tensor, float]:
    """Return ``exp(scale * apply) vector`` and the relative defect it reached.

    The Krylov basis is a single window of at most ``restart`` vectors, itself
    clipped by ``max_iterations``. Unlike a restarted linear solve, a restarted
    exponential cannot reuse the basis it accumulated, so a window that misses
    ``tolerance`` returns the advance it did compute together with the defect it
    measured; the caller owns the policy for what defect is acceptable, and no
    step is ever reported as converged on the strength of an internal estimate.

    The relative defect is the measured defect divided by the norm of the
    returned advance, so a contracting step is judged against what it produced
    rather than against the state it started from.

    Raises:
        ValueError: If ``vector`` is not a flat vector, or if ``restart`` is
            below one and would leave the Krylov basis without a vector.
    """

    if vector.ndim != 1:
        raise ValueError("the exponential argument must be a flat vector")
    if restart < 1:
        raise ValueError("restart must admit at least one Krylov vector")
    # The norm stays a tensor: it scales the returned vector, so reading it into
    # a Python float would detach the magnitude of the argument from the graph
    # and return a wrong derivative together with a right value.
    norm_tensor = torch.linalg.vector_norm(vector)
    norm = float(norm_tensor.detach())
    if norm == 0.0:
        return torch.zeros_like(vector), 0.0
    window = min(restart, max_iterations)
    if window < 1:
        return vector.clone(), float("inf")
    basis = [vector / norm_tensor]
    hessenberg = torch.zeros((window + 1, window), dtype=_COEFFICIENT_DTYPE)
    coefficients = torch.zeros(1, dtype=_COEFFICIENT_DTYPE)
    relative = float("inf")
    for index in range(window):
        # Arnoldi step: apply the scaled operator and orthogonalise the result
        # against the basis built so far. The subdiagonal entry is the measured
        # norm of the part of the operator's action that leaves the basis.
        candidate = scale * apply(basis[index])
        for row in range(index + 1):
            projection = torch.vdot(basis[row], candidate)
            hessenberg[row, index] = projection.to(_COEFFICIENT_DTYPE)
            candidate = candidate - projection * basis[row]
        next_norm_tensor = torch.linalg.vector_norm(candidate)
        next_norm = float(next_norm_tensor.detach())
        hessenberg[index + 1, index] = next_norm_tensor
        # exp(H_m) e_1 is the exponential of the projected generator applied to
        # the projected starting vector; it is the only exponential computed.
        coefficients = torch.linalg.matrix_exp(hessenberg[: index + 1, : index + 1])[
            :, 0
        ]
        if next_norm == 0.0:
            # The Krylov space closed on the operator: the advance is exact.
            relative = 0.0
            break
        relative = (
            next_norm
            * float(torch.abs(coefficients[index]).detach())
            / float(torch.linalg.vector_norm(coefficients).detach())
        )
        if relative <= tolerance:
            break
        basis.append(candidate / next_norm_tensor)
    result = norm_tensor * torch.stack(
        [
            coefficient.to(vector.dtype) * basis[row]
            for row, coefficient in enumerate(coefficients)
        ]
    ).sum(dim=0)
    return result, relative
