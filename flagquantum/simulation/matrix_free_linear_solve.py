"""Matrix-free Krylov solution of a complex linear system.

An implicit time step needs ``(I - h/2 L) x = b`` for the Lindblad generator
``L``. Forming that operator costs ``(4**n)**2`` complex entries, so the system
is solved from the generator alone: GMRES only ever applies ``L``, and its cost
is set by the number of Krylov iterations rather than by the ``4**n`` state.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from .matrix_free_autograd import closed_subspace_gradient, linear_adjoint

__all__ = ("gmres",)

_COEFFICIENT_DTYPE = torch.complex128


def gmres(
    apply: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    *,
    tolerance: float,
    max_iterations: int,
    restart: int,
) -> tuple[torch.Tensor, float]:
    """Solve ``apply(x) = rhs`` for a linear ``apply`` by restarted GMRES.

    The return value is the solution together with the relative residual that
    was actually achieved, measured by applying the operator once more to the
    returned solution. That final measurement, not an internal estimate, is what
    the caller compares against ``tolerance``: an iteration only stops early
    when its own estimate says so, and a caller must never accept a step on the
    strength of an estimate alone.

    The Krylov basis uses the precision of ``rhs``; the Hessenberg coefficients
    use complex128 because they occupy only ``restart + 1`` rows and columns
    regardless of the system size.

    Raises:
        ValueError: If ``rhs`` is not a flat vector, or if ``restart`` is below
            one and would leave the Krylov basis without a vector.
    """

    if rhs.ndim != 1:
        raise ValueError("the right-hand side must be a flat vector")
    if restart < 1:
        raise ValueError("restart must admit at least one Krylov vector")
    solution = torch.zeros_like(rhs)
    rhs_norm = float(torch.linalg.vector_norm(rhs).detach())
    if rhs_norm == 0.0:
        return (
            _closed_solve_gradient(
                apply,
                solution,
                rhs,
                tolerance=tolerance,
                max_iterations=max_iterations,
                restart=restart,
            ),
            0.0,
        )
    residual = rhs
    # The norm that scales the returned solution stays a tensor. Reading it into
    # a Python float would detach the magnitude of the right-hand side from the
    # graph, and a solve whose right-hand side carries a gradient would return a
    # wrong derivative while still returning the right value. It is recomputed
    # with every residual, because a later window scales its basis by the norm
    # of the residual that opened it.
    residual_norm_tensor = torch.linalg.vector_norm(residual)
    residual_norm = float(residual_norm_tensor.detach())
    relative = 1.0
    iterations = 0
    closed = False
    while relative > tolerance:
        # One Krylov window: the restart length, clipped by whatever part of the
        # caller's iteration budget is left. A spent budget ends the solve and
        # the caller decides from the returned residual whether that is enough.
        window = min(restart, max_iterations - iterations)
        if window <= 0:
            break
        residual_norm_tensor = torch.linalg.vector_norm(residual)
        basis = [residual / residual_norm_tensor]
        hessenberg = torch.zeros(
            (restart + 1, restart), dtype=_COEFFICIENT_DTYPE, device=rhs.device
        )
        target = torch.zeros(restart + 1, dtype=_COEFFICIENT_DTYPE, device=rhs.device)
        target[0] = residual_norm_tensor
        coefficients = torch.zeros(
            (window, 1), dtype=_COEFFICIENT_DTYPE, device=rhs.device
        )
        columns = 0
        for index in range(window):
            candidate = apply(basis[index])
            for row in range(index + 1):
                projection = torch.vdot(basis[row], candidate)
                hessenberg[row, index] = projection.to(_COEFFICIENT_DTYPE)
                candidate = candidate - projection * basis[row]
            next_norm_tensor = torch.linalg.vector_norm(candidate)
            next_norm = float(next_norm_tensor.detach())
            # A complete basis has no exterior direction. Recording the norm
            # of its exactly zero residual would introduce an undefined norm
            # Hessian through a structurally unused entry.
            complete = next_norm == 0.0 and index + 1 == rhs.numel()
            hessenberg[index + 1, index] = (
                next_norm_tensor.detach() if complete else next_norm_tensor
            )
            columns = index + 1
            iterations += 1
            columns_target = target[: columns + 1].unsqueeze(1)
            # The small Hessenberg system shares the basis device. Pivoted QR can vary
            # between identical LAPACK calls, breaking serialized replay.
            # Rank-aware SVD avoids that pivot path without assuming full rank
            # at Krylov breakdown; it never materializes the physical operator.
            projected = hessenberg[: columns + 1, :columns]
            if projected.device.type == "cpu":
                approximation = torch.linalg.lstsq(
                    projected, columns_target, driver="gelsd"
                ).solution
            else:
                # Rank-aware SVD of the bounded projected system, on the same
                # device as the Krylov basis. Match gelsd's default cutoff.
                u, singular, vh = torch.linalg.svd(projected, full_matrices=False)
                cutoff = (
                    torch.finfo(singular.dtype).eps
                    * max(projected.shape)
                    * singular.max()
                )
                if bool((singular > cutoff).all()):
                    # A nearly repeated singular spectrum has an ill-conditioned
                    # SVD factor derivative even when the least-squares map is
                    # smooth. Full-rank QR avoids that artificial sensitivity.
                    q, r = torch.linalg.qr(projected, mode="reduced")
                    approximation = torch.linalg.solve_triangular(
                        r, q.mH @ columns_target, upper=True
                    )
                else:
                    inverse = torch.where(
                        singular > cutoff,
                        singular.reciprocal(),
                        torch.zeros_like(singular),
                    )
                    approximation = vh.mH @ (
                        inverse.unsqueeze(-1) * (u.mH @ columns_target)
                    )
            estimate = float(
                torch.linalg.vector_norm(
                    columns_target - hessenberg[: columns + 1, :columns] @ approximation
                ).detach()
            )
            coefficients = approximation
            if estimate <= tolerance * rhs_norm or next_norm == 0.0:
                closed = next_norm == 0.0 and columns < rhs.numel()
                break
            basis.append(candidate / next_norm_tensor)
        for index in range(columns):
            solution = solution + basis[index] * coefficients[index, 0].to(rhs.dtype)
        residual = rhs - apply(solution)
        residual_norm = float(torch.linalg.vector_norm(residual).detach())
        relative = residual_norm / rhs_norm
    if closed and relative <= tolerance:
        solution = _closed_solve_gradient(
            apply,
            solution,
            residual,
            tolerance=tolerance,
            max_iterations=max_iterations,
            restart=restart,
        )
    return solution, relative


def _closed_solve_gradient(
    apply: Callable[[torch.Tensor], torch.Tensor],
    solution: torch.Tensor,
    residual: torch.Tensor,
    *,
    tolerance: float,
    max_iterations: int,
    restart: int,
) -> torch.Tensor:
    # d x = d x_old + A^-1 d(b - A x_old). This cancels the truncated
    # subspace derivative and recovers implicit differentiation, including
    # directions outside the closed forward Krylov space.
    def adjoint_correction(gradient: torch.Tensor) -> torch.Tensor:
        adjoint = linear_adjoint(apply, gradient)
        correction, relative = gmres(
            adjoint,
            gradient,
            tolerance=tolerance,
            max_iterations=max_iterations,
            restart=restart,
        )
        if relative > tolerance:
            raise RuntimeError("the closed-subspace adjoint solve did not converge")
        return correction

    return closed_subspace_gradient(solution, residual, adjoint_correction)
