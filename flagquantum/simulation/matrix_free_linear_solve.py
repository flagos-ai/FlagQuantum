"""Matrix-free Krylov solution of a complex linear system.

An implicit time step needs ``(I - h/2 L) x = b`` for the Lindblad generator
``L``. Forming that operator costs ``(4**n)**2`` complex entries, so the system
is solved from the generator alone: GMRES only ever applies ``L``, and its cost
is set by the number of Krylov iterations rather than by the ``4**n`` state.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

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
    rhs_norm = float(torch.linalg.vector_norm(rhs))
    if rhs_norm == 0.0:
        return solution, 0.0
    residual = rhs
    residual_norm = rhs_norm
    relative = 1.0
    iterations = 0
    while relative > tolerance:
        # One Krylov window: the restart length, clipped by whatever part of the
        # caller's iteration budget is left. A spent budget ends the solve, and
        # the caller decides from the returned residual whether that is enough.
        window = min(restart, max_iterations - iterations)
        if window <= 0:
            break
        basis = [residual / residual_norm]
        hessenberg = torch.zeros((restart + 1, restart), dtype=_COEFFICIENT_DTYPE)
        target = torch.zeros(restart + 1, dtype=_COEFFICIENT_DTYPE)
        target[0] = residual_norm
        coefficients = torch.zeros((window, 1), dtype=_COEFFICIENT_DTYPE)
        columns = 0
        for index in range(window):
            candidate = apply(basis[index])
            for row in range(index + 1):
                projection = torch.vdot(basis[row], candidate)
                hessenberg[row, index] = projection.to(_COEFFICIENT_DTYPE)
                candidate = candidate - projection * basis[row]
            next_norm = float(torch.linalg.vector_norm(candidate))
            hessenberg[index + 1, index] = next_norm
            columns = index + 1
            iterations += 1
            columns_target = target[: columns + 1].unsqueeze(1)
            approximation = torch.linalg.lstsq(
                hessenberg[: columns + 1, :columns], columns_target
            ).solution
            estimate = float(
                torch.linalg.vector_norm(
                    columns_target - hessenberg[: columns + 1, :columns] @ approximation
                )
            )
            coefficients = approximation
            if estimate <= tolerance * rhs_norm or next_norm == 0.0:
                break
            basis.append(candidate / next_norm)
        for index in range(columns):
            solution = solution + basis[index] * coefficients[index, 0].to(rhs.dtype)
        residual = rhs - apply(solution)
        residual_norm = float(torch.linalg.vector_norm(residual))
        relative = residual_norm / rhs_norm
    return solution, relative
