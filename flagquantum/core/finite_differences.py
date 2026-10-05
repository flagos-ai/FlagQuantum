"""Central differences, and the one step policy every difference route uses.

A difference quotient is the only derivative FlagQuantum can produce without
first asking the operator rules what a program differentiates, so two very
different callers need it: the user-facing ``fq.gradient`` fallback, and the
CPU conformance oracle that checks an analytic route against an independent
method.  Both must perturb by the same amount and divide by the same
denominator, so both are served here rather than each keeping its own copy.

The quotient has two opposing errors.  Truncation falls as the step grows,
roundoff grows as the step shrinks, and for a central difference the step that
balances them is the cube root of the machine epsilon of the precision the
*program computes in* -- not of the precision the caller's parameter tensor
happens to carry.  :func:`default_difference_step` derives it from a delivered
sample for exactly that reason.

The method is deliberately imprecise.  :func:`central_difference_gradient`
costs two evaluations per parameter, its error is bounded only by the step,
and it is meant to be an *independent* check on an analytic route (a
conformance oracle evaluates it in a wider precision than the route under
test) or a last-resort fallback.  It is not a production derivative, and no
FlagQuantum route presents it as an exact one.

Nothing here owns a user-visible error taxonomy.  :func:`central_difference_gradient`
guards the invariant that the step it divides by is usable, and raises
``ValueError``; a public entry point that has to raise its own documented error
type validates its argument before handing it over.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

# A scalar-valued function of a parameter tensor shaped like the base.
DifferenceEvaluator = Callable[[torch.Tensor], torch.Tensor]


def default_difference_step(sample: torch.Tensor) -> float:
    """Derive a difference displacement from the precision the loss carries.

    A difference quotient loses significant digits to roundoff as the step
    shrinks, and the roundoff floor is set by the dtype the program computes in,
    not by the dtype the caller's parameters happen to have. The loss's dtype is
    the measured one, and the cube root of its machine epsilon is the step that
    balances truncation against roundoff for a central difference.
    """

    return math.pow(float(torch.finfo(sample.dtype).eps), 1.0 / 3.0)


def central_difference_gradient(
    evaluate: DifferenceEvaluator,
    parameters: torch.Tensor,
    *,
    step: float,
) -> torch.Tensor:
    """Differentiate ``evaluate`` at ``parameters`` by a central difference.

    Every element of ``parameters`` is perturbed on its own, so the result costs
    ``2 * parameters.numel()`` evaluations and is exact only up to the step.
    ``evaluate`` is handed a tensor shaped like ``parameters`` with exactly one
    element displaced, which is what lets a caller project a flat parameter
    vector back onto whatever structure its program actually reads.

    The result is a flat stack, one entry per element of ``parameters`` in its
    own logical order, at the dtype the difference arithmetic produced. A caller
    that owes its own caller the input's shape, dtype, and device restores them
    itself.
    """

    if not math.isfinite(step) or step <= 0.0:
        raise ValueError(
            "a central difference step must be a positive finite displacement"
        )
    base = parameters.detach()
    flat = base.reshape(-1)
    terms = []
    for index in range(flat.numel()):
        plus = flat.clone()
        minus = flat.clone()
        plus[index] += step
        minus[index] -= step
        terms.append(
            (evaluate(plus.reshape_as(base)) - evaluate(minus.reshape_as(base)))
            / (2.0 * step)
        )
    return torch.stack(terms)


__all__ = [
    "DifferenceEvaluator",
    "central_difference_gradient",
    "default_difference_step",
]
