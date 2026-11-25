"""Classical parameters and exact finite-grid bounds; no quantum input evaluation."""

import math

DISTRIBUTION = (
    0.065634503010069,
    0.434365496989931,
    0.434365496989931,
    0.065634503010069,
)
_WEIGHTS = tuple(math.exp(0.25 * word) for word in range(16))
EXPONENTIAL = tuple(weight / math.fsum(_WEIGHTS) for weight in _WEIGHTS)


def predicate_bounds(shift: float) -> list[int]:
    """Translate coefficient-truncated signed increments to unsigned comparisons.

    The source quantizes each affine coefficient toward zero at precision 2.
    Its raw increment is grid - cut. These bounds preserve the binary calls,
    strict barrier crossings and the sign-bit-flipped integration threshold.
    Only the two independently validated model configurations are admitted.
    """
    if isinstance(shift, bool) or not isinstance(shift, (int, float)):
        raise ValueError("Shift must be a finite real model parameter")
    if not math.isfinite(shift) or shift not in (0.1274075, 0.22):
        raise ValueError("Validated shifts are 0.1274075 and 0.22")
    cut = -math.trunc((shift - 3 * 0.238268134860707) * 4)
    return [cut, 2 * cut, 3 * cut, cut - 1, 2 * cut - 1, 3 * cut - 1, 8 - 3 * cut]
