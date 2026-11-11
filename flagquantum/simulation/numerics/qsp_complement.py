# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 qsp_complementary_polynomial.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/qsp/generalized_qsp.py

"""Root-based complementary polynomial in the original selection order."""

from collections.abc import Iterable, Sequence
from typing import cast

import numpy as np
from numpy.polynomial import Polynomial
from numpy.typing import NDArray


def qsp_complementary_polynomial(
    coefficients: Sequence[complex],
    *,
    verify: bool = False,
    verify_precision: float = 1e-7,
) -> NDArray[np.complex128] | NDArray[np.float64]:
    """Compute the source root-selected Q without changing P or repairing roots.

    For a valid bounded P, the intended identity is abs(P)**2+abs(Q)**2=1 on
    the circle. This numerical routine assumes that input condition; verify
    checks root properties, not a rigorous polynomial bound. Preserve NumPy
    root order, allclose defaults, unit-root pairing, reciprocal verification,
    and interpolation from smaller roots. Empty/degenerate and ill-conditioned
    cases retain source failures or numerical limitations. No gates, work wires,
    or physical resource estimate are produced.
    """
    degree = len(coefficients) - 1
    residual = Polynomial.basis(degree) - Polynomial(coefficients) * Polynomial(
        np.conj(coefficients[::-1])
    )
    units: list[complex] = []
    larger: list[complex] = []
    smaller: list[complex] = []
    roots = cast(Iterable[complex], residual.roots())
    for root in roots:
        if verify:
            assert not (np.abs(root) <= verify_precision), "zero root!"
        if np.allclose(np.abs(root), 1):
            units.append(root)
        elif np.abs(root) > 1:
            larger.append(root)
        else:
            smaller.append(root)

    paired: list[complex] = []
    unpaired: list[complex] = []
    for root in units:
        matched = None
        for candidate in unpaired:
            if np.allclose(root, candidate, rtol=verify_precision):
                matched = candidate
                break
        if matched is not None:
            paired.append(root)
            unpaired.remove(matched)
        else:
            unpaired.append(root)

    if verify:
        assert len(unpaired) == 0
        reciprocal = 1 / np.array(larger).conj()
        assert len(smaller) == len(reciprocal)
        remaining = list(smaller)
        unmatched: list[complex] = []
        for root in reciprocal:
            for candidate in remaining:
                if np.allclose(
                    root, candidate, rtol=verify_precision, atol=verify_precision
                ):
                    remaining.remove(candidate)
                    break
            else:
                unmatched.append(root)
        assert len(unmatched) == 0

    leading = residual.coef[-1]
    scale = np.sqrt(np.abs(leading * np.prod(larger)))
    result: NDArray[np.complex128] | NDArray[np.float64] = (
        scale * Polynomial.fromroots(paired + smaller)
    ).coef
    return result
