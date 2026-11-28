# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# From locked Qualtran v0.7.0 HamiltonianSimulationByGQSP.approx_cos.

"""Own finite and shape-only Hamiltonian QSP coefficient derivation."""

from __future__ import annotations

from typing import cast

from ...core.value_math import SymbolicInt, SymbolicReal
from ...core.value_parameters import is_symbolic_value
from ..value_dimensions import SymbolicArrayShape
from .jacobi_anger_coefficients import approx_exp_cos_by_jacobi_anger
from .jacobi_anger_degree import degree_jacobi_anger_approximation
from .qsp_grid import scale_down_to_qsp_polynomial


def hamiltonian_qsp_coefficients(
    alpha: SymbolicReal,
    time: SymbolicReal,
    precision: SymbolicReal,
    *,
    degree: SymbolicInt | None = None,
) -> tuple[complex, ...] | SymbolicArrayShape:
    """Keep source sign, precision split, FFT normalization and Issue860 margin.

    Symbolic parameters retain only the shape; no numerical values are invented.
    A supplied degree is an existing caller-owned derivation, not an accuracy
    certificate. Numerical/root/grid failures and the original finite-grid
    limitation propagate unchanged.
    """
    count = (
        degree_jacobi_anger_approximation(time * alpha, precision=precision / 2)
        if degree is None
        else degree
    )
    if is_symbolic_value(time, alpha, precision):
        return SymbolicArrayShape((2 * count + 1,))
    assert isinstance(count, int)
    raw = approx_exp_cos_by_jacobi_anger(cast(float, -time * alpha), degree=count)
    normalized = scale_down_to_qsp_polynomial(list(raw))
    return tuple(complex(value) for value in normalized * (1 - 2 * precision))
