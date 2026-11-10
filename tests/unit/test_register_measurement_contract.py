"""Original destructive measurement instrument before native implementation."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_source_measurement_instrument_retains_spectator_complex_phase() -> None:
    from qualtran.bloqs.basic_gates import MeasureZ
    from qualtran.simulation.tensor import DiscardInd

    copy, discarded = MeasureZ().my_tensors({"q": "input"}, {"c": "output"})
    assert isinstance(discarded, DiscardInd)
    assert copy.inds[:2] == (("input", 0), ("output", 0))
    state = np.array([1 + 2j, -3j, 2 - 1j, -4 + 3j], dtype=np.complex128)
    state /= np.linalg.norm(state)
    state = state.reshape(2, 2)
    branches = [np.einsum("qc,qs->cs", copy.data[:, :, b], state) for b in (0, 1)]
    for outcome, branch in enumerate(branches):
        np.testing.assert_array_equal(branch[outcome], state[outcome])
        np.testing.assert_array_equal(branch[1 - outcome], np.zeros(2))
    assert sum(float(np.vdot(b, b).real) for b in branches) == pytest.approx(1)
    density = sum(np.outer(b.ravel(), b.ravel().conj()) for b in branches)
    assert np.linalg.matrix_rank(density) == 2
