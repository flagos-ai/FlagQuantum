"""Compare the locked programmed basis rotation resource declaration."""

import pytest

from flagquantum.algorithms.primitives.double_factorization_rotation_model import (
    source_double_factorization_programmed_rotation_cost,
)
from tests.unit.test_double_factorization_prepare_model import assert_df_leaf

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "spin,eigenvalues,bits", [(10, 250, 8), (16, 190, 6), (4, 8, 2)]
)
@pytest.mark.parametrize("adjoint", [False, True])
def test_programmed_rotation_call_graph(spin, eigenvalues, bits, adjoint):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.df.select_bloq import ProgRotGateArray

    source = ProgRotGateArray(50, spin, eigenvalues, bits)
    assert_df_leaf(
        source_double_factorization_programmed_rotation_cost(
            spin, eigenvalues, bits, adjoint=adjoint
        ),
        source.adjoint() if adjoint else source,
    )
