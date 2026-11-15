"""Compare the locked programmed basis rotation resource declaration."""

import pytest

from flagquantum.algorithms.primitives.double_factorization_rotation_model import (
    source_double_factorization_programmed_rotation_cost,
    source_double_factorization_programmed_rotation_signature,
)
from tests.unit.test_double_factorization_prepare_model import (
    assert_df_leaf,
    assert_df_signature,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "aux,spin,eigenvalues,bits",
    [(50, 10, 250, 8), (50, 16, 190, 6), (50, 4, 8, 2), (360, 108, 13031, 10)],
)
@pytest.mark.parametrize("adjoint", [False, True])
def test_programmed_rotation_call_graph(aux, spin, eigenvalues, bits, adjoint):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.df.select_bloq import ProgRotGateArray

    source = ProgRotGateArray(aux, spin, eigenvalues, bits)
    directional = source.adjoint() if adjoint else source
    assert_df_signature(
        source_double_factorization_programmed_rotation_signature(
            spin, eigenvalues, bits
        ),
        directional,
    )
    assert_df_leaf(
        source_double_factorization_programmed_rotation_cost(
            spin, eigenvalues, bits, adjoint=adjoint
        ),
        directional,
    )


def test_programmed_rotation_keeps_explicit_empty_opaque_words():
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.df.select_bloq import ProgRotGateArray

    assert_df_signature(
        source_double_factorization_programmed_rotation_signature(2, 0, 2),
        ProgRotGateArray(0, 2, 0, 2),
    )
