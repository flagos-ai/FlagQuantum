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


@pytest.mark.parametrize("bits", [0, 1, 2, 18])
def test_raw_rotation_declaration_preserves_signed_source_counts(bits):
    from qualtran.bloqs.basic_gates import Toffoli
    from qualtran.bloqs.chemistry.black_boxes import QROAM
    from qualtran.bloqs.chemistry.df.select_bloq import ProgRotGateArray
    from qualtran.resource_counting import (
        QECGatesCost,
        SympySymbolAllocator,
        get_cost_value,
    )

    from flagquantum.algorithms.primitives.double_factorization_rotation_model import (
        source_double_factorization_programmed_rotation_declaration,
    )

    source = ProgRotGateArray(50, 10, 250, bits)
    declaration = source_double_factorization_programmed_rotation_declaration(
        10, 250, bits
    )
    assert_df_signature(declaration.signature, source)
    assert source.build_call_graph(SympySymbolAllocator()) == {
        Toffoli(): declaration.local_toffolis,
        QROAM(declaration.lookup_rows, declaration.lookup_width): 1,
    }
    if bits < 2:
        with pytest.raises(ValueError, match="negative gate count"):
            source_double_factorization_programmed_rotation_cost(10, 250, bits)
        if bits == 0:
            with pytest.raises(ZeroDivisionError):
                get_cost_value(source, QECGatesCost())
        else:
            assert get_cost_value(source, QECGatesCost()).toffoli == 64
    else:
        model = source_double_factorization_programmed_rotation_cost(10, 250, bits)
        assert model.name == str(source)
        assert_df_leaf(model, source)


def test_nonnegative_one_bit_rotation_is_estimable():
    from qualtran.bloqs.chemistry.df.select_bloq import ProgRotGateArray

    assert_df_leaf(
        source_double_factorization_programmed_rotation_cost(2, 7, 1),
        ProgRotGateArray(1, 2, 7, 1),
    )
