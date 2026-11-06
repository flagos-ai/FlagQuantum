"""Compare both directions of the locked double-factorization PREPARE models."""

from dataclasses import asdict

import pytest

from flagquantum.algorithms.primitives.double_factorization_prepare_model import (
    source_double_factorization_indexed_data_cost,
    source_double_factorization_inner_prepare_cost,
    source_double_factorization_outer_prepare_cost,
)
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from tests.unit.test_single_factorization_resource_model import _assert_model

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def assert_df_leaf(model, source):
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    _assert_model(model, source)
    for edge, child in zip(model.calls, source.bloq_counts(), strict=True):
        gates = estimate_component_resources(edge.component).gates
        assert {
            name: value for name, value in asdict(gates).items() if value
        } == get_cost_value(child, QECGatesCost()).asdict()


@pytest.mark.parametrize(
    "aux,spin,eigenvalues,rotation,alias",
    [(50, 10, 250, 7, 12), (31, 8, 124, 8, 6), (42, 16, 190, 5, 3)],
)
@pytest.mark.parametrize("adjoint", [False, True])
def test_locked_prepare_leaves(aux, spin, eigenvalues, rotation, alias, adjoint):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.df.prepare import (
        InnerPrepareDoubleFactorization,
        OuterPrepareDoubleFactorization,
        OutputIndexedData,
    )

    for model, source in (
        (
            source_double_factorization_inner_prepare_cost(
                spin, eigenvalues, rotation, alias, adjoint=adjoint
            ),
            InnerPrepareDoubleFactorization(aux, spin, eigenvalues, rotation, alias),
        ),
        (
            source_double_factorization_outer_prepare_cost(aux, alias, adjoint=adjoint),
            OuterPrepareDoubleFactorization(aux, alias, rotation),
        ),
        (
            source_double_factorization_indexed_data_cost(
                aux, spin, eigenvalues, rotation, adjoint=adjoint
            ),
            OutputIndexedData(aux, spin, eigenvalues, rotation),
        ),
    ):
        assert_df_leaf(model, source.adjoint() if adjoint else source)


def test_locked_indexed_data_small_forward_model_boundary():
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.df.prepare import OutputIndexedData
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = OutputIndexedData(1, 4, 3, 8)
    with pytest.raises(AssertionError):
        get_cost_value(source, QECGatesCost())
    with pytest.raises(ValueError, match="rows >= target width"):
        source_double_factorization_indexed_data_cost(1, 4, 3, 8)
    assert_df_leaf(
        source_double_factorization_indexed_data_cost(1, 4, 3, 8, adjoint=True),
        source.adjoint(),
    )
