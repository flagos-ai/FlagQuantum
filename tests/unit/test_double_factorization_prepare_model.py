"""Compare both directions of the locked double-factorization PREPARE models."""

from dataclasses import asdict

import pytest

from flagquantum.algorithms.primitives.double_factorization_prepare_model import (
    source_double_factorization_indexed_data_cost,
    source_double_factorization_indexed_data_signature,
    source_double_factorization_inner_prepare_cost,
    source_double_factorization_inner_prepare_signature,
    source_double_factorization_outer_prepare_cost,
    source_double_factorization_outer_prepare_signature,
)
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from tests.unit.test_single_factorization_resource_model import _assert_model

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def assert_df_signature(signature, source):
    from qualtran import QBit

    assert [
        (port.name, port.bitsize, port.shape, port.side.value, port.dtype.encoding)
        for port in signature
    ] == [
        (
            port.name,
            port.bitsize,
            port.shape,
            port.side.value,
            "bit" if isinstance(port.dtype, QBit) else "opaque",
        )
        for port in source.signature
    ]


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
    [
        (50, 10, 250, 7, 12),
        (50, 10, 250, 18, 12),
        (31, 8, 124, 8, 6),
        (42, 16, 190, 5, 3),
        (360, 108, 13031, 7, 10),
    ],
)
@pytest.mark.parametrize("adjoint", [False, True])
def test_locked_prepare_leaves(aux, spin, eigenvalues, rotation, alias, adjoint):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.df.prepare import (
        InnerPrepareDoubleFactorization,
        OuterPrepareDoubleFactorization,
        OutputIndexedData,
    )

    for model, signature, source in (
        (
            source_double_factorization_inner_prepare_cost(
                spin, eigenvalues, rotation, alias, adjoint=adjoint
            ),
            source_double_factorization_inner_prepare_signature(
                spin, eigenvalues, rotation
            ),
            InnerPrepareDoubleFactorization(aux, spin, eigenvalues, rotation, alias),
        ),
        (
            source_double_factorization_outer_prepare_cost(aux, alias, adjoint=adjoint),
            source_double_factorization_outer_prepare_signature(aux),
            OuterPrepareDoubleFactorization(aux, alias, rotation),
        ),
        (
            source_double_factorization_indexed_data_cost(
                aux, spin, eigenvalues, rotation, adjoint=adjoint
            ),
            source_double_factorization_indexed_data_signature(
                aux, spin, eigenvalues, rotation
            ),
            OutputIndexedData(aux, spin, eigenvalues, rotation),
        ),
    ):
        directional = source.adjoint() if adjoint else source
        assert_df_signature(signature, directional)
        assert_df_leaf(model, directional)


def test_locked_prepare_omits_empty_ports():
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.df.prepare import (
        InnerPrepareDoubleFactorization,
        OuterPrepareDoubleFactorization,
        OutputIndexedData,
    )

    assert_df_signature(
        source_double_factorization_inner_prepare_signature(2, 0, 8),
        InnerPrepareDoubleFactorization(0, 2, 0, 8, 1),
    )
    assert_df_signature(
        source_double_factorization_outer_prepare_signature(0),
        OuterPrepareDoubleFactorization(0, 1),
    )
    assert_df_signature(
        source_double_factorization_indexed_data_signature(0, 2, 0),
        OutputIndexedData(0, 2, 0),
    )


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


def test_original_prepare_examples_and_display_names():
    from qualtran.bloqs.chemistry.df.prepare import (
        _indexed_data,
        _prep_inner,
        _prep_outer,
    )

    from examples.algorithms.double_factorization_resources import prepare_examples

    examples = (_prep_inner(), _prep_outer(), _indexed_data())
    assert tuple(example.num_bits_rot_aa for example in examples) == (7, 18, 18)
    for (label, signature, model), source in zip(
        prepare_examples(), examples, strict=True
    ):
        assert label == str(source)
        assert_df_signature(signature, source)
        assert_df_leaf(model, source)


def test_prepare_resource_declarations_preserve_valid_zero_precisions():
    from qualtran.bloqs.chemistry.df.prepare import (
        InnerPrepareDoubleFactorization,
        OuterPrepareDoubleFactorization,
        OutputIndexedData,
    )

    inner = InnerPrepareDoubleFactorization(50, 10, 250, 0, 0)
    assert_df_signature(
        source_double_factorization_inner_prepare_signature(10, 250, 0), inner
    )
    assert_df_leaf(source_double_factorization_inner_prepare_cost(10, 250, 0, 0), inner)
    outer = OuterPrepareDoubleFactorization(50, 0, 18)
    assert_df_leaf(source_double_factorization_outer_prepare_cost(50, 0), outer)
    indexed = OutputIndexedData(50, 10, 250, 0)
    assert_df_signature(
        source_double_factorization_indexed_data_signature(50, 10, 250, 0), indexed
    )
    assert_df_leaf(
        source_double_factorization_indexed_data_cost(50, 10, 250, 0), indexed
    )
