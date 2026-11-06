"""Concrete source alias-table preprocessing and register contracts."""

import pytest

from flagquantum.algorithms.primitives.alias_sampling import (
    AliasSamplingTable,
    alias_sampling_table,
    sparse_alias_sampling_table,
    sparse_alias_sampling_table_from_dense,
)

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "weights,precision",
    [
        ([1], 0.5),
        ([1, 1, 3, 2], 7 / 32),
        ([0, 1, 0, 2, 1], 1 / 16),
        ([3, 2, 1], 1 / 32),
        ([0.125, 0.25, 0.5], 0.875),
    ],
)
def test_dense_integer_tables_match_locked_factory(weights, precision):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.state_preparation.state_preparation_alias_sampling import (
        StatePreparationAliasSampling,
    )

    source = StatePreparationAliasSampling.from_probabilities(
        weights, precision=precision
    )
    table = alias_sampling_table(weights, precision=precision)
    assert table == AliasSamplingTable(
        len(weights),
        tuple(map(int, source.alt)),
        tuple(map(int, source.keep)),
        source.mu,
        source.l1_norm_of_coeffs,
    )


@pytest.mark.ftqc_reference
def test_sparse_factories_preserve_mapping_order_and_relative_precision():
    pytest.importorskip("qualtran")
    from qualtran.bloqs.state_preparation.state_preparation_alias_sampling import (
        SparseStatePreparationAliasSampling,
    )

    values = {5: 3, 0: 1, 3: 1}
    precision = 1 / 16
    table = sparse_alias_sampling_table(values, 6, precision=precision)
    source = SparseStatePreparationAliasSampling.from_sparse_dict(
        values, 6, precision=precision
    )
    assert table.indices == tuple(map(int, source.index)) == (5, 0, 3)
    assert table.alternates == tuple(map(int, source.alt))
    assert table.keep == tuple(map(int, source.keep))
    assert table.mu == source.mu
    weights = [1, 1e-6, 0, 2]
    table = sparse_alias_sampling_table_from_dense(weights, precision=precision)
    source = SparseStatePreparationAliasSampling.from_dense_probabilities(
        weights, precision=precision
    )
    assert table.indices == tuple(map(int, source.index)) == (0, 3)
    assert table.alternates == tuple(map(int, source.alt))
    assert table.keep == tuple(map(int, source.keep))
    assert table.normalization == source.l1_norm_of_coeffs


@pytest.mark.parametrize("weights", [(), (0, 0), (-1, 2), (float("nan"), 1)])
def test_rejects_invalid_weights(weights):
    with pytest.raises(ValueError, match="Alias sampling"):
        alias_sampling_table(weights)


def test_sparse_validation():
    with pytest.raises(ValueError, match="in-range"):
        sparse_alias_sampling_table({2: 1}, 2)
    with pytest.raises(ValueError, match="zero threshold"):
        sparse_alias_sampling_table_from_dense((1, 2), nonzero_threshold=-1)
