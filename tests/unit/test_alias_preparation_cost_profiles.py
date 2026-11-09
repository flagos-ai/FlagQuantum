"""Alias and coefficient factories propagate one QEC profile to every child."""

import pytest

from flagquantum.algorithms.primitives.alias_preparation import (
    source_dense_alias_preparation_cost,
    source_sparse_alias_preparation_cost,
)
from flagquantum.algorithms.primitives.alias_sampling import (
    alias_sampling_table,
    sparse_alias_sampling_table,
)
from flagquantum.algorithms.primitives.linear_combination_preparation import (
    LinearCombinationPreparation,
)
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from tests.unit.test_preparation_cost_profiles import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "weights,precision",
    [((1, 3), 1.0), ((1, 1, 1), 0.25), ((1, 2, 3), 0.5), ((1, 2, 3, 4), 0.3)],
)
@pytest.mark.parametrize("legacy", (False, True))
def test_dense_alias_every_child_uses_original_profile(weights, precision, legacy):
    from qualtran.bloqs.state_preparation import StatePreparationAliasSampling

    table = alias_sampling_table(weights, precision=precision)
    source = StatePreparationAliasSampling.from_probabilities(
        weights, precision=precision
    )
    model = source_dense_alias_preparation_cost(table, legacy_shims=legacy)
    assert_source(model, source, legacy)
    assert_source(model, source.adjoint(), legacy)
    assert source_dense_alias_preparation_cost(
        table
    ) == source_dense_alias_preparation_cost(table, legacy_shims=False)


@pytest.mark.parametrize(
    "weights,size,precision",
    [
        ({3: 1, 0: 3}, 4, 0.25),
        ({6: 1, 2: 2, 0: 4}, 7, 0.25),
        ({0: 1, 3: 1, 9: 3, 1: 2}, 10, 0.125),
        ({15: 4, 1: 1}, 16, 0.125),
    ],
)
@pytest.mark.parametrize("legacy", (False, True))
def test_sparse_alias_every_child_uses_original_profile(
    weights, size, precision, legacy
):
    from qualtran.bloqs.state_preparation.state_preparation_alias_sampling import (
        SparseStatePreparationAliasSampling,
    )

    table = sparse_alias_sampling_table(weights, size, precision=precision)
    source = SparseStatePreparationAliasSampling.from_sparse_dict(
        weights, size, precision=precision
    )
    model = source_sparse_alias_preparation_cost(table, legacy_shims=legacy)
    assert_source(model, source, legacy)
    assert_source(model, source.adjoint(), legacy)
    assert source_sparse_alias_preparation_cost(
        table
    ) == source_sparse_alias_preparation_cost(table, legacy_shims=False)


@pytest.mark.parametrize("bits", (1, 3, 8))
@pytest.mark.parametrize("legacy", (False, True))
def test_lcu_factory_matches_original_packed_prepare_without_reference_model_inputs(
    bits, legacy
):
    from tests.unit.test_linear_combination_block_encoding import fixture

    native, source, *_ = fixture(2, bits, True, True, False)
    model = native.coefficient_preparation.resource_model(legacy_shims=legacy)
    assert_source(model, source.prepare, legacy)
    assert_source(model, source.prepare.adjoint(), legacy)


@pytest.mark.parametrize("value", (None, 0, 1, "legacy"))
def test_alias_and_coefficient_factories_require_boolean_profiles(value):
    dense = alias_sampling_table((1, 3), precision=0.25)
    sparse = sparse_alias_sampling_table({0: 1, 3: 3}, 4, precision=0.25)
    coefficient = LinearCombinationPreparation((1.0, -3.0), (1.0, 1.0), 3)
    for factory, obj in (
        (source_dense_alias_preparation_cost, dense),
        (source_sparse_alias_preparation_cost, sparse),
    ):
        with pytest.raises(ValueError):
            factory(obj, legacy_shims=value)
    with pytest.raises(ValueError):
        coefficient.resource_model(legacy_shims=value)


def test_model_only_profile_query_does_not_invoke_a_quantum_emitter(monkeypatch):
    import flagquantum.algorithms.primitives.alias_preparation as alias
    from flagquantum.compiler.registers.comparison import less_equal
    from flagquantum.compiler.registers.preparation import uniform_preparation

    def forbidden(*args, **kwargs):
        raise AssertionError("Cost query invoked an emitter")

    monkeypatch.setattr(alias, "alias_preparation_ir", forbidden)
    monkeypatch.setattr(less_equal, "less_equal_ir", forbidden)
    monkeypatch.setattr(uniform_preparation, "uniform_preparation_ir", forbidden)
    operation = LinearCombinationPreparation((1.0, -3.0), (1.0, 1.0), 8)
    modern = estimate_ftqc_resources(operation.resource_model())
    legacy = estimate_ftqc_resources(operation.resource_model(legacy_shims=True))
    assert modern.emitted is legacy.emitted is None
    assert modern.logical.gates.clifford < legacy.logical.gates.clifford
    assert modern.logical.gates.total_t_count() == legacy.logical.gates.total_t_count()
    assert modern.logical.peak_logical_qubits == legacy.logical.peak_logical_qubits


def test_runnable_native_legacy_model_and_emitted_ir_remain_separate(capsys):
    from examples.algorithms.linear_combination_controls import main

    main(repeated=True, legacy_shims=True)
    output = capsys.readouterr().out
    assert "QEC profile legacy" in output
    assert "Model T equivalents (abstract Fredkin) 28" in output
    assert "Emitted T/Tdag (exact Fredkin) 34" in output
