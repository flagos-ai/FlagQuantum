"""Concrete comparison/uniform QEC profiles use shared predicate classification."""

from dataclasses import asdict

import pytest

from flagquantum.compiler.registers.comparison.less_equal import (
    source_equal_width_less_equal_cost,
)
from flagquantum.compiler.registers.preparation.uniform_preparation import (
    source_uniform_preparation_cost,
)
from flagquantum.qec.resources.estimation import estimate_ftqc_resources

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def assert_source(model, source, legacy):
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    actual = estimate_ftqc_resources(model)
    expected = get_cost_value(source, QECGatesCost(legacy_shims=legacy))
    assert asdict(actual.logical.gates) == {
        key: getattr(expected, key) for key in asdict(actual.logical.gates)
    }
    assert actual.logical.peak_logical_qubits == get_cost_value(source, QubitCount())
    assert actual.emitted is None


@pytest.mark.parametrize("width", [1, 2, 3, 4, 5, 8, 16])
@pytest.mark.parametrize("legacy", [False, True])
def test_equal_width_comparator_original_categories_and_width(width, legacy):
    from qualtran.bloqs.arithmetic.comparison import LessThanEqual

    model = source_equal_width_less_equal_cost(width, legacy_shims=legacy)
    source = LessThanEqual(width, width)
    assert_source(model, source, legacy)
    assert_source(model, source.adjoint(), legacy)
    assert source_equal_width_less_equal_cost(
        width
    ) == source_equal_width_less_equal_cost(width, legacy_shims=False)


@pytest.mark.parametrize("count", [2, 3, 4, 5, 6, 7, 8, 9, 12, 17, 31, 64])
@pytest.mark.parametrize("legacy", [False, True])
def test_uniform_preparation_original_categories_and_width(count, legacy):
    from qualtran.bloqs.state_preparation.prepare_uniform_superposition import (
        PrepareUniformSuperposition,
    )

    model = source_uniform_preparation_cost(count, legacy_shims=legacy)
    source = PrepareUniformSuperposition(count)
    assert_source(model, source, legacy)
    assert_source(model, source.adjoint(), legacy)
    assert source_uniform_preparation_cost(count) == source_uniform_preparation_cost(
        count, legacy_shims=False
    )


@pytest.mark.parametrize("value", [None, 0, 1, "legacy"])
def test_profile_switch_requires_a_literal_boolean(value):
    with pytest.raises(ValueError):
        source_equal_width_less_equal_cost(3, legacy_shims=value)
    with pytest.raises(ValueError):
        source_uniform_preparation_cost(4, legacy_shims=value)
