"""Compare concrete analytical rotation costs with unmodified Qualtran."""

from dataclasses import asdict
from math import sqrt

import pytest

from flagquantum.algorithms.primitives.rotation_preparation import RotationPreparation
from flagquantum.algorithms.primitives.rotation_resource_model import (
    rotation_preparation_cost,
    sparse_rotation_preparation_cost,
)
from flagquantum.algorithms.primitives.sparse_rotation_preparation import (
    SparseRotationPreparation,
)
from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    estimate_component_resources,
)

pytestmark = pytest.mark.unit


def assert_source_cost(cost: ComponentCost, source: object) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    model = estimate_component_resources(cost)
    gates = get_cost_value(source, QECGatesCost())
    assert {
        key: value for key, value in asdict(model.gates).items() if value
    } == gates.asdict()
    assert model.gates.total_t_count() == gates.total_t_count()
    assert model.peak_logical_qubits == get_cost_value(source, QubitCount())


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("controls", [0, 1, 2])
@pytest.mark.parametrize("uncompute", [False, True])
@pytest.mark.parametrize(
    "coefficients,width",
    [
        ((1, 0), 2),
        ((0.5, 0.5j, -0.5, -0.5j), 8),
        (tuple(value / sqrt(15) for value in (1j, 2, -3j, -1)), 4),
    ],
)
def test_dense_model_matches_source(
    coefficients: tuple[complex, ...], width: int, controls: int, uncompute: bool
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.state_preparation.state_preparation_via_rotation import (
        StatePreparationViaRotations,
    )

    source = StatePreparationViaRotations(coefficients, width, controls, uncompute)
    model = rotation_preparation_cost(
        RotationPreparation(coefficients, width), controls, uncompute=uncompute
    )
    assert_source_cost(model, source)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "indices,size,width,phase_width",
    [((0, 1), 2, 1, 2), ((2, 1), 3, 2, 2), ((2, 1), 3, 4, 8), ((3, 1, 4), 5, 3, 2)],
)
def test_sparse_model_matches_source(
    indices: tuple[int, ...], size: int, width: int, phase_width: int
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.state_preparation.sparse_state_preparation_via_rotations import (
        SparseStatePreparationViaRotations,
    )

    coefficients = (1 + 1j, 1 - 1j, -1j)[: len(indices)]
    source = SparseStatePreparationViaRotations(
        indices, coefficients, size, phase_width, width
    )
    model = sparse_rotation_preparation_cost(
        SparseRotationPreparation(indices, coefficients, size, phase_width), width
    )
    assert_source_cost(model, source)
