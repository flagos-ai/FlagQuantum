"""Compare native analytical estimates against the pinned surface-code model."""

from dataclasses import asdict

import pytest

from flagquantum.algorithms.primitives.multiplexers.thc_resource_model import (
    select_thc_cost,
)
from flagquantum.compiler._component_trace.resource_model import (
    GateCost,
    ModeledResources,
    estimate_component_resources,
)
from flagquantum.qec.resources.beverland import estimate_beverland_resources

pytestmark = pytest.mark.unit


def test_estimate_thc_from_dimensions_alone() -> None:
    logical = estimate_component_resources(select_thc_cost(2, 4, 8, 1))
    result = estimate_beverland_resources(
        logical, error_budget=1e-3, physical_error=1e-4
    )
    assert result.minimum_time_steps == 354
    assert result.t_states == 472
    assert result.discrete_logical_gates == logical.gates
    assert result.data_block_tiles == 63
    assert result.distance_budget_time_steps == result.minimum_time_steps
    assert result.code_distance == 7
    assert result.data_block_physical_qubits == 6174


def test_zero_steps_and_invalid_assumptions() -> None:
    logical = ModeledResources(GateCost(clifford=10), 2)
    result = estimate_beverland_resources(
        logical, error_budget=1e-3, physical_error=1e-4
    )
    assert result.minimum_time_steps == result.t_states == 0
    assert result.data_block_tiles == 9
    assert result.code_distance is None
    assert result.data_block_physical_qubits is None
    with pytest.raises(ValueError, match="Physical error"):
        estimate_beverland_resources(logical, error_budget=1e-3, physical_error=0.01)
    with pytest.raises(ValueError, match="Error budget"):
        estimate_beverland_resources(logical, error_budget=0, physical_error=1e-4)
    with pytest.raises(ValueError, match="Time steps"):
        estimate_beverland_resources(
            logical, error_budget=1e-3, physical_error=1e-4, time_steps=float("inf")
        )
    with pytest.raises(ValueError, match="Rotation layers"):
        estimate_beverland_resources(
            logical, error_budget=1e-3, physical_error=1e-4, rotation_layers=-1
        )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    ("logical", "layers", "steps", "budget"),
    [
        (
            ModeledResources(GateCost(rotation=30100, measurement=1400000), 100),
            501,
            150000,
            1e-3,
        ),
        (
            ModeledResources(
                GateCost(
                    t=55300000,
                    rotation=206000000,
                    toffoli=135000000000,
                    measurement=1370000000,
                ),
                1318,
            ),
            205000000,
            410000000000,
            1e-2,
        ),
        (
            ModeledResources(
                GateCost(t=12, rotation=12, toffoli=3730000000, measurement=1080000000),
                12581,
            ),
            12,
            12300000000,
            1 / 3,
        ),
        (
            ModeledResources(GateCost(rotation=7, and_bloq=2, cswap=5, t=2), 10),
            None,
            None,
            1e-5,
        ),
        (ModeledResources(GateCost(rotation=7, measurement=1), 10), 0, None, 1e-5),
    ],
)
def test_estimate_matches_source(
    logical: ModeledResources,
    layers: int | None,
    steps: float | None,
    budget: float,
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.resource_counting import GateCounts
    from qualtran.surface_code import (
        AlgorithmSummary,
        FastDataBlock,
        QECScheme,
        beverland_et_al_model,
    )
    from qualtran.surface_code.rotation_cost_model import BeverlandEtAlRotationCost

    result = estimate_beverland_resources(
        logical,
        error_budget=budget,
        physical_error=1e-4,
        rotation_layers=layers,
        time_steps=steps,
    )
    source = AlgorithmSummary(
        n_algo_qubits=logical.peak_logical_qubits,
        n_logical_gates=GateCounts(**asdict(logical.gates)),
        n_rotation_layers=layers,
    )
    assert result.minimum_time_steps == beverland_et_al_model.minimum_time_steps(
        error_budget=budget, alg=source, rotation_model=BeverlandEtAlRotationCost
    )
    assert result.t_states == beverland_et_al_model.t_states(
        error_budget=budget, alg=source, rotation_model=BeverlandEtAlRotationCost
    )
    discrete = beverland_et_al_model.n_discrete_logical_gates(
        eps_syn=budget / 3, alg=source, rotation_model=BeverlandEtAlRotationCost
    )
    assert {
        key: value
        for key, value in asdict(result.discrete_logical_gates).items()
        if value
    } == discrete.asdict()
    scheme = QECScheme.make_beverland_et_al()
    distance = beverland_et_al_model.code_distance(
        error_budget=budget,
        time_steps=result.minimum_time_steps if steps is None else steps,
        alg=source,
        qec_scheme=scheme,
        physical_error=1e-4,
    )
    block = FastDataBlock(data_d=distance)
    assert result.code_distance == distance
    assert result.data_block_tiles == block.n_tiles(source.n_algo_qubits)
    assert result.data_block_physical_qubits == block.n_physical_qubits(
        source.n_algo_qubits
    )


@pytest.mark.ftqc_reference
def test_thc_matches_complete_source_model() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.thc.select_bloq import SelectTHC
    from qualtran.surface_code import AlgorithmSummary, QECScheme, beverland_et_al_model
    from qualtran.surface_code.rotation_cost_model import BeverlandEtAlRotationCost

    logical = estimate_component_resources(select_thc_cost(2, 4, 8, 1))
    result = estimate_beverland_resources(
        logical, error_budget=1e-3, physical_error=1e-4
    )
    source = AlgorithmSummary.from_bloq(SelectTHC(2, 4, 8, 1))
    assert result.minimum_time_steps == beverland_et_al_model.minimum_time_steps(
        error_budget=1e-3, alg=source, rotation_model=BeverlandEtAlRotationCost
    )
    assert result.t_states == beverland_et_al_model.t_states(
        error_budget=1e-3, alg=source, rotation_model=BeverlandEtAlRotationCost
    )
    assert result.code_distance == beverland_et_al_model.code_distance(
        error_budget=1e-3,
        time_steps=result.minimum_time_steps,
        alg=source,
        qec_scheme=QECScheme.make_beverland_et_al(),
        physical_error=1e-4,
    )
