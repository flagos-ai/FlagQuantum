"""Compare the catalyzed factory scenario with the pinned physical cost model."""

from dataclasses import asdict

import pytest

from flagquantum.algorithms.primitives.multiplexers.thc_resource_model import (
    select_thc_cost,
)
from flagquantum.compiler._component_trace.resource_model import (
    GateCost,
    ModeledResources,
)
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from flagquantum.qec.resources.gidney_fowler import estimate_gidney_fowler_resources

pytestmark = pytest.mark.unit


def test_thc_estimate_requires_only_the_declared_model() -> None:
    logical = estimate_ftqc_resources(select_thc_cost(2, 4, 8, 1)).logical
    result = estimate_gidney_fowler_resources(
        logical, data_distance=7, physical_error=1e-4
    )

    assert result.cycles == 20119
    assert result.data_block_physical_qubits == 3528
    assert result.factory_physical_qubits == 147904
    assert result.physical_qubits == 151432
    assert result.duration_hours * 3600 == pytest.approx(0.020119)
    assert result.failure_probability > 0


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    ("logical", "data_d", "physical_error", "cycle_time", "l1", "l2"),
    [
        (ModeledResources(GateCost(clifford=10), 2), 3, 1e-3, 1.0, 15, 31),
        (ModeledResources(GateCost(t=1, toffoli=2), 3), 9, 1e-3, 1.0, 15, 31),
        (
            ModeledResources(GateCost(rotation=3, t=2, and_bloq=7, cswap=3), 15),
            11,
            1e-4,
            0.75,
            9,
            13,
        ),
        (
            estimate_ftqc_resources(select_thc_cost(2, 4, 8, 1)).logical,
            7,
            1e-4,
            1.0,
            15,
            31,
        ),
    ],
)
def test_matches_source(
    logical: ModeledResources,
    data_d: int,
    physical_error: float,
    cycle_time: float,
    l1: int,
    l2: int,
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.resource_counting import GateCounts
    from qualtran.surface_code import (
        AlgorithmSummary,
        CCZ2TFactory,
        PhysicalCostModel,
        PhysicalParameters,
        QECScheme,
        SimpleDataBlock,
    )
    from qualtran.surface_code.physical_cost_summary import PhysicalCostsSummary

    gates = GateCounts(**asdict(logical.gates))
    summary = AlgorithmSummary(
        n_algo_qubits=logical.peak_logical_qubits, n_logical_gates=gates
    )
    source = PhysicalCostModel(
        PhysicalParameters(physical_error=physical_error, cycle_time_us=cycle_time),
        SimpleDataBlock(data_d=data_d),
        CCZ2TFactory(l1, l2),
        QECScheme.make_gidney_fowler(),
    )
    result = estimate_gidney_fowler_resources(
        logical,
        data_distance=data_d,
        physical_error=physical_error,
        cycle_time_us=cycle_time,
        distillation_l1_distance=l1,
        distillation_l2_distance=l2,
    )
    source_summary = PhysicalCostsSummary(
        source.error(summary),
        source.n_phys_qubits(summary),
        source.duration_hr(summary),
    )

    assert result.cycles == source.n_cycles(summary)
    assert result.duration_hours == source_summary.duration_hr
    assert result.physical_qubits == source_summary.footprint
    assert result.qubit_hours == source_summary.qubit_hours
    assert result.failure_probability == source_summary.failure_prob
    assert result.factory_physical_qubits == source.factory.n_physical_qubits()
    assert result.data_block_physical_qubits == source.data_block.n_physical_qubits(
        logical.peak_logical_qubits
    )
    assert result.factory_error == source.factory.factory_error(
        gates, source.logical_error_model
    )
    assert result.data_error == source.data_block.data_error(
        logical.peak_logical_qubits, result.cycles, source.logical_error_model
    )


def test_rejects_invalid_scenarios() -> None:
    logical = ModeledResources(GateCost(), 1)
    with pytest.raises(ValueError, match="code distances"):
        estimate_gidney_fowler_resources(logical, data_distance=0)
    with pytest.raises(ValueError, match="Physical error"):
        estimate_gidney_fowler_resources(logical, data_distance=3, physical_error=0.01)
    with pytest.raises(ValueError, match="Cycle time"):
        estimate_gidney_fowler_resources(logical, data_distance=3, cycle_time_us=0)
    with pytest.raises(ValueError, match="nonnegative"):
        estimate_gidney_fowler_resources(
            ModeledResources(GateCost(t=-1), 1), data_distance=3
        )
