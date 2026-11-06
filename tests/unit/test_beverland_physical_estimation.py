"""Check the 15-to-1 physical architecture against locked Qualtran."""

from dataclasses import asdict

import pytest

from flagquantum.algorithms.primitives.multiplexers.thc_resource_model import (
    select_thc_cost,
)
from flagquantum.compiler._component_trace.resource_model import (
    GateCost,
    ModeledResources,
)
from flagquantum.qec.resources.beverland_physical import (
    estimate_beverland_physical_resources,
)
from flagquantum.qec.resources.data_block import DataLayout
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from flagquantum.qec.resources.fifteen_to_one import FifteenToOneFactory
from flagquantum.qec.resources.physical_parameters import (
    PhysicalParameters,
    QubitModality,
)

pytestmark = pytest.mark.unit


def test_thc_with_default_hardware() -> None:
    logical = estimate_ftqc_resources(select_thc_cost(2, 4, 8, 1)).logical
    result = estimate_beverland_physical_resources(logical, data_distance=25)
    assert result.cycles == 26550
    assert result.physical_qubits == 46146
    assert result.duration_hours * 3600 == pytest.approx(0.01062)
    assert result.failure_probability > 0


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("layout", ["compact", "intermediate", "fast"])
def test_default_configuration_matches_source(layout: DataLayout) -> None:
    surface = pytest.importorskip("qualtran.surface_code")
    from qualtran.resource_counting import GateCounts

    logical = estimate_ftqc_resources(select_thc_cost(2, 4, 8, 1)).logical
    reference = surface.PhysicalCostModel.make_beverland_et_al(25, layout)
    summary = surface.AlgorithmSummary(
        n_algo_qubits=logical.peak_logical_qubits,
        n_logical_gates=GateCounts(**asdict(logical.gates)),
    )
    result = estimate_beverland_physical_resources(
        logical, data_distance=25, data_layout=layout
    )
    assert result.cycles == reference.n_cycles(summary)
    assert result.duration_hours == reference.duration_hr(summary)
    assert result.physical_qubits == reference.n_phys_qubits(summary)
    assert result.failure_probability == pytest.approx(
        reference.error(summary), abs=1e-12
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    ("modality", "factories", "gates"),
    [
        ("superconducting", 3, GateCost(t=2, cswap=2, and_bloq=5, rotation=3)),
        ("ion", 1, GateCost(t=101, measurement=11)),
        ("majorana", 4, GateCost(clifford=7)),
    ],
)
def test_hardware_and_factory_overrides(
    modality: QubitModality, factories: int, gates: GateCost
) -> None:
    surface = pytest.importorskip("qualtran.surface_code")
    from qualtran.resource_counting import GateCounts
    from qualtran.surface_code.multi_factory import MultiFactory

    source_factory = MultiFactory(surface.FifteenToOne(7, 3, 3), factories)
    reference = surface.PhysicalCostModel(
        surface.PhysicalParameters.make_beverland_et_al(modality, True),
        surface.FastDataBlock(13),
        source_factory,
        surface.QECScheme.make_beverland_et_al(),
    )
    source_gates = GateCounts(**asdict(gates))
    logical = ModeledResources(gates, 6)
    summary = surface.AlgorithmSummary(n_algo_qubits=6, n_logical_gates=source_gates)
    result = estimate_beverland_physical_resources(
        logical,
        data_distance=13,
        data_layout="fast",
        factory=FifteenToOneFactory(7, 3, 3),
        hardware=PhysicalParameters.beverland(modality, optimistic_error_rate=True),
        factory_count=factories,
    )
    assert result.cycles == reference.n_cycles(summary)
    assert result.duration_hours == reference.duration_hr(summary)
    assert result.data_block_physical_qubits == reference.data_block.n_physical_qubits(
        6
    )
    assert result.factory_physical_qubits == source_factory.n_physical_qubits()
    assert result.data_error == reference.data_block.data_error(
        6, result.cycles, reference.logical_error_model
    )
    assert result.factory_error == pytest.approx(
        source_factory.factory_error(source_gates, reference.logical_error_model),
        abs=1e-12,
    )
