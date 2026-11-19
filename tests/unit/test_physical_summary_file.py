"""Map the complete source physical summary to the existing native result."""

from dataclasses import FrozenInstanceError
from hashlib import sha1
from pathlib import Path

import pytest
from qualtran.surface_code.physical_cost_summary import PhysicalCostsSummary

from flagquantum.compiler._component_trace.resource_model import (
    GateCost,
    ModeledResources,
)
from flagquantum.qec.resources.physical_cost import (
    PhysicalCostModel,
    PhysicalResourceEstimate,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_frozen_source_summary() -> None:
    path = (
        Path(__file__).resolve().parents[3]
        / "ftqc-reference-env/qualtran/surface_code/physical_cost_summary.py"
    )
    raw = path.read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "fa6806ae0a1b7dd8cc63d19f0cb6635101aa807d"
    )


@pytest.mark.parametrize(
    "qubits,hours,data_error,factory_error",
    ((0, 0.0, 0.0, 0.0), (17, 2.5, 0.01, 0.02), (10000000, 1e-6, 0.9, 0.8)),
)
def test_complete_fields_and_qubit_hours(
    qubits: int,
    hours: float,
    data_error: float,
    factory_error: float,
) -> None:
    actual = PhysicalResourceEstimate(
        123, hours, qubits // 3, qubits - qubits // 3, data_error, factory_error
    )
    expected = PhysicalCostsSummary(data_error + factory_error, qubits, hours)
    assert actual.failure_probability == expected.failure_prob
    assert actual.physical_qubits == expected.footprint
    assert actual.duration_hours == expected.duration_hr
    assert actual.qubit_hours == expected.qubit_hours
    assert actual.cycles == 123
    assert hash(actual) == hash(
        PhysicalResourceEstimate(
            123, hours, qubits // 3, qubits - qubits // 3, data_error, factory_error
        )
    )
    with pytest.raises(FrozenInstanceError):
        setattr(actual, "duration_hours", 2)


def test_existing_search_cost_consumer_uses_native_summary() -> None:
    logical = ModeledResources(GateCost(toffoli=10**8), 100)
    candidates = [
        PhysicalCostModel.gidney_fowler(d).estimate(logical) for d in (17, 25, 33)
    ]
    original = [
        PhysicalCostsSummary(c.failure_probability, c.physical_qubits, c.duration_hours)
        for c in candidates
    ]
    assert [c.qubit_hours for c in candidates] == [c.qubit_hours for c in original]
    assert min(range(3), key=lambda i: candidates[i].qubit_hours) == min(
        range(3), key=lambda i: original[i].qubit_hours
    )
