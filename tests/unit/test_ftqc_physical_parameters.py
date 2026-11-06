"""Check the physical assumptions used by the modeled architectures."""

import pytest

from flagquantum.qec.resources.physical_parameters import (
    PhysicalParameters,
    QubitModality,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("modality", "cycle_time", "realistic", "optimistic"),
    [
        ("superconducting", 0.4, 1e-3, 1e-4),
        ("ion", 600.0, 1e-3, 1e-4),
        ("majorana", 0.6, 1e-4, 1e-6),
    ],
)
def test_hardware_scenarios(
    modality: QubitModality, cycle_time: float, realistic: float, optimistic: float
) -> None:
    assert PhysicalParameters.beverland(modality) == PhysicalParameters(
        realistic, cycle_time
    )
    assert PhysicalParameters.beverland(
        modality, optimistic_error_rate=True
    ) == PhysicalParameters(optimistic, cycle_time)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("modality", ["superconducting", "ion", "majorana"])
@pytest.mark.parametrize("optimistic", [False, True])
def test_hardware_matches_source(modality: QubitModality, optimistic: bool) -> None:
    source = pytest.importorskip("qualtran.surface_code").PhysicalParameters
    expected = source.make_beverland_et_al(modality, optimistic)
    assert PhysicalParameters.beverland(
        modality, optimistic_error_rate=optimistic
    ) == PhysicalParameters(expected.physical_error, expected.cycle_time_us)
    expected = source.make_gidney_fowler(optimistic)
    assert PhysicalParameters.gidney_fowler(
        optimistic_error_rate=optimistic
    ) == PhysicalParameters(expected.physical_error, expected.cycle_time_us)


def test_rejects_invalid_hardware() -> None:
    with pytest.raises(ValueError, match="Physical error"):
        PhysicalParameters(0.01)
    with pytest.raises(ValueError, match="Cycle time"):
        PhysicalParameters(cycle_time_us=float("inf"))
