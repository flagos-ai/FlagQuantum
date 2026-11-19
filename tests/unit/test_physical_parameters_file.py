"""Close hardware presets while preserving native physical admission rules."""

from dataclasses import FrozenInstanceError
from hashlib import sha1
from pathlib import Path

import pytest
from qualtran.surface_code.physical_parameters import (
    PhysicalParameters as SourceParameters,
)

from flagquantum.qec.resources.physical_parameters import (
    PhysicalParameters,
    QubitModality,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_frozen_source_hardware() -> None:
    path = (
        Path(__file__).resolve().parents[3]
        / "ftqc-reference-env/qualtran/surface_code/physical_parameters.py"
    )
    raw = path.read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "08d00e221fd0292fc4b0f19d47b790398fe04d36"
    )


@pytest.mark.parametrize("modality", ("superconducting", "ion", "majorana"))
@pytest.mark.parametrize("optimistic", (False, True))
def test_every_constructor_and_literature_preset(
    modality: QubitModality, optimistic: bool
) -> None:
    expected = SourceParameters.make_beverland_et_al(modality, optimistic)
    actual = PhysicalParameters.beverland(modality, optimistic_error_rate=optimistic)
    assert (actual.physical_error, actual.cycle_time_us) == (
        expected.physical_error,
        expected.cycle_time_us,
    )
    assert actual == PhysicalParameters(expected.physical_error, expected.cycle_time_us)
    expected = SourceParameters.make_gidney_fowler(optimistic)
    actual = PhysicalParameters.gidney_fowler(optimistic_error_rate=optimistic)
    assert (actual.physical_error, actual.cycle_time_us) == (
        expected.physical_error,
        expected.cycle_time_us,
    )
    assert PhysicalParameters() == PhysicalParameters(1e-3, 1.0)
    assert hash(actual) == hash(
        PhysicalParameters(actual.physical_error, actual.cycle_time_us)
    )
    with pytest.raises(FrozenInstanceError):
        setattr(actual, "physical_error", 1e-5)


@pytest.mark.parametrize(
    "physical_error,cycle_time",
    (
        (0.0, 1.0),
        (0.01, 1.0),
        (-1e-3, 1.0),
        (float("nan"), 1.0),
        (float("inf"), 1.0),
        (1e-3, 0.0),
        (1e-3, -1.0),
        (1e-3, float("nan")),
        (1e-3, float("inf")),
    ),
)
def test_source_permissive_constructor_and_native_admission_boundary(
    physical_error: float, cycle_time: float
) -> None:
    source = SourceParameters(physical_error, cycle_time)
    assert source is not None
    with pytest.raises(ValueError):
        PhysicalParameters(physical_error, cycle_time)
