"""Compare the complete frozen scheme and replace real estimation consumers."""

from hashlib import sha1
from pathlib import Path

import pytest
from qualtran.surface_code import qec_scheme as source

from flagquantum.compiler._component_trace.resource_model import (
    GateCost,
    ModeledResources,
)
from flagquantum.qec.resources.beverland import estimate_beverland_resources
from flagquantum.qec.resources.beverland_physical import (
    estimate_beverland_physical_resources,
)
from flagquantum.qec.resources.error_model import LogicalErrorModel, SurfaceCodeScheme
from flagquantum.qec.resources.fifteen_to_one import FifteenToOneFactory
from flagquantum.qec.resources.gidney_fowler import estimate_gidney_fowler_resources
from flagquantum.qec.resources.physical_cost import PhysicalCostModel

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_complete_source_and_test_blobs_are_frozen() -> None:
    assert source.__file__ is not None
    for name, blob in (
        ("qec_scheme.py", "00e2bd5acfc55309cb1b340edec25ceeb4183774"),
        ("qec_scheme_test.py", "43b935b1dad82afa477556b536425e3771d1c8b7"),
    ):
        raw = Path(source.__file__).with_name(name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


@pytest.mark.parametrize("scaler,threshold", ((0.1, 0.01), (0.03, 0.01), (0.2, 0.02)))
def test_all_formulas_bound_model_and_budget_inversion(
    scaler: float, threshold: float
) -> None:
    actual = SurfaceCodeScheme(scaler, threshold)
    original = source.QECScheme(scaler, threshold)
    for physical in (0.0, 1e-6, 1e-4, 1e-3, threshold, 2 * threshold):
        model = LogicalErrorModel(physical, actual)
        expected_model = source.LogicalErrorModel(
            physical_error=physical, qec_scheme=original
        )
        for distance in range(32):
            assert actual.logical_error_rate(
                distance, physical
            ) == original.logical_error_rate(distance, physical)
            assert model(code_distance=distance) == expected_model(
                code_distance=distance
            )
            assert actual.physical_qubits(distance) == original.physical_qubits(
                distance
            )
    for physical in (1e-6, 1e-4, 1e-3):
        for exponent in range(1, 40):
            budget = 10.0 ** (-exponent)
            distance = actual.code_distance_from_budget(physical, budget)
            assert distance == original.code_distance_from_budget(physical, budget)
            assert distance >= 3 and distance % 2 == 1
            assert actual.logical_error_rate(
                distance, physical
            ) == original.logical_error_rate(distance, physical)


def test_source_logarithm_rounding_can_exceed_the_exact_budget() -> None:
    actual = SurfaceCodeScheme.gidney_fowler()
    original = source.QECScheme.make_gidney_fowler()
    physical, budget = 1e-4, 1e-7
    distance = actual.code_distance_from_budget(physical, budget)
    assert distance == original.code_distance_from_budget(physical, budget) == 5
    assert (
        actual.logical_error_rate(distance, physical)
        == original.logical_error_rate(distance, physical)
        == 1.0000000000000002e-7
    )
    assert actual.logical_error_rate(distance, physical) > budget


def test_both_factories_subclasses_and_frozen_fields() -> None:
    for actual, original in (
        (SurfaceCodeScheme.gidney_fowler(), source.QECScheme.make_gidney_fowler()),
        (SurfaceCodeScheme.beverland(), source.QECScheme.make_beverland_et_al()),
    ):
        assert actual.error_rate_scaler == original.error_rate_scaler
        assert actual.error_rate_threshold == original.error_rate_threshold
        with pytest.raises(AttributeError):
            setattr(actual, "error_rate_scaler", 0.7)

    class CustomScheme(SurfaceCodeScheme):
        pass

    assert isinstance(CustomScheme.gidney_fowler(), CustomScheme)
    assert isinstance(CustomScheme.beverland(), CustomScheme)


@pytest.mark.parametrize(
    "physical,budget",
    (
        (0.0, 1e-3),
        (0.01, 1e-3),
        (-1e-3, 1e-3),
        (1e-3, 0.0),
        (1e-3, -1.0),
        (1e-3, float("inf")),
    ),
)
def test_arithmetic_failure_categories_match_source(
    physical: float, budget: float
) -> None:
    original = source.QECScheme.make_gidney_fowler()
    with pytest.raises((ValueError, ZeroDivisionError, OverflowError)) as failure:
        original.code_distance_from_budget(physical, budget)
    with pytest.raises(type(failure.value)):
        SurfaceCodeScheme.gidney_fowler().code_distance_from_budget(physical, budget)


def test_real_consumers_use_replaceable_shared_formulas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logical = ModeledResources(GateCost(t=11, toffoli=7, rotation=3, measurement=5), 9)
    factory = FifteenToOneFactory(9, 3, 3)

    def scenarios() -> tuple[object, ...]:
        return (
            estimate_gidney_fowler_resources(logical, data_distance=7),
            estimate_beverland_resources(
                logical, error_budget=0.01, physical_error=1e-3
            ),
            estimate_beverland_physical_resources(logical, data_distance=7),
            factory.estimate(logical.gates, physical_error=1e-3, error_rate_scaler=0.1),
        )

    expected = scenarios()
    rates: set[float] = set()
    inversions: list[float] = []

    def reference_rate(
        self: SurfaceCodeScheme, code_distance: int, physical_error: float
    ) -> float:
        rates.add(self.error_rate_scaler)
        value = source.QECScheme(
            self.error_rate_scaler, self.error_rate_threshold
        ).logical_error_rate(code_distance, physical_error)
        assert isinstance(value, float)
        return value

    def reference_distance(
        self: SurfaceCodeScheme, physical_error: float, budget: float
    ) -> int:
        inversions.append(budget)
        value = source.QECScheme(
            self.error_rate_scaler, self.error_rate_threshold
        ).code_distance_from_budget(physical_error, budget)
        assert isinstance(value, int)
        return value

    monkeypatch.setattr(SurfaceCodeScheme, "logical_error_rate", reference_rate)
    monkeypatch.setattr(
        SurfaceCodeScheme, "code_distance_from_budget", reference_distance
    )
    PhysicalCostModel.estimate.cache_clear()
    assert scenarios() == expected
    assert rates == {0.03, 0.1}
    assert len(inversions) == 1
