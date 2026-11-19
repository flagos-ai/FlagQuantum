"""Complete frozen rotation-cost interfaces, models and presets."""

from collections.abc import Callable
from dataclasses import FrozenInstanceError, asdict
from hashlib import sha1
from pathlib import Path

import pytest
from qualtran.surface_code import rotation_cost_model as source

from flagquantum.qec.resources.rotation_synthesis import (
    BEVERLAND_ROTATION_COST,
    SEVEN_BIT_PHASE_GRADIENT_COST,
    LogarithmicRotationCost,
    PhaseGradientRotationCost,
    RotationCostModel,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "native,original",
    (
        (BEVERLAND_ROTATION_COST, source.BeverlandEtAlRotationCost),
        (SEVEN_BIT_PHASE_GRADIENT_COST, source.SevenDigitsOfPrecisionConstantCost),
        (
            LogarithmicRotationCost(2.3, -1.2, "custom"),
            source.RotationLogarithmicModel(2.3, -1.2, "custom"),
        ),
        (
            PhaseGradientRotationCost(13, LogarithmicRotationCost(1, 1)),
            source.ConstantWithOverheadRotationCost(
                13, source.RotationLogarithmicModel(1, 1)
            ),
        ),
        (
            PhaseGradientRotationCost(1, BEVERLAND_ROTATION_COST),
            source.ConstantWithOverheadRotationCost(
                1, source.BeverlandEtAlRotationCost
            ),
        ),
    ),
)
@pytest.mark.parametrize("budget", (1.0, 2.0, 2**-3, 2**-20, 1e-12))
def test_all_complete_model_methods(
    native: RotationCostModel, original: source.RotationCostModel, budget: float
) -> None:
    for method in ("rotation_cost", "preparation_overhead"):
        actual = getattr(native, method)(budget)
        expected = getattr(original, method)(budget)
        assert {
            key: value for key, value in asdict(actual).items() if value
        } == expected.asdict()


def test_frozen_blobs_abstract_interface_and_immutable_models() -> None:
    assert source.__file__ is not None
    for name, blob in (
        ("rotation_cost_model.py", "b747ce18a839b1b5cb0b4b5766c9ba9db1f3fd17"),
        ("rotation_cost_model_test.py", "2b766a5a36f40f813ce592843278c59383ef26d3"),
    ):
        raw = Path(source.__file__).with_name(name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob
    assert (
        RotationCostModel.__abstractmethods__
        == source.RotationCostModel.__abstractmethods__
        == {"rotation_cost", "preparation_overhead"}
    )
    for cls in (RotationCostModel, source.RotationCostModel):
        constructor: Callable[[], object] = cls
        with pytest.raises(TypeError):
            constructor()
    assert (
        BEVERLAND_ROTATION_COST.gateset
        == source.BeverlandEtAlRotationCost.gateset
        == "Clifford+T"
    )
    with pytest.raises(FrozenInstanceError):
        setattr(BEVERLAND_ROTATION_COST, "slope", 1)


def test_original_math_and_zero_precision_failures() -> None:
    for budget in (0.0, -1.0):
        with pytest.raises(ValueError):
            source.BeverlandEtAlRotationCost.rotation_cost(budget)
        with pytest.raises(ValueError):
            BEVERLAND_ROTATION_COST.rotation_cost(budget)
    for model in (
        PhaseGradientRotationCost(0, BEVERLAND_ROTATION_COST),
        source.ConstantWithOverheadRotationCost(0, source.BeverlandEtAlRotationCost),
    ):
        assert model.rotation_cost(0.0).toffoli == 0
        with pytest.raises(ZeroDivisionError):
            model.preparation_overhead(0.1)
