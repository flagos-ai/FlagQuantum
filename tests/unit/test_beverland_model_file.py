"""All four source Beverland formulas with replaceable rotation preparation."""

from dataclasses import asdict
from hashlib import sha1
from pathlib import Path

import pytest
from qualtran.resource_counting import GateCounts
from qualtran.surface_code import AlgorithmSummary, QECScheme
from qualtran.surface_code import beverland_et_al_model as source
from qualtran.surface_code import rotation_cost_model as rotation_source

from flagquantum.compiler._component_trace.resource_model import (
    GateCost,
    ModeledResources,
)
from flagquantum.qec.resources import beverland
from flagquantum.qec.resources.error_model import SurfaceCodeScheme
from flagquantum.qec.resources.rotation_synthesis import (
    BEVERLAND_ROTATION_COST,
    SEVEN_BIT_PHASE_GRADIENT_COST,
    LogarithmicRotationCost,
    PhaseGradientRotationCost,
    RotationCostModel,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "native_model,source_model",
    (
        (BEVERLAND_ROTATION_COST, rotation_source.BeverlandEtAlRotationCost),
        (
            SEVEN_BIT_PHASE_GRADIENT_COST,
            rotation_source.SevenDigitsOfPrecisionConstantCost,
        ),
        (
            PhaseGradientRotationCost(13, LogarithmicRotationCost(1, 1)),
            rotation_source.ConstantWithOverheadRotationCost(
                13, rotation_source.RotationLogarithmicModel(1, 1)
            ),
        ),
    ),
)
@pytest.mark.parametrize("rotations", (0, 1, 4, 100))
@pytest.mark.parametrize("layers", (None, 0, 1, 4))
def test_all_four_full_formulas(
    native_model: RotationCostModel,
    source_model: rotation_source.RotationCostModel,
    rotations: int,
    layers: int | None,
) -> None:
    logical = ModeledResources(
        GateCost(
            t=3,
            toffoli=2,
            cswap=5,
            and_bloq=7,
            rotation=rotations,
            measurement=11,
            clifford=13,
        ),
        17,
    )
    alg = AlgorithmSummary(
        n_algo_qubits=17,
        n_logical_gates=GateCounts(**asdict(logical.gates)),
        n_rotation_layers=layers,
    )
    budget = 1e-3
    assert beverland.minimum_time_steps(
        logical,
        error_budget=budget,
        rotation_model=native_model,
        rotation_layers=layers,
    ) == source.minimum_time_steps(
        error_budget=budget, alg=alg, rotation_model=source_model
    )
    actual = beverland.discrete_logical_gates(
        logical, synthesis_budget=budget / 3, rotation_model=native_model
    )
    original = source.n_discrete_logical_gates(
        eps_syn=budget / 3, alg=alg, rotation_model=source_model
    )
    assert {
        key: value for key, value in asdict(actual).items() if value
    } == original.asdict()
    assert beverland.t_states(
        logical, error_budget=budget, rotation_model=native_model
    ) == source.t_states(error_budget=budget, alg=alg, rotation_model=source_model)
    assert beverland.code_distance(
        logical,
        error_budget=budget,
        time_steps=123,
        qec_scheme=SurfaceCodeScheme.beverland(),
        physical_error=1e-4,
    ) == source.code_distance(
        error_budget=budget,
        time_steps=123,
        alg=alg,
        qec_scheme=QECScheme.make_beverland_et_al(),
        physical_error=1e-4,
    )


def test_complete_frozen_source_and_test_files() -> None:
    assert source.__file__ is not None
    for name, blob in (
        ("beverland_et_al_model.py", "239a262863e381f6d05653398120ee74406ba1df"),
        ("beverland_et_al_model_test.py", "5da9c101d708bf515e8573a3df46d6feb0c830f6"),
    ):
        raw = Path(source.__file__).with_name(name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


def test_existing_estimation_consumer_uses_replaced_rotation_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class SourceModel(RotationCostModel):
        def rotation_cost(self, error_budget: float) -> GateCost:
            calls.append("rotation")
            return GateCost(
                **rotation_source.BeverlandEtAlRotationCost.rotation_cost(
                    error_budget
                ).asdict()
            )

        def preparation_overhead(self, error_budget: float) -> GateCost:
            calls.append("preparation")
            return GateCost(
                **rotation_source.BeverlandEtAlRotationCost.preparation_overhead(
                    error_budget
                ).asdict()
            )

    logical = ModeledResources(GateCost(t=3, toffoli=5, rotation=7, measurement=11), 17)
    expected = beverland.estimate_beverland_resources(
        logical, error_budget=1e-3, physical_error=1e-4
    )
    monkeypatch.setattr(beverland, "BEVERLAND_ROTATION_COST", SourceModel())
    assert (
        beverland.estimate_beverland_resources(
            logical, error_budget=1e-3, physical_error=1e-4
        )
        == expected
    )
    assert calls == ["rotation", "preparation", "preparation", "rotation"]
