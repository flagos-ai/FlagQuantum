"""Whole-source physical model composition, defaults and real consumer replacement."""

from dataclasses import asdict
from functools import lru_cache
from hashlib import sha1
from importlib import import_module
from pathlib import Path

import pytest
from qualtran.resource_counting import GateCounts
from qualtran.surface_code import AlgorithmSummary
from qualtran.surface_code import PhysicalCostModel as SourceModel

from flagquantum.compiler._component_trace.gate_budget import GateCategories
from flagquantum.compiler._component_trace.resource_model import (
    GateCost,
    ModeledResources,
)
from flagquantum.core.value_math import SymbolicReal
from flagquantum.qec.resources.beverland_physical import (
    estimate_beverland_physical_resources,
)
from flagquantum.qec.resources.data_block import DataBlock, DataLayout
from flagquantum.qec.resources.error_model import LogicalErrorModel
from flagquantum.qec.resources.gidney_fowler import estimate_gidney_fowler_resources
from flagquantum.qec.resources.magic_state_factory import MagicStateFactory
from flagquantum.qec.resources.physical_cost import (
    PhysicalCostModel,
    PhysicalResourceEstimate,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_frozen_complete_source_test_and_notebook() -> None:
    module = import_module("qualtran.surface_code.physical_cost_model")
    assert module.__file__ is not None
    for name, blob in (
        ("physical_cost_model.py", "4134fe0c77629987e309e0f94fab520e5423ab1f"),
        ("physical_cost_model_test.py", "6db1fd031fa4f39815e152f85125b91650fe39c2"),
        ("physical_cost_model.ipynb", "5cb5070d0ea667fcb51b1fe972258d8761f8240d"),
    ):
        raw = Path(module.__file__).with_name(name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


@pytest.mark.parametrize("architecture", ("gidney_fowler", "beverland"))
@pytest.mark.parametrize("layout", ("simple", "compact", "intermediate", "fast"))
@pytest.mark.parametrize(
    "gates", (GateCost(), GateCost(t=7, toffoli=3, cswap=5, and_bloq=2, rotation=11))
)
def test_complete_cost_composition_and_all_query_methods(
    architecture: str, layout: DataLayout, gates: GateCost
) -> None:
    from qualtran.surface_code import (
        CompactDataBlock,
        FastDataBlock,
        IntermediateDataBlock,
        SimpleDataBlock,
    )

    native = getattr(PhysicalCostModel, architecture)(7)
    source = getattr(
        SourceModel,
        (
            "make_gidney_fowler"
            if architecture == "gidney_fowler"
            else "make_beverland_et_al"
        ),
    )(7)
    layouts = {
        "simple": SimpleDataBlock,
        "compact": CompactDataBlock,
        "intermediate": IntermediateDataBlock,
        "fast": FastDataBlock,
    }
    native = PhysicalCostModel(
        native.physical_params,
        DataBlock(layout, 7)._data_layout(),
        native.factory,
        native.qec_scheme,
    )
    source = SourceModel(
        source.physical_params, layouts[layout](7), source.factory, source.qec_scheme
    )
    logical = ModeledResources(gates, 23)
    reference = AlgorithmSummary(
        n_algo_qubits=23, n_logical_gates=GateCounts(**asdict(gates))
    )
    result = native.estimate(logical)
    assert native.n_cycles(logical) == result.cycles == source.n_cycles(reference)
    assert (
        native.duration_hours(logical)
        == result.duration_hours
        == source.duration_hr(reference)
    )
    assert (
        native.n_physical_qubits(logical)
        == result.physical_qubits
        == source.n_phys_qubits(reference)
    )
    assert (
        native.error(logical)
        == result.failure_probability
        == pytest.approx(source.error(reference), rel=0, abs=4e-12)
    )
    assert result.factory_physical_qubits == source.factory.n_physical_qubits()
    assert result.data_error == source.data_block.data_error(
        23, result.cycles, source.logical_error_model
    )
    assert native.logical_error_model(7) == source.logical_error_model(7)
    assert native.logical_error_model is native.logical_error_model
    assert native.estimate(logical) is result


def test_source_factories_distances_and_unknown_layout() -> None:
    for layout in ("compact", "intermediate", "fast"):
        actual = PhysicalCostModel.beverland(13, layout, (7, 3, 3))
        source = SourceModel.make_beverland_et_al(13, layout, (7, 3, 3))
        assert actual.factory.n_physical_qubits() == source.factory.n_physical_qubits()
        assert actual.data_block.tiles(17) == source.data_block.n_tiles(17)
    with pytest.raises(ValueError):
        PhysicalCostModel.beverland(13, "simple")
    with pytest.raises(ValueError):
        SourceModel.make_beverland_et_al(13, "simple")

    class ModelSubclass(PhysicalCostModel):
        pass

    assert isinstance(ModelSubclass.gidney_fowler(7), ModelSubclass)
    assert isinstance(ModelSubclass.beverland(7), ModelSubclass)


def test_factory_boundary_cache_and_consumption_bottleneck() -> None:
    calls: list[str] = []

    class FixedFactory(MagicStateFactory):
        def n_physical_qubits(self) -> int:
            calls.append("footprint")
            return 100

        def n_cycles(
            self,
            n_logical_gates: GateCategories,
            logical_error_model: LogicalErrorModel,
        ) -> int:
            calls.append("cycles")
            return 1

        def factory_error(
            self,
            n_logical_gates: GateCategories,
            logical_error_model: LogicalErrorModel,
        ) -> SymbolicReal:
            calls.append("error")
            return 0.01

    default = PhysicalCostModel.gidney_fowler(7)
    model = PhysicalCostModel(
        default.physical_params,
        DataBlock("compact", 7)._data_layout(),
        FixedFactory(),
        default.qec_scheme,
    )
    logical = ModeledResources(GateCost(t=2), 3)
    result = model.estimate(logical)
    assert result.cycles == 126
    assert result.factory_physical_qubits == 100
    assert result.factory_error == 0.01
    assert model.n_cycles(logical) == 126
    assert model.n_physical_qubits(logical) == result.physical_qubits
    assert model.error(logical) == result.failure_probability
    assert calls == ["cycles", "footprint", "error"]
    assert model.estimate.cache_info().maxsize == 128


def test_both_real_consumers_use_replaceable_shared_aggregation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flagquantum.qec.resources.beverland_physical as beverland
    import flagquantum.qec.resources.gidney_fowler as gidney

    logical = ModeledResources(GateCost(t=7, toffoli=3, rotation=2), 19)
    expected_gf = estimate_gidney_fowler_resources(
        logical, data_distance=7, factory_count=3
    )
    expected_beverland = estimate_beverland_physical_resources(
        logical, data_distance=7, factory_count=3
    )
    calls: list[float] = []

    class Replacement(PhysicalCostModel):
        @lru_cache
        def estimate(self, logical: ModeledResources) -> PhysicalResourceEstimate:
            calls.append(self.qec_scheme.error_rate_scaler)
            return super().estimate(logical)

    monkeypatch.setattr(gidney, "PhysicalCostModel", Replacement)
    monkeypatch.setattr(beverland, "PhysicalCostModel", Replacement)
    assert (
        estimate_gidney_fowler_resources(logical, data_distance=7, factory_count=3)
        == expected_gf
    )
    assert (
        estimate_beverland_physical_resources(logical, data_distance=7, factory_count=3)
        == expected_beverland
    )
    assert calls == [0.1, 0.03]


def test_all_48_original_notebook_scenarios() -> None:
    from itertools import product

    from qualtran.surface_code import (
        CCZ2TFactory,
        CompactDataBlock,
        FastDataBlock,
        FifteenToOne,
        QECScheme,
        SimpleDataBlock,
    )
    from qualtran.surface_code import (
        PhysicalParameters as SourceHardware,
    )

    from flagquantum.qec.resources.ccz_to_t_factory import CCZToTFactory
    from flagquantum.qec.resources.error_model import SurfaceCodeScheme
    from flagquantum.qec.resources.fifteen_to_one import FifteenToOneFactory
    from flagquantum.qec.resources.physical_parameters import PhysicalParameters

    layouts: tuple[DataLayout, ...] = ("simple", "compact", "fast")
    source_layouts = {
        "simple": SimpleDataBlock,
        "compact": CompactDataBlock,
        "fast": FastDataBlock,
    }
    factories = (
        (CCZToTFactory(), CCZ2TFactory()),
        (FifteenToOneFactory(25, 25, 25), FifteenToOne(25, 25, 25)),
    )
    schemes = (
        (SurfaceCodeScheme.gidney_fowler(), QECScheme.make_gidney_fowler()),
        (SurfaceCodeScheme.beverland(), QECScheme.make_beverland_et_al()),
    )
    hardware = tuple(
        (
            native(optimistic_error_rate=optimistic),
            source(optimistic_err_rate=optimistic),
        )
        for native, source in (
            (PhysicalParameters.beverland, SourceHardware.make_beverland_et_al),
            (PhysicalParameters.gidney_fowler, SourceHardware.make_gidney_fowler),
        )
        for optimistic in (False, True)
    )
    logical = ModeledResources(GateCost(toffoli=10**8), 100)
    summary = AlgorithmSummary(
        n_algo_qubits=100, n_logical_gates=GateCounts(toffoli=10**8)
    )
    scenarios = 0
    differences: list[float] = []
    for (
        layout,
        (factory, source_factory),
        (scheme, source_scheme),
        (physical, source_physical),
    ) in product(layouts, factories, schemes, hardware):
        actual = PhysicalCostModel(
            physical, DataBlock(layout, 25)._data_layout(), factory, scheme
        ).estimate(logical)
        expected = SourceModel(
            source_physical, source_layouts[layout](25), source_factory, source_scheme
        )
        assert actual.cycles == expected.n_cycles(summary)
        assert actual.duration_hours == expected.duration_hr(summary)
        assert actual.physical_qubits == expected.n_phys_qubits(summary)
        # Density-matrix rounding accumulates with the 400-million-T demand.
        assert actual.failure_probability == pytest.approx(
            expected.error(summary), rel=0, abs=8e-7
        )
        differences.append(abs(actual.failure_probability - expected.error(summary)))
        scenarios += 1
    assert scenarios == 48
    print(f"Original notebook error max absolute difference: {max(differences):.17g}")
