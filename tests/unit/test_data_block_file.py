"""Complete surface-code data layouts and the production estimator boundary."""

from collections.abc import Callable
from dataclasses import FrozenInstanceError, asdict
from hashlib import sha1
from importlib import import_module
from pathlib import Path

import pytest
import sympy
from qualtran.resource_counting import GateCounts
from qualtran.surface_code import (
    CompactDataBlock,
    FastDataBlock,
    IntermediateDataBlock,
    QECScheme,
    SimpleDataBlock,
)
from qualtran.surface_code import LogicalErrorModel as SourceErrorModel
from qualtran.surface_code.data_block import DataBlock as SourceDataBlock

from flagquantum.compiler._component_trace.gate_budget import GateCategories
from flagquantum.compiler._component_trace.qec_gate_cost import QECGateCost
from flagquantum.compiler._component_trace.resource_model import (
    GateCost,
    ModeledResources,
)
from flagquantum.core.value_math import SymbolicInt
from flagquantum.qec.resources.data_block import DataBlock, DataLayout
from flagquantum.qec.resources.data_layout import (
    RoutingDataLayout,
    SingleHallwayDataLayout,
    SquareDataLayout,
    SurfaceCodeDataLayout,
    TwoRowDataLayout,
)
from flagquantum.qec.resources.error_model import LogicalErrorModel, SurfaceCodeScheme
from flagquantum.qec.resources.gidney_fowler import estimate_gidney_fowler_resources

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]

LAYOUTS = (
    ("simple", RoutingDataLayout, SimpleDataBlock),
    ("compact", SingleHallwayDataLayout, CompactDataBlock),
    ("intermediate", TwoRowDataLayout, IntermediateDataBlock),
    ("fast", SquareDataLayout, FastDataBlock),
)


def test_complete_files_and_abstract_interface() -> None:
    module = import_module("qualtran.surface_code.data_block")
    assert module.__file__ is not None
    for filename, blob in (
        ("data_block.py", "c0237613c2e1ef7d56726ba33c5469712b84a66f"),
        ("data_block_test.py", "86931572a58dfcd0aaa6dc9b88ba3008d13f8eec"),
    ):
        raw = Path(module.__file__).with_name(filename).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob
    assert SurfaceCodeDataLayout.__abstractmethods__ == {
        "distance",
        "tiles",
        "steps_per_magic_state",
    }
    assert SourceDataBlock.__abstractmethods__ == {
        "data_d",
        "n_tiles",
        "n_steps_to_consume_a_magic_state",
    }
    for layout in (SurfaceCodeDataLayout, SourceDataBlock):
        constructor: Callable[[], object] = layout
        with pytest.raises(TypeError):
            constructor()


@pytest.mark.parametrize("name,native,source", LAYOUTS)
@pytest.mark.parametrize("distance", (0, 3, 7, 21))
@pytest.mark.parametrize("width", (0, 1, 7, 100, 1318, 12581))
def test_full_layout_geometry_consumption_and_error(
    name: DataLayout,
    native: Callable[[int], SurfaceCodeDataLayout],
    source: Callable[[int], SourceDataBlock],
    distance: int,
    width: int,
) -> None:
    actual, expected = native(distance), source(distance)
    gates = GateCost(t=3, toffoli=5, cswap=2, and_bloq=4, rotation=7)
    model = LogicalErrorModel(1e-3, SurfaceCodeScheme.gidney_fowler())
    reference = SourceErrorModel(1e-3, QECScheme.make_gidney_fowler())
    assert actual.tiles(width) == expected.n_tiles(width)
    assert actual.physical_qubits(width) == expected.n_physical_qubits(width)
    assert actual.steps_per_magic_state == expected.n_steps_to_consume_a_magic_state
    assert actual.consumption_cycles(gates) == expected.n_cycles(
        GateCounts(**asdict(gates)), reference
    )
    assert actual.storage_error(width, 137, model) == expected.data_error(
        width, 137, reference
    )
    if distance:
        facade = DataBlock(name, distance)
        assert facade.tiles(width) == actual.tiles(width)
        assert facade.physical_qubits(width) == actual.physical_qubits(width)
        assert facade.consumption_cycles(gates) == actual.consumption_cycles(gates)
        assert facade.storage_error(
            width, 137, model(distance)
        ) == actual.storage_error(width, 137, model)


@pytest.mark.parametrize("overhead", (-0.5, 0.0, 0.25, 0.5, 1.3))
def test_custom_routing_and_immutable_fields(overhead: float) -> None:
    actual, source = RoutingDataLayout(7, overhead), SimpleDataBlock(7, overhead)
    assert actual.routing_overhead == source.routing_overhead
    for width in (0, 1, 2, 7, 39):
        assert actual.tiles(width) == source.n_tiles(width)
    with pytest.raises(FrozenInstanceError):
        setattr(actual, "distance", 9)


@pytest.mark.parametrize("name,native,source", LAYOUTS)
def test_symbolic_and_empty_demand(
    name: DataLayout,
    native: Callable[[int], SurfaceCodeDataLayout],
    source: Callable[[int], SourceDataBlock],
) -> None:
    count = sympy.Symbol("count", integer=True, positive=True)
    model = SourceErrorModel(1e-3, QECScheme.make_gidney_fowler())
    actual, expected = native(7), source(7)
    assert (
        actual.consumption_cycles(GateCost())
        == expected.n_cycles(GateCounts(), model)
        == 0
    )
    assert (
        sympy.simplify(
            actual.consumption_cycles(QECGateCost(t=count, toffoli=2 * count))
            - expected.n_cycles(GateCounts(t=count, toffoli=2 * count), model)
        )
        == 0
    )


@pytest.mark.parametrize("scheme", ("gidney_fowler", "beverland"))
@pytest.mark.parametrize("width", (0, 1, 100, 1318))
@pytest.mark.parametrize("budget", (1e-3, 1e-7, 1e-12))
def test_budget_factory_and_subclass_construction(
    scheme: str, width: int, budget: float
) -> None:
    native_scheme = getattr(SurfaceCodeScheme, scheme)()
    source_scheme = (
        QECScheme.make_gidney_fowler()
        if scheme == "gidney_fowler"
        else QECScheme.make_beverland_et_al()
    )
    actual = SquareDataLayout.from_error_budget(budget, width, native_scheme, 1e-3)
    expected = FastDataBlock.from_error_budget(budget, width, source_scheme, 1e-3)
    assert actual.distance == expected.data_d
    assert actual.tiles(width) == expected.n_tiles(width)

    class ReplacementSquare(SquareDataLayout):
        pass

    assert isinstance(
        ReplacementSquare.from_error_budget(budget, width, native_scheme, 1e-3),
        ReplacementSquare,
    )


def test_arithmetic_domains_and_existing_validation_are_preserved() -> None:
    for native, source in ((SquareDataLayout.tile_count, FastDataBlock.get_n_tiles),):
        with pytest.raises(ValueError):
            native(-1)
        with pytest.raises(ValueError):
            source(-1)
    for budget, physical in ((0.0, 1e-3), (1e-3, 0.0), (1e-3, 0.01)):
        with pytest.raises((ValueError, ZeroDivisionError)) as reference:
            FastDataBlock.from_error_budget(
                budget, 7, QECScheme.make_gidney_fowler(), physical
            )
        with pytest.raises(type(reference.value)):
            SquareDataLayout.from_error_budget(
                budget, 7, SurfaceCodeScheme.gidney_fowler(), physical
            )
    for invalid_distance in (0, -1):
        with pytest.raises(ValueError):
            DataBlock("fast", invalid_distance)
    for method in (DataBlock("simple", 7).tiles, DataBlock("fast", 7).physical_qubits):
        with pytest.raises(ValueError):
            method(-1)


def test_production_estimator_accepts_replaced_layout_implementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import flagquantum.qec.resources.data_block as boundary

    logical = ModeledResources(GateCost(t=7, toffoli=11, rotation=3), 23)
    expected = estimate_gidney_fowler_resources(
        logical, data_distance=7, data_layout="fast"
    )
    calls: list[str] = []

    class ReferenceLayout(SquareDataLayout):
        def tiles(self, logical_qubits: int) -> int:
            calls.append("tiles")
            return int(FastDataBlock(self.distance).n_tiles(logical_qubits))

        def consumption_cycles(self, gates: GateCategories) -> SymbolicInt:
            calls.append("cycles")
            counts = GateCounts(
                t=gates.t,
                toffoli=gates.toffoli,
                cswap=gates.cswap,
                and_bloq=gates.and_bloq,
                rotation=gates.rotation,
            )
            result: SymbolicInt = FastDataBlock(self.distance).n_cycles(
                counts, SourceErrorModel(1e-3, QECScheme.make_gidney_fowler())
            )
            return result

    monkeypatch.setitem(boundary._LAYOUTS, "fast", ReferenceLayout)
    assert (
        estimate_gidney_fowler_resources(logical, data_distance=7, data_layout="fast")
        == expected
    )
    assert "cycles" in calls and "tiles" in calls
