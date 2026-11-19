"""Whole-source factory interface, catalysis stages and parallel composition."""

from collections.abc import Callable
from dataclasses import asdict
from hashlib import sha1
from importlib import import_module
from pathlib import Path

import pytest
import sympy
from qualtran.resource_counting import GateCounts
from qualtran.surface_code import CCZ2TFactory, MultiFactory, QECScheme
from qualtran.surface_code import LogicalErrorModel as SourceErrorModel
from qualtran.surface_code.magic_state_factory import MagicStateFactory as SourceFactory

from flagquantum.compiler._component_trace.gate_budget import GateCategories
from flagquantum.compiler._component_trace.resource_model import (
    GateCost,
    ModeledResources,
)
from flagquantum.qec.resources.ccz_to_t_factory import CCZToTFactory
from flagquantum.qec.resources.error_model import LogicalErrorModel, SurfaceCodeScheme
from flagquantum.qec.resources.gidney_fowler import estimate_gidney_fowler_resources
from flagquantum.qec.resources.magic_state_factory import MagicStateFactory
from flagquantum.qec.resources.parallel_factory import ParallelFactory

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_all_complete_source_blobs_and_interface_are_preserved() -> None:
    for name, blob in (
        ("magic_state_factory.py", "8f6020554bb004567c7e77eab771a3446557ac51"),
        ("ccz2t_factory.py", "937748981f673d46d081a42b1904a2e3cfbd7529"),
        ("ccz2t_factory_test.py", "5fd35ce68b0caaa4a1267a82e8c0c11d33258772"),
        ("multi_factory.py", "3d57ab743ff595cb3c7fdccdf0a033b22cfeedde"),
    ):
        module = import_module("qualtran.surface_code.magic_state_factory")
        assert module.__file__ is not None
        raw = Path(module.__file__).with_name(name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob
    assert (
        MagicStateFactory.__abstractmethods__
        == SourceFactory.__abstractmethods__
        == {"n_physical_qubits", "n_cycles", "factory_error"}
    )
    for factory_type in (MagicStateFactory, SourceFactory):
        with pytest.raises(TypeError):
            constructor: Callable[[], object] = factory_type
            constructor()
    assert isinstance(CCZToTFactory(), MagicStateFactory)
    assert isinstance(ParallelFactory(CCZToTFactory(), 2), MagicStateFactory)


@pytest.mark.parametrize("level_one,level_two", ((2, 1), (7, 15), (9, 13), (15, 31)))
@pytest.mark.parametrize("physical", (0.0, 1e-6, 1e-4, 1e-3, 0.02))
def test_all_eight_distillation_stages_footprints_and_named_parameters(
    level_one: int, level_two: int, physical: float
) -> None:
    actual = CCZToTFactory(level_one_distance=level_one, level_two_distance=level_two)
    expected = CCZ2TFactory(distillation_l1_d=level_one, distillation_l2_d=level_two)
    model = LogicalErrorModel(physical, SurfaceCodeScheme.gidney_fowler())
    source_model = SourceErrorModel(
        physical_error=physical, qec_scheme=QECScheme.make_gidney_fowler()
    )
    for method in (
        "l0_state_injection_error",
        "l0_topo_error_t_gate",
        "l0_error",
        "l1_topo_error_factory",
        "l1_topo_error_t_gate",
        "l1_distillation_error",
        "l1_error",
        "l2_error",
    ):
        assert getattr(actual, method)(model) == getattr(expected, method)(source_model)
    assert actual.n_physical_qubits() == expected.n_physical_qubits()


@pytest.mark.parametrize(
    "gates",
    (
        GateCost(),
        GateCost(t=1),
        GateCost(t=2),
        GateCost(t=3, toffoli=7),
        GateCost(t=10**8, toffoli=10**8),
        GateCost(t=2, rotation=3, and_bloq=4, cswap=5),
    ),
)
@pytest.mark.parametrize("factories", (1, 2, 3, 7))
def test_complete_demand_conversion_and_parallel_factory_rules(
    gates: GateCost, factories: int
) -> None:
    actual = ParallelFactory(CCZToTFactory(), factories)
    original = MultiFactory(CCZ2TFactory(), factories)
    model = LogicalErrorModel(1e-3, SurfaceCodeScheme.gidney_fowler())
    source_model = SourceErrorModel(
        physical_error=1e-3, qec_scheme=QECScheme.make_gidney_fowler()
    )
    source_gates = GateCounts(**asdict(gates))
    assert actual.n_physical_qubits() == original.n_physical_qubits()
    assert actual.n_cycles(gates, model) == original.n_cycles(
        source_gates, source_model
    )
    assert actual.factory_error(gates, model) == original.factory_error(
        source_gates, source_model
    )
    assert actual.factory_error(gates, model) == actual.base_factory.factory_error(
        gates, model
    )


def test_symbolic_error_and_original_unbound_cycle_failure() -> None:
    from flagquantum.compiler._component_trace.qec_gate_cost import QECGateCost

    count = sympy.Symbol("count", positive=True, integer=True)
    actual, original = CCZToTFactory(), CCZ2TFactory()
    model = LogicalErrorModel(1e-3, SurfaceCodeScheme.gidney_fowler())
    source_model = SourceErrorModel(
        physical_error=1e-3, qec_scheme=QECScheme.make_gidney_fowler()
    )
    assert (
        sympy.simplify(
            actual.factory_error(QECGateCost(toffoli=count), model)
            - original.factory_error(GateCounts(toffoli=count), source_model)
        )
        == 0
    )
    with pytest.raises(TypeError):
        original.n_cycles(GateCounts(toffoli=count), source_model)
    with pytest.raises(TypeError):
        actual.n_cycles(QECGateCost(toffoli=count), model)
    with pytest.raises(ZeroDivisionError):
        MultiFactory(original, 0).n_cycles(GateCounts(t=1), source_model)
    with pytest.raises(ZeroDivisionError):
        ParallelFactory(actual, 0).n_cycles(GateCost(t=1), model)


def test_genuine_estimation_consumer_uses_the_factory_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logical = ModeledResources(GateCost(t=5, rotation=2, cswap=3), 9)
    expected = estimate_gidney_fowler_resources(
        logical, data_distance=7, factory_count=3
    )
    calls: list[str] = []

    class ReplacementFactory(CCZToTFactory):
        def n_physical_qubits(self) -> int:
            calls.append("qubits")
            return super().n_physical_qubits()

        def n_cycles(
            self,
            n_logical_gates: GateCategories,
            logical_error_model: LogicalErrorModel,
        ) -> int:
            calls.append("cycles")
            return super().n_cycles(n_logical_gates, logical_error_model)

    # Replace the actual model implementation without changing the estimator.
    import flagquantum.qec.resources.gidney_fowler as consumer

    monkeypatch.setattr(consumer, "CCZToTFactory", ReplacementFactory)
    assert (
        estimate_gidney_fowler_resources(logical, data_distance=7, factory_count=3)
        == expected
    )
    assert calls == ["cycles", "qubits"]
