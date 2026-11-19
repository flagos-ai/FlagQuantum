"""Complete budget and ordered-grid source definitions and original examples."""

from dataclasses import asdict
from hashlib import sha1
from pathlib import Path

import pytest
from qualtran.resource_counting import GateCounts
from qualtran.surface_code import CCZ2TFactory, MultiFactory, SimpleDataBlock
from qualtran.surface_code import gidney_fowler_model as source

from flagquantum.compiler._component_trace.resource_model import (
    GateCost,
    ModeledResources,
)
from flagquantum.qec.resources.ccz_to_t_factory import CCZToTFactory
from flagquantum.qec.resources.data_layout import RoutingDataLayout
from flagquantum.qec.resources.parallel_factory import ParallelFactory
from flagquantum.qec.resources.physical_search import (
    estimate_gidney_fowler_from_budget,
    iter_ccz_factories,
    iter_routing_data_layouts,
    search_gidney_fowler_resources,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_all_frozen_source_blobs() -> None:
    assert source.__file__ is not None
    for name, blob in (
        ("gidney_fowler_model.py", "3452b58e02553c2995751898a5cbcfea30e2aee9"),
        ("gidney_fowler_model_test.py", "006dbb7289064df1ed4b88d23018c55718da121a"),
    ):
        raw = Path(source.__file__).with_name(name).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


@pytest.mark.parametrize("replicas", (1, 2, 4))
def test_complete_ordered_factory_and_layout_iterators(replicas: int) -> None:
    actual = list(iter_ccz_factories(5, 12, 17, n_factories=replicas))
    expected = list(source.iter_ccz2t_factories(5, 12, 17, n_factories=replicas))
    assert len(actual) == len(expected)
    for native, original in zip(actual, expected, strict=True):
        if replicas > 1:
            assert isinstance(native, ParallelFactory) and isinstance(
                original, MultiFactory
            )
            assert native.n_factories == original.n_factories == replicas
            native, original = native.base_factory, original.base_factory
        assert isinstance(native, CCZToTFactory) and isinstance(original, CCZ2TFactory)
        assert (native.level_one_distance, native.level_two_distance) == (
            original.distillation_l1_d,
            original.distillation_l2_d,
        )
    assert [x.distance for x in iter_routing_data_layouts(7, 16)] == [
        x.data_d for x in source.iter_simple_data_blocks(7, 16)
    ]
    for invalid in (0, -1):
        with pytest.raises(ValueError):
            list(iter_ccz_factories(n_factories=invalid))
        with pytest.raises(ValueError):
            list(source.iter_ccz2t_factories(n_factories=invalid))


@pytest.mark.parametrize("overhead", (0.0, 0.25, 0.5, 1.3))
def test_budget_inversion_and_explicit_layout_override(overhead: float) -> None:
    logical = ModeledResources(GateCost(t=10**8, toffoli=10**8), 100)
    for native_layout, original_layout in (
        (None, None),
        (RoutingDataLayout(13, overhead), SimpleDataBlock(13, overhead)),
    ):
        actual = estimate_gidney_fowler_from_budget(
            logical, routing_overhead=overhead, data_block=native_layout
        )
        expected = source.get_ccz2t_costs_from_error_budget(
            n_logical_gates=GateCounts(**asdict(logical.gates)),
            n_algo_qubits=100,
            routing_overhead=overhead,
            data_block=original_layout,
        )
        assert actual.failure_probability == expected.failure_prob
        assert actual.physical_qubits == expected.footprint
        assert actual.duration_hours == expected.duration_hr
        assert actual.qubit_hours == expected.qubit_hours


def test_original_paper_grid_and_thc_searches() -> None:
    logical = ModeledResources(GateCost(t=10**8, toffoli=10**8), 100)
    actual, factory, data = search_gidney_fowler_resources(logical, error_budget=0.1)
    expected, original_factory, original_data = source.get_ccz2t_costs_from_grid_search(
        n_logical_gates=GateCounts(**asdict(logical.gates)),
        n_algo_qubits=100,
        error_budget=0.1,
    )
    assert actual.failure_probability == expected.failure_prob
    assert actual.physical_qubits == expected.footprint
    assert actual.duration_hours == expected.duration_hr
    assert isinstance(factory, CCZToTFactory)
    assert (
        (factory.level_one_distance, factory.level_two_distance, data.distance)
        == (
            original_factory.distillation_l1_d,
            original_factory.distillation_l2_d,
            original_data.data_d,
        )
        == (15, 23, 25)
    )
    thc = ModeledResources(GateCost(toffoli=6665400000), 696)
    cost, parallel, layout = search_gidney_fowler_resources(
        thc,
        factories=iter_ccz_factories(n_factories=4),
        cost_function=lambda estimate: estimate.duration_hours,
    )
    assert cost.failure_probability == 0.007725395132201774
    assert cost.physical_qubits == 2933032
    assert cost.duration_hours == 89.1034375
    assert isinstance(parallel, ParallelFactory) and isinstance(
        parallel.base_factory, CCZToTFactory
    )
    assert (
        parallel.base_factory.level_one_distance,
        parallel.base_factory.level_two_distance,
        layout.distance,
    ) == (17, 29, 33)


def test_one_pass_layout_iterator_matches_source_exhaustion_and_ties() -> None:
    logical = ModeledResources(GateCost(t=1), 3)
    factories = (CCZToTFactory(15, 31), CCZToTFactory(5, 7))
    source_factories = (CCZ2TFactory(15, 31), CCZ2TFactory(5, 7))
    native = search_gidney_fowler_resources(
        logical,
        error_budget=100,
        factories=factories,
        data_blocks=iter((RoutingDataLayout(7), RoutingDataLayout(9))),
    )
    original = source.get_ccz2t_costs_from_grid_search(
        n_logical_gates=GateCounts(t=1),
        n_algo_qubits=3,
        error_budget=100,
        factory_iter=source_factories,
        data_block_iter=iter((SimpleDataBlock(7), SimpleDataBlock(9))),
    )
    assert native[1] is factories[0] and original[1] is source_factories[0]
    assert native[2].distance == original[2].data_d == 7
    assert native[0].failure_probability == original[0].failure_prob
    assert native[0].physical_qubits == original[0].footprint
    # A reusable layout sequence reaches the cheaper second factory.
    reusable = search_gidney_fowler_resources(
        logical,
        error_budget=100,
        factories=factories,
        data_blocks=(RoutingDataLayout(7), RoutingDataLayout(9)),
    )
    assert reusable[1] is factories[1]
    tied = search_gidney_fowler_resources(
        logical,
        error_budget=100,
        factories=factories,
        data_blocks=(RoutingDataLayout(7), RoutingDataLayout(9)),
        cost_function=lambda cost: 0,
    )
    assert tied[1] is factories[0] and tied[2].distance == 7


def test_original_empty_demand_zero_budget_and_no_candidate_failures() -> None:
    for gates, width, budget in (
        (GateCost(), 3, 0.01),
        (GateCost(t=1), 0, 0.01),
        (GateCost(t=10**8, toffoli=10**8), 100, 1e-20),
    ):
        with pytest.raises((ValueError, ZeroDivisionError)) as expected:
            source.get_ccz2t_costs_from_error_budget(
                n_logical_gates=GateCounts(**asdict(gates)),
                n_algo_qubits=width,
                error_budget=budget,
            )
        with pytest.raises(type(expected.value)):
            estimate_gidney_fowler_from_budget(
                ModeledResources(gates, width), error_budget=budget
            )
    logical = ModeledResources(GateCost(t=1), 3)
    with pytest.raises(ValueError, match="No valid factories"):
        search_gidney_fowler_resources(logical, factories=())
    with pytest.raises(ValueError, match="No valid factories"):
        source.get_ccz2t_costs_from_grid_search(
            n_logical_gates=GateCounts(t=1), n_algo_qubits=3, factory_iter=()
        )
