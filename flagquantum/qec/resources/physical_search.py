# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 surface_code/gidney_fowler_model.py at
# bcb17bb25360ed68ce3bc084645db3e929799389.

"""Budget-based data distance and ordered factory/layout design searches."""

from collections.abc import Callable, Iterable, Iterator
from math import ceil

from ...compiler._component_trace.resource_model import ModeledResources
from .ccz_to_t_factory import CCZToTFactory
from .data_layout import RoutingDataLayout, SurfaceCodeDataLayout
from .error_model import LogicalErrorModel, SurfaceCodeScheme
from .magic_state_factory import MagicStateFactory
from .parallel_factory import ParallelFactory
from .physical_cost import PhysicalCostModel, PhysicalResourceEstimate
from .physical_parameters import PhysicalParameters


def estimate_gidney_fowler_from_budget(
    logical: ModeledResources,
    *,
    physical_error: float = 1e-3,
    error_budget: float = 1e-2,
    cycle_time_us: float = 1.0,
    routing_overhead: float = 0.5,
    factory: MagicStateFactory | None = None,
    data_block: SurfaceCodeDataLayout | None = None,
) -> PhysicalResourceEstimate:
    """Choose the source data distance from the budget left after distillation.

    An explicit data layout supplies its own distance and routing assumptions.
    Source empty-demand division and nonpositive logarithm failures propagate.
    """
    if factory is None:
        factory = CCZToTFactory()
    scheme = SurfaceCodeScheme.gidney_fowler()
    error_model = LogicalErrorModel(physical_error, scheme)
    factory_error = factory.factory_error(logical.gates, error_model)
    cycles = factory.n_cycles(logical.gates, error_model)
    if data_block is None:
        remaining_budget = error_budget - factory_error
        if remaining_budget < 0:
            raise ValueError(
                f"Factory error {factory_error} is larger than the error budget {error_budget}"
            )
        tiles = ceil((1 + routing_overhead) * logical.peak_logical_qubits)
        cells = tiles * cycles
        error_per_cycle = float(remaining_budget / cells)
        distance = scheme.code_distance_from_budget(physical_error, error_per_cycle)
        data_block = RoutingDataLayout(distance, routing_overhead)
    return PhysicalCostModel(
        PhysicalParameters(physical_error, cycle_time_us),
        data_block,
        factory,
        scheme,
    ).estimate(logical)


def iter_ccz_factories(
    level_one_start: int = 5,
    level_one_stop: int = 25,
    level_two_stop: int = 41,
    *,
    n_factories: int = 1,
) -> Iterator[MagicStateFactory]:
    """Yield the original odd-distance grid, optionally with parallel replicas."""
    if n_factories < 1:
        raise ValueError("The number of factories should be a positive integer")
    for level_one in range(level_one_start, level_one_stop, 2):
        for level_two in range(level_one + 2, level_two_stop, 2):
            factory: MagicStateFactory = CCZToTFactory(level_one, level_two)
            if n_factories > 1:
                factory = ParallelFactory(factory, n_factories)
            yield factory


def iter_routing_data_layouts(
    distance_start: int = 7, distance_stop: int = 35
) -> Iterator[RoutingDataLayout]:
    """Yield the source odd-distance grid with default routing overhead."""
    for distance in range(distance_start, distance_stop, 2):
        yield RoutingDataLayout(distance)


def search_gidney_fowler_resources(
    logical: ModeledResources,
    *,
    physical_error: float = 1e-3,
    error_budget: float = 1e-2,
    cycle_time_us: float = 1.0,
    factories: Iterable[MagicStateFactory] = tuple(iter_ccz_factories()),
    data_blocks: Iterable[SurfaceCodeDataLayout] = tuple(iter_routing_data_layouts()),
    cost_function: Callable[
        [PhysicalResourceEstimate], float
    ] = lambda cost: cost.qubit_hours,
) -> tuple[PhysicalResourceEstimate, MagicStateFactory, SurfaceCodeDataLayout]:
    """Select the first minimum-cost admissible pair in source traversal order.

    The nested traversal retains the source behavior for one-pass iterables.
    A consumed data-layout iterator is not restarted for subsequent factories.
    """
    best_cost: PhysicalResourceEstimate | None = None
    best_parameters: tuple[MagicStateFactory, SurfaceCodeDataLayout] | None = None
    hardware = PhysicalParameters(physical_error, cycle_time_us)
    scheme = SurfaceCodeScheme.gidney_fowler()
    for factory in factories:
        for data_block in data_blocks:
            cost = PhysicalCostModel(hardware, data_block, factory, scheme).estimate(
                logical
            )
            if cost.failure_probability > error_budget:
                continue
            if best_cost is None or cost_function(cost) < cost_function(best_cost):
                best_cost = cost
                best_parameters = factory, data_block
    if best_parameters is None or best_cost is None:
        raise ValueError("No valid factories found!")
    return best_cost, *best_parameters
