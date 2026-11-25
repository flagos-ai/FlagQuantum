"""Compatibility imports for :mod:`flagquantum.compiler.topology.routing`."""

from .topology.routing import (
    DEPLOYABLE_ROUTING_STRATEGIES as DEPLOYABLE_ROUTING_STRATEGIES,
)
from .topology.routing import (
    LINK_ERROR_ROUTING_STRATEGIES as LINK_ERROR_ROUTING_STRATEGIES,
)
from .topology.routing import (
    ROUTING_STRATEGIES as ROUTING_STRATEGIES,
)
from .topology.routing import (
    SABRE_ROUTING_STRATEGIES as SABRE_ROUTING_STRATEGIES,
)
from .topology.routing import (
    UNREACHABLE_DISTANCE as UNREACHABLE_DISTANCE,
)
from .topology.routing import (
    CouplingMap as CouplingMap,
)
from .topology.routing import (
    LinkErrorRates as LinkErrorRates,
)
from .topology.routing import (
    RoutingCostEstimate as RoutingCostEstimate,
)
from .topology.routing import (
    RoutingStrategySelection as RoutingStrategySelection,
)
from .topology.routing import __all__ as __all__
from .topology.routing import (
    _routing_metadata as _routing_metadata,
)
from .topology.routing import (
    estimate_routing_cost as estimate_routing_cost,
)
from .topology.routing import (
    record_post_routing_optimization as record_post_routing_optimization,
)
from .topology.routing import (
    route_to_topology as route_to_topology,
)
from .topology.routing import (
    select_routing_strategy as select_routing_strategy,
)
from .topology.sabre import plan_restore_swaps as plan_restore_swaps
from .topology.sabre import plan_sabre_layout as plan_sabre_layout
from .topology.sabre import plan_sabre_swaps as plan_sabre_swaps
