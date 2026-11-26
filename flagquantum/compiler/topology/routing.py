"""Validate routing inputs and coordinate strategy selection and instruction emission.

Connectivity lives in ``coupling.py`` and ``directed_topology.py``. Candidate
pricing and selection live in ``routing_strategy.py``. ``routing_emission.py``
materializes routed instructions, restores layouts and records routing evidence.
Existing routing imports remain available as direct aliases here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ...core.ir import CircuitIR, ensure_circuit_ir
from ..optimization.ordering_barrier import is_ordering_barrier as is_ordering_barrier
from .coupling import UNREACHABLE_DISTANCE as UNREACHABLE_DISTANCE
from .coupling import CouplingMap as CouplingMap
from .routing_cost import LinkErrorRates as LinkErrorRates
from .routing_cost import (
    _PricedDevice,
    _topology_cost_model,
    require_weight_reading_strategy,
)
from .routing_cost import (
    emitted_link_faults as emitted_link_faults,
)
from .routing_emission import _route_persistent_layout as _route_persistent_layout
from .routing_emission import _route_restore_after_each_gate
from .routing_emission import _route_sabre as _route_sabre
from .routing_emission import (
    record_post_routing_optimization as record_post_routing_optimization,
)
from .routing_strategy import (
    DEPLOYABLE_ROUTING_STRATEGIES as DEPLOYABLE_ROUTING_STRATEGIES,
)
from .routing_strategy import (
    LINK_ERROR_ROUTING_STRATEGIES as LINK_ERROR_ROUTING_STRATEGIES,
)
from .routing_strategy import (
    ROUTING_STRATEGIES as ROUTING_STRATEGIES,
)
from .routing_strategy import (
    SABRE_ROUTING_STRATEGIES as SABRE_ROUTING_STRATEGIES,
)
from .routing_strategy import (
    RoutingCostEstimate as RoutingCostEstimate,
)
from .routing_strategy import (
    RoutingStrategySelection as RoutingStrategySelection,
)
from .routing_strategy import (
    _select_routing_strategy as _select_routing_strategy,
)
from .routing_strategy import (
    estimate_routing_cost as estimate_routing_cost,
)
from .routing_strategy import (
    select_routing_strategy as select_routing_strategy,
)
from .sabre import (
    SabrePlan as SabrePlan,
)
from .sabre import (
    plan_restore_swaps as plan_restore_swaps,
)
from .sabre import (
    plan_sabre_layout as plan_sabre_layout,
)
from .sabre import (
    plan_sabre_swaps as plan_sabre_swaps,
)


def _route_automatically(
    ir: CircuitIR,
    coupling_map: _PricedDevice,
    *,
    route_program: Callable[..., CircuitIR] | None = None,
) -> tuple[CircuitIR, RoutingStrategySelection]:
    """Select and route using one call-local SABRE plan, without a global cache."""
    topology = _topology_cost_model(coupling_map, ir.n_wires, "select_routing_strategy")
    selection, plan = _select_routing_strategy(ir, topology)
    selected_router = route_to_topology if route_program is None else route_program
    if selected_router is not route_to_topology:
        routed = selected_router(ir, coupling_map, strategy=selection.selected_strategy)
    elif plan is None:
        routed = route_to_topology(ir, topology, strategy=selection.selected_strategy)
    else:
        routed = _route_sabre(
            ir,
            topology,
            strategy=selection.selected_strategy,
            path_cache_before=topology.path_cache_info(),
            plan=plan,
        )
    return routed, selection


def route_to_topology(
    circuit_or_ir: Any,
    coupling_map: _PricedDevice,
    *,
    strategy: str = "restore_after_each_gate",
) -> CircuitIR:
    """Insert SWAP gates so two-qubit operations respect hardware topology.

    An undirected device is routed as before: a gate is executable wherever its
    two qubits are linked, so a hop is inserted only to bring the qubits together.

    A device that also states a per-link error rate is routed along its cleanest
    route rather than its shortest: the two shortest-path strategies take the route
    from one cost-model entry, so the calibration changes which physical qubits a
    gate is brought to without changing anything else about the program. The
    ``expected_link_faults`` key in the routing metadata reports what the emitted
    program pays for that, and it is ``None`` on a device that declares only its
    couplings. A planner that chooses routes by its own search is refused on such a
    device, because it would route without reading the calibration.

    An ordered device is routed against the direction of its links as well. A gate
    whose operands run against the direction its link is declared in is not
    executable where it stands even though the link exists, and the router repairs
    it with the one instruction that is legal across that link either way: a SWAP
    between the operands. That repair costs a SWAP and leaves the output layout
    where the strategy intended, because every strategy here either restores the
    identity layout or wraps the gate -- the persistent strategies keep the repair
    in their mapping and undo it with their own final restore, while
    ``restore_after_each_gate`` sandwiches the gate. Emitting the repair is what
    keeps ``direction_swap_count`` a price of this router rather than a bill handed
    to a later pass.

    Args:
        circuit_or_ir: The program to route.
        coupling_map: The device to route against, as a ``CouplingMap``, a
            ``DirectedCouplingMap``, a ``LinkErrorRates`` device, or an edge
            sequence.
        strategy: One of ``ROUTING_STRATEGIES``.

    Raises:
        TypeError: If ``coupling_map`` is none of the three accepted forms.
        ValueError: If ``strategy`` is not in ``ROUTING_STRATEGIES``, the coupling
            map has fewer qubits than the circuit, a needed path leaves the device,
            a placed two-qubit opcode has no recorded operand semantics on an
            ordered device, or a calibrated device is handed to a strategy that
            does not read the calibration.
    """

    if strategy not in ROUTING_STRATEGIES:
        raise ValueError(
            "routing strategy must be one of "
            + ", ".join(repr(name) for name in ROUTING_STRATEGIES)
        )
    ir = ensure_circuit_ir(circuit_or_ir)
    topology = _topology_cost_model(coupling_map, ir.n_wires, "route_to_topology")
    if topology.n_qubits < ir.n_wires:
        raise ValueError("Coupling map has fewer qubits than the circuit.")
    path_cache_before = topology.path_cache_info()
    if strategy == "persistent_layout":
        return _route_persistent_layout(
            ir,
            topology,
            path_cache_before=path_cache_before,
        )
    if strategy in SABRE_ROUTING_STRATEGIES:
        require_weight_reading_strategy(
            strategy,
            topology,
            "route_to_topology",
            LINK_ERROR_ROUTING_STRATEGIES,
        )
        return _route_sabre(
            ir,
            topology,
            strategy=strategy,
            path_cache_before=path_cache_before,
        )

    return _route_restore_after_each_gate(
        ir,
        topology,
        strategy=strategy,
        path_cache_before=path_cache_before,
    )


__all__ = [
    "CouplingMap",
    "LINK_ERROR_ROUTING_STRATEGIES",
    "LinkErrorRates",
    "RoutingCostEstimate",
    "RoutingStrategySelection",
    "estimate_routing_cost",
    "record_post_routing_optimization",
    "route_to_topology",
    "select_routing_strategy",
]
