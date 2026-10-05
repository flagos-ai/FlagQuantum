import random

import pytest
import torch

import flagquantum as fq
import flagquantum.runtime.planner as fqxp
from flagquantum.compiler.routing import (
    CouplingMap,
    estimate_routing_cost,
    route_to_topology,
    select_routing_strategy,
)

pytestmark = pytest.mark.unit


def _workload() -> fq.Circuit:
    circuit = fq.Circuit(5)
    circuit.h(0).cx(0, 4).ry(2, theta=0.2)
    circuit.swap(4, 1).crz(0, 3, theta=-0.4).cx(2, 4)
    return circuit


@pytest.mark.parametrize(
    "strategy",
    ("restore_after_each_gate", "persistent_layout"),
)
@pytest.mark.parametrize(
    "coupling",
    (
        CouplingMap.line(5),
        CouplingMap.ring(5),
        CouplingMap(5, ((0, 1), (1, 2), (1, 3), (3, 4))),
    ),
    ids=("line", "ring", "irregular"),
)
def test_routing_estimate_matches_materialized_plan(
    strategy: str,
    coupling: CouplingMap,
) -> None:
    estimate = estimate_routing_cost(
        _workload(),
        coupling,
        strategy=strategy,
    )
    routed = route_to_topology(
        _workload(),
        coupling,
        strategy=strategy,
    )
    metadata = routed.metadata["routing"]

    assert estimate.source_instruction_count == len(_workload())
    assert estimate.estimated_instruction_count == len(routed)
    assert (
        estimate.planned_inserted_swap_count == metadata["planned_inserted_swap_count"]
    )
    assert estimate.topology_gate_count == metadata["topology_gate_count"]
    assert estimate.routed_gate_count == metadata["routed_gate_count"]
    assert (
        estimate.pre_restore_logical_to_physical
        == metadata["pre_restore_logical_to_physical"]
    )
    assert estimate.final_logical_to_physical == metadata["final_logical_to_physical"]


def test_auto_strategy_selects_persistent_when_it_reduces_swaps() -> None:
    circuit = fq.Circuit(5)
    circuit.cx(0, 4).h(4).cx(0, 4)
    coupling = CouplingMap.line(5)

    selection = select_routing_strategy(circuit, coupling)
    compiled = circuit.compile(
        coupling_map=coupling,
        routing_strategy="auto",
    )
    plan = fqxp.plan_advanced(
        circuit,
        coupling_map=coupling,
        routing_strategy="auto",
    )

    assert selection.selected_strategy == "persistent_layout"
    assert compiled.to_ir().metadata["routing"]["strategy"] == "persistent_layout"
    assert (
        compiled.to_ir().metadata["routing_strategy_selection"]["selected_strategy"]
        == "persistent_layout"
    )
    assert (
        plan.summary()["routing_plan"]["strategy_selection"]["selected_strategy"]
        == "persistent_layout"
    )
    assert plan.summary()["routing_plan"]["strategy"] == "persistent_layout"
    assert torch.allclose(compiled.state(), circuit.state(), atol=1e-6)


def test_auto_strategy_tie_breaks_to_restore_after_each_gate() -> None:
    circuit = fq.Circuit(3).cx(0, 1).h(2)

    selection = select_routing_strategy(circuit, CouplingMap.line(3))

    assert selection.restore_after_each_gate.planned_inserted_swap_count == 0
    assert selection.persistent_layout.planned_inserted_swap_count == 0
    assert selection.selected_strategy == "restore_after_each_gate"


def test_the_estimate_refuses_the_planner_strategies_by_name() -> None:
    """The estimate prices two strategies; the planner is refused, not priced.

    `routing.ROUTING_STRATEGIES` names four strategies, so a caller could
    reasonably ask the estimator for any of them. Two of those names are SWAP
    planners rather than cost models, and the refusal says so instead of
    returning a number that would be an estimate of nothing.
    """

    for strategy in ("sabre", "sabre_layout"):
        with pytest.raises(ValueError, match="plan SWAPs instead of estimating them"):
            estimate_routing_cost(_workload(), CouplingMap.line(5), strategy=strategy)


def _spread_workload(n_wires: int, depth: int, seed: int) -> fq.Circuit:
    """A two-wire-heavy program on many wires, which is what routing costs."""

    rng = random.Random(seed)
    circuit = fq.Circuit(n_wires)
    for _ in range(depth // 2):
        left, right = rng.sample(range(n_wires), 2)
        circuit.cx(left, right)
        circuit.ry(rng.randrange(n_wires), theta=rng.uniform(-0.7, 0.7))
        left, right = rng.sample(range(n_wires), 2)
        circuit.cz(left, right)
    return circuit


def test_auto_strategy_cannot_reach_a_planner_and_states_the_gap() -> None:
    """The selector's candidate set is pinned, because the gap is the row's.

    The parity row `topology_aware_routing` is `partial` on the routing decision
    rather than on the router: a SABRE-class planner is present, and the
    automatic choice cannot see it. This test re-derives that from the tree so
    the reason cannot go stale in either direction -- the assertion is an
    inequality, so it still holds after any improvement to the planner, and it
    fails if the selector gains a planner strategy without the row being updated.
    """

    program = _spread_workload(n_wires=12, depth=20, seed=4)
    coupling = CouplingMap.line(12)

    selection = select_routing_strategy(program, coupling)

    # The candidate set is exactly the two estimating strategies, which is what
    # the selection record exposes as its candidate fields.
    assert set(selection.summary()["candidates"]) == {
        "restore_after_each_gate",
        "persistent_layout",
    }
    selected = selection.selected_strategy
    selected_swaps = route_to_topology(program, coupling, strategy=selected).metadata[
        "routing"
    ]["inserted_swap_count"]

    for planner in ("sabre", "sabre_layout"):
        planner_swaps = route_to_topology(program, coupling, strategy=planner).metadata[
            "routing"
        ]["inserted_swap_count"]
        assert selected_swaps > planner_swaps, (
            f"{selected} spent {selected_swaps} swaps where {planner} needed "
            f"{planner_swaps}, so this workload no longer shows the gap the "
            "parity row's reason rests on"
        )
