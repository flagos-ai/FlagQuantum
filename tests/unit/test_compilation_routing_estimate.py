import pytest
import torch

import flagquantum as fq
import flagquantum.compilation.planner as fqxp
from flagquantum.compilation.routing import (
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
