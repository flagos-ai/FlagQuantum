import random

import pytest
import torch

import flagquantum as fq
import flagquantum.runtime.planner as fqxp
from flagquantum.compiler.routing import (
    DEPLOYABLE_ROUTING_STRATEGIES,
    ROUTING_STRATEGIES,
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


@pytest.mark.parametrize("strategy", ROUTING_STRATEGIES)
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
    """A price is the plan a strategy produces, not a bound on it.

    Every name the router accepts is priced, including the two planners, whose
    price is the plan itself. The two SABRE strategies reorder operations and
    ``persistent_layout`` counts a two-wire operation as unrouted only where the
    device lacks the coupling, so the counting fields have to agree too.
    """

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
    assert estimate.planned_inserted_swap_count == metadata["inserted_swap_count"]
    assert (
        estimate.planned_inserted_swap_count == metadata["planned_inserted_swap_count"]
    )
    assert estimate.topology_gate_count == metadata["topology_gate_count"]
    assert estimate.routed_gate_count == metadata["routed_gate_count"]
    assert estimate.skipped_channel_count == metadata["skipped_channel_count"]
    assert (
        estimate.pre_restore_logical_to_physical
        == metadata["pre_restore_logical_to_physical"]
    ), "pre_restore_logical_to_physical"
    assert estimate.final_logical_to_physical == metadata["final_logical_to_physical"]


def test_the_estimate_prices_every_strategy_the_router_accepts() -> None:
    """The candidate set is the router's whole vocabulary, planners included.

    The estimate used to refuse ``sabre`` and ``sabre_layout`` because a planner
    does not estimate its SWAPs, which left the automatic choice unable to compare
    a planner against an estimate. Pricing a planner is planning it, which is what
    makes the two lists equal rather than a pinned pair of names.
    """

    priced = {
        strategy: estimate_routing_cost(
            _workload(), CouplingMap.line(5), strategy=strategy
        )
        for strategy in ROUTING_STRATEGIES
    }

    assert set(priced) == set(ROUTING_STRATEGIES)
    assert (
        priced["sabre_layout"].planned_inserted_swap_count
        < priced["persistent_layout"].planned_inserted_swap_count
    )


def test_the_estimate_refuses_a_name_the_router_does_not_accept() -> None:
    with pytest.raises(ValueError, match="routing strategy must be one of"):
        estimate_routing_cost(
            _workload(), CouplingMap.line(5), strategy="restore_after_each_swap"
        )


def test_auto_strategy_selects_the_cheapest_deployable_strategy() -> None:
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

    assert set(selection.candidates) == set(DEPLOYABLE_ROUTING_STRATEGIES)
    assert selection.candidates[
        selection.selected_strategy
    ].planned_inserted_swap_count == min(
        estimate.planned_inserted_swap_count
        for estimate in selection.candidates.values()
    )
    # ``persistent_layout`` and ``sabre`` both price at 6 SWAPs here and the
    # candidate order puts the estimate first, so an equal price does not pay for
    # a plan.
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


def test_the_automatic_choice_reaches_the_sabre_planner() -> None:
    """The gap the parity row named: ``auto`` could not select a planner at all.

    The same selection is driven through ``compile`` and through the deployment
    package builder, because the automatic choice must not hand one of them a
    program the other refuses.
    """

    circuit = _spread_workload(n_wires=12, depth=20, seed=4)
    coupling = CouplingMap.line(12)

    selection = select_routing_strategy(circuit, coupling)

    assert selection.selected_strategy == "sabre"
    assert (
        selection.candidates["sabre"].planned_inserted_swap_count
        < selection.candidates["persistent_layout"].planned_inserted_swap_count
    )
    compiled = circuit.compile(coupling_map=coupling, routing_strategy="auto")
    routing = compiled.to_ir().metadata["routing"]
    assert routing["strategy"] == "sabre"
    assert routing["initial_logical_to_physical"] == tuple(range(circuit.n_qubits))
    assert routing["final_logical_to_physical"] == tuple(range(circuit.n_qubits))
    assert torch.allclose(compiled.state(), circuit.state(), atol=1e-6)


def test_auto_strategy_tie_breaks_to_the_estimating_strategy() -> None:
    """Equal prices must not re-plan, so the estimates keep their place in order."""

    circuit = fq.Circuit(3).cx(0, 1).h(2)

    selection = select_routing_strategy(circuit, CouplingMap.line(3))

    assert {
        name: estimate.planned_inserted_swap_count
        for name, estimate in selection.candidates.items()
    } == dict.fromkeys(DEPLOYABLE_ROUTING_STRATEGIES, 0)
    assert selection.selected_strategy == "restore_after_each_gate"


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


def test_the_selector_ranks_the_whole_candidate_set() -> None:
    """The automatic choice is the minimum of the prices it reports.

    The parity row `topology_aware_routing` was `partial` because the automatic
    choice priced two strategies and so could not reach the SABRE planner. The
    assertion is the invariant rather than a pinned winner, so it still holds
    after any improvement to any strategy, and it fails if a candidate is priced
    but not considered, or if the chosen plan is not the plan it was priced at.
    """

    program = _spread_workload(n_wires=12, depth=20, seed=4)
    coupling = CouplingMap.line(12)

    selection = select_routing_strategy(program, coupling)
    candidates = dict(selection.candidates)

    assert set(candidates) == set(DEPLOYABLE_ROUTING_STRATEGIES)
    chosen = candidates[selection.selected_strategy]
    assert chosen.planned_inserted_swap_count == min(
        estimate.planned_inserted_swap_count for estimate in candidates.values()
    )
    # The record is written into compile metadata as the evidence of the choice,
    # so the view of what was chosen over is not a dict a caller can edit after
    # the fact.
    with pytest.raises(TypeError):
        selection.candidates["sabre"] = chosen  # type: ignore[index]
    inserted = route_to_topology(
        program, coupling, strategy=selection.selected_strategy
    ).metadata["routing"]["inserted_swap_count"]
    assert inserted == chosen.planned_inserted_swap_count

    for strategy in DEPLOYABLE_ROUTING_STRATEGIES:
        materialized = route_to_topology(program, coupling, strategy=strategy).metadata[
            "routing"
        ]["inserted_swap_count"]
        assert inserted <= materialized, (
            f"auto chose {selection.selected_strategy} at {inserted} swaps while "
            f"{strategy} used {materialized}, so the selector is not ranking the "
            "candidate set it reports"
        )


def test_the_automatic_choice_is_always_accepted_by_deployment() -> None:
    """A widening of the candidate set must not make ``auto`` undeployable.

    `sabre_layout` is the one strategy outside the set, and it is outside for a
    structural reason rather than a ranking: it starts from a searched layout, so
    its ``initial_logical_to_physical`` is not the identity that a deployment
    artifact has to state. This asserts the separation on the workload that
    widening was measured on, and it fails if a strategy is added to the
    candidate set without the deployment contract accepting its plan.
    """

    from flagquantum.deployment.routing_evidence import (
        DeploymentRoutingEvidenceError,
        validate_deployment_routing_plan,
    )

    program = _spread_workload(n_wires=12, depth=20, seed=4)
    coupling = CouplingMap.line(12)
    selection = select_routing_strategy(program, coupling)
    assert "sabre_layout" not in DEPLOYABLE_ROUTING_STRATEGIES
    assert set(DEPLOYABLE_ROUTING_STRATEGIES) < set(ROUTING_STRATEGIES)

    for strategy in DEPLOYABLE_ROUTING_STRATEGIES:
        plan = dict(
            route_to_topology(program, coupling, strategy=strategy).metadata["routing"]
        )
        assert validate_deployment_routing_plan(
            plan, n_wires=program.n_qubits, coupling_map=coupling
        )

    undeployable = dict(
        route_to_topology(program, coupling, strategy="sabre_layout").metadata[
            "routing"
        ]
    )
    undeployable.pop("strategy_selection", None)
    with pytest.raises(
        DeploymentRoutingEvidenceError, match="unsupported routing strategy"
    ):
        validate_deployment_routing_plan(
            undeployable, n_wires=program.n_qubits, coupling_map=coupling
        )
    assert selection.selected_strategy in DEPLOYABLE_ROUTING_STRATEGIES


def test_the_three_entries_read_a_directed_device_and_price_its_ordering() -> None:
    """An ordered device is priced, and the price states what the ordering cost.

    This replaces an earlier test that pinned the opposite behaviour: all three
    entries used to refuse a ``DirectedCouplingMap`` and named
    ``legalize_circuit_topology`` as the entry that read direction. That refusal
    was the row's own recorded gap -- the routers priced hop distance on the
    undirected graph alone, so direction was a fact only a later legalization pass
    paid for. Direction is now a term in this cost model: a placed gate whose
    operands run against the direction its link is declared in is not executable
    where it stands, and the router repairs it with a SWAP, so the repair is in
    ``planned_inserted_swap_count`` and ``direction_swap_count`` states how much of
    that total the ordering forced.

    The refusal-by-name property the old test protected is kept below for an
    argument that really is opaque, because the reason for naming the accepted
    forms was the message, not the rejected type.
    """

    from flagquantum.compiler.directed_topology import DirectedCouplingMap
    from flagquantum.compiler.operand_semantics import _TWO_WIRE_OPERAND_SYMMETRY

    program = _workload()
    edges = tuple(CouplingMap.line(5).edges)
    ordered = DirectedCouplingMap(5, edges)
    undirected = CouplingMap(5, edges)
    ordering_is_free_somewhere = False

    for strategy in ROUTING_STRATEGIES:
        directed_price = estimate_routing_cost(program, ordered, strategy=strategy)
        undirected_price = estimate_routing_cost(program, undirected, strategy=strategy)
        assert directed_price.direction_semantics == "directed_cx"
        assert undirected_price.direction_semantics == "logical_wire_order_preserved"
        # An undirected device cannot produce an ordering swap at all, so a
        # nonzero count there would be a price for a repair it never makes.
        assert undirected_price.direction_swap_count == 0
        # The count is a price, not an annotation: every repair it names is two
        # SWAPs inside the total. It is not a claim that the ordered device is
        # dearer overall -- the ordering can route a planner onto a cheaper
        # mapping, and ``grid-3x4 depth-20`` under ``persistent_layout`` is a
        # measured case where the ordered device is cheaper (46 against 48).
        assert 2 * directed_price.direction_swap_count <= (
            directed_price.planned_inserted_swap_count
        )
        if directed_price.direction_swap_count > 0:
            ordering_is_free_somewhere = True

        directed_program = route_to_topology(program, ordered, strategy=strategy)
        assert directed_program.metadata["routing"]["direction_swap_count"] == (
            directed_price.direction_swap_count
        )
        assert directed_program.metadata["routing"]["inserted_swap_count"] == (
            directed_price.planned_inserted_swap_count
        )
        for instruction in directed_program:
            if len(instruction.wires) != 2:
                continue
            # A symmetric opcode only needs the link; a control/target opcode has
            # to run the way the device declares that link.
            assert ordered.has_weak_edge(*instruction.wires)
            if _TWO_WIRE_OPERAND_SYMMETRY.get(instruction.name) is False:
                assert ordered.has_edge(*instruction.wires)

    # The workload and the device do force at least one repair somewhere, so the
    # loop above is not silently asserting nothing.
    assert ordering_is_free_somewhere

    selection = select_routing_strategy(program, ordered)
    assert selection.selected_strategy in DEPLOYABLE_ROUTING_STRATEGIES
    for candidate in selection.candidates.values():
        assert candidate.direction_semantics == "directed_cx"

    for entry, call in (
        ("estimate_routing_cost", lambda: estimate_routing_cost(program, 3.5)),
        ("select_routing_strategy", lambda: select_routing_strategy(program, 3.5)),
        ("route_to_topology", lambda: route_to_topology(program, 3.5)),
    ):
        with pytest.raises(TypeError) as refusal:
            call()
        message = str(refusal.value)
        assert entry in message
        assert "CouplingMap" in message
        assert "DirectedCouplingMap" in message
        # The old failure described the iteration, not the argument.
        assert "not iterable" not in message


def test_an_edge_sequence_and_a_coupling_map_route_the_same_program() -> None:
    """The coercion accepts both documented forms and decides identically.

    The price is compared whole, because it is a plan. The routed program is
    compared by its instructions and by its routing plan minus ``path_cache``:
    that entry is a delta against the cache the map instance had already
    accumulated, so it reads differently for a map the caller passed in and for
    the map the coercion built, without either routing having differed.
    """

    program = _workload()
    edges = ((0, 1), (1, 2), (2, 3), (3, 4))
    coupling = CouplingMap(5, edges)

    assert estimate_routing_cost(
        program, coupling, strategy="sabre"
    ) == estimate_routing_cost(program, edges, strategy="sabre")
    assert select_routing_strategy(program, coupling) == select_routing_strategy(
        program, edges
    )

    from_map = route_to_topology(program, coupling)
    from_edges = route_to_topology(program, edges)
    assert from_map.instructions == from_edges.instructions
    plan = from_map.metadata["routing"]
    other = from_edges.metadata["routing"]
    assert {key: value for key, value in plan.items() if key != "path_cache"} == {
        key: value for key, value in other.items() if key != "path_cache"
    }
    assert plan["strategy"] == other["strategy"]
