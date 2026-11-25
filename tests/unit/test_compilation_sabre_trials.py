"""Bounded routing search preserves full operators and delivered-layout costs."""

from __future__ import annotations

import random
from unittest.mock import patch

import pytest

from flagquantum.compiler import CouplingMap, route_to_topology
from flagquantum.compiler.topology import routing, routing_strategy, sabre
from flagquantum.compiler.topology.directed_topology import DirectedCouplingMap
from flagquantum.compiler.topology.routing_cost import _topology_cost_model
from flagquantum.core.ir import CircuitIR, Instruction

pytestmark = pytest.mark.unit


def _program(width: int, seed: int) -> CircuitIR:
    rng = random.Random(seed)
    return CircuitIR(
        width,
        tuple(
            Instruction("cx", tuple(rng.sample(range(width), 2)))
            for _ in range(4 * width)
        ),
    )


def _linear_operator(
    program: CircuitIR, initial_layout: tuple[int, ...] | None = None
) -> tuple[int, ...]:
    # Each output bit is an exact GF(2) linear combination of all input bits.
    rows = [1 << q for q in range(program.n_wires)]
    if initial_layout is not None:
        for logical, physical in enumerate(initial_layout):
            rows[physical] = 1 << logical
    for instruction in program:
        left, right = instruction.wires
        if instruction.name == "cx":
            rows[right] ^= rows[left]
        else:
            assert instruction.name == "swap"
            rows[left], rows[right] = rows[right], rows[left]
    return tuple(rows)


@pytest.mark.parametrize("width", [6, 8, 12, 16])
@pytest.mark.parametrize("seed", range(4))
def test_trials_retain_baseline_cost_and_complete_operator(
    width: int, seed: int
) -> None:
    source = _program(width, seed)
    coupling = CouplingMap.grid(2, width // 2)
    baseline = sabre._plan_sabre_trial(source, coupling)
    assert baseline is not None
    before = random.getstate()
    selected = sabre.plan_sabre_swaps(source, coupling)
    assert random.getstate() == before
    assert selected == sabre.plan_sabre_swaps(source, coupling)
    assert sabre._restored_swap_count(selected, coupling) <= sabre._restored_swap_count(
        baseline, coupling
    )
    routed = route_to_topology(source, coupling, strategy="sabre")
    assert _linear_operator(routed) == _linear_operator(source)
    assert all(coupling.has_edge(*op.wires) for op in routed)
    assert routed.metadata["routing"]["mapping_restored"] is True


@pytest.mark.parametrize("seed", range(6))
def test_pruning_never_discards_a_trial_that_beats_its_limit(seed: int) -> None:
    source = _program(8, seed)
    coupling = CouplingMap.line(8)
    unbounded = sabre._plan_sabre_trial(source, coupling, tie_rng=random.Random(seed))
    assert unbounded is not None
    total = sabre._restored_swap_count(unbounded, coupling)
    for limit in (0, 1, total // 2, total, total + 1):
        bounded = sabre._plan_sabre_trial(
            source, coupling, tie_rng=random.Random(seed), committed_swap_limit=limit
        )
        if bounded is None:
            assert total >= limit
        else:
            assert bounded == unbounded
        if total < limit:
            assert bounded == unbounded


def test_search_uses_at_most_five_trials_and_skips_already_local_program() -> None:
    coupling = CouplingMap.line(6)
    with patch.object(
        sabre, "_plan_sabre_trial", wraps=sabre._plan_sabre_trial
    ) as trial:
        sabre.plan_sabre_swaps(CircuitIR(6, (Instruction("cx", (0, 1)),)), coupling)
        assert trial.call_count == 1
        trial.reset_mock()
        sabre.plan_sabre_swaps(_program(6, 3), coupling)
        assert trial.call_count == 5
        assert all(
            call.kwargs["committed_swap_limit"] > 0 for call in trial.call_args_list[1:]
        )


@pytest.mark.parametrize("seed", range(6))
def test_ordered_device_ranks_trials_with_orientation_and_restoration(
    seed: int,
) -> None:
    source = _program(6, seed)
    device = DirectedCouplingMap(6, [(q, q + 1) for q in range(5)])
    topology = _topology_cost_model(device, 6, "routing trial test")
    coupling = topology.weak_coupling

    def cost(plan: sabre.SabrePlan) -> int:
        return sabre._restored_swap_count(
            plan, coupling
        ) + 2 * routing._count_sabre_orientation_swaps(
            source, topology, plan.placements
        )

    candidates = [sabre._plan_sabre_trial(source, coupling)]
    candidates.extend(
        sabre._plan_sabre_trial(source, coupling, tie_rng=random.Random(trial))
        for trial in range(3)
    )
    candidates.append(sabre._plan_sabre_trial(source, coupling, layer_lookahead=True))
    assert all(candidate is not None for candidate in candidates)
    minimum = min(cost(candidate) for candidate in candidates if candidate is not None)
    selected = routing._plan_sabre_for_topology(source, topology, tuple(range(6)))
    assert cost(selected) == minimum
    estimate = routing.estimate_routing_cost(source, device, strategy="sabre")
    routed = route_to_topology(source, device, strategy="sabre")
    assert estimate.planned_inserted_swap_count == minimum
    assert routed.metadata["routing"]["inserted_swap_count"] == minimum
    assert _linear_operator(routed) == _linear_operator(source)


@pytest.mark.parametrize("seed", (0, 3))
def test_materialization_reuses_the_selected_layout_search(seed: int) -> None:
    source = _program(8, seed)
    topology = _topology_cost_model(CouplingMap.grid(2, 4), 8, "trial reuse")
    with patch.object(
        routing_strategy, "plan_sabre_layout", wraps=sabre.plan_sabre_layout
    ) as search:
        estimate, plan = routing._estimate_routing_cost(
            source, topology, strategy="sabre_layout"
        )
        assert plan is not None
        routed = routing._route_sabre(
            source,
            topology,
            strategy="sabre_layout",
            path_cache_before=topology.path_cache_info(),
            plan=plan,
        )
    assert search.call_count == 1
    assert (
        routed.metadata["routing"]["inserted_swap_count"]
        == estimate.planned_inserted_swap_count
    )
    # This caller-selected strategy expects its declared physical input layout.
    assert _linear_operator(
        routed, plan.initial_logical_to_physical
    ) == _linear_operator(source)
    if seed == 0:
        assert plan.initial_logical_to_physical != tuple(range(source.n_wires))
        assert _linear_operator(routed) != _linear_operator(source)
    else:
        assert plan.initial_logical_to_physical == tuple(range(source.n_wires))


def test_completed_trial_stream_scores_each_plan_once_and_preserves_ties() -> None:
    source = _program(6, 3)
    coupling = CouplingMap.line(6)
    baseline = sabre._plan_sabre_trial(source, coupling)
    assert baseline is not None
    scored: list[sabre.SabrePlan] = []

    def score(plan: sabre.SabrePlan) -> int:
        scored.append(plan)
        return sabre._restored_swap_count(plan, coupling)

    trials = list(sabre._sabre_trial_costs(source, coupling, baseline, cost=score))
    assert scored == [plan for plan, _ in trials]
    assert trials[0][0] == baseline
    expected = min(trials, key=lambda item: item[1])[0]
    assert sabre._select_sabre_trials(source, coupling, baseline) == expected
    # Equal-cost alternatives must not displace the earlier incumbent.
    assert (
        sabre._select_sabre_trials(source, coupling, baseline, cost=lambda plan: 10000)
        == baseline
    )


def test_local_selection_does_not_call_a_custom_cost_or_create_a_stream() -> None:
    source = CircuitIR(6, (Instruction("cx", (0, 1)),))
    coupling = CouplingMap.line(6)
    baseline = sabre._plan_sabre_trial(source, coupling)
    assert baseline is not None
    with patch.object(sabre, "_sabre_trial_costs") as stream:
        assert (
            sabre._select_sabre_trials(
                source,
                coupling,
                baseline,
                cost=lambda plan: pytest.fail("Local fast path must not score trials"),
            )
            == baseline
        )
    stream.assert_not_called()
