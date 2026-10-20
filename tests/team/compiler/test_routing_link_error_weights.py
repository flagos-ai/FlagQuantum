"""Routing that reads what a device's links cost to use.

A ``CouplingMap`` states which qubit pairs a device couples and says nothing about
how well it couples them, so the cheapest legal route and the cleanest legal route
are the same route: a hop count. ``LinkErrorRates`` is the same couplings with one
measured two-qubit error probability per link, and it makes those two questions
different questions -- a longer route over clean links can be the better route, and
a hop count cannot express that.

These tests pin the four claims the change makes:

* the route a calibrated device produces is priced in faults, and on a program
  where the hop count and the fault total disagree the emitted SWAP series changes
  while the SWAP count does not;
* the price and the plan state the same fault total, so an estimate is still a
  plan and not a bound;
* a caller who asked for a fault-priced device is never handed a plan a planner
  chose by hop count, and the refusal names the strategies that read the rate;
* the ranking that picks a strategy ranks on faults first, and on a program where
  the two rankings disagree the fault-first one is the one that runs.
"""

from __future__ import annotations

import math
import random
from typing import Any

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler import CouplingMap, LinkErrorRates
from flagquantum.compiler.directed_topology import DirectedCouplingMap
from flagquantum.compiler.routing import (
    DEPLOYABLE_ROUTING_STRATEGIES,
    LINK_ERROR_ROUTING_STRATEGIES,
    estimate_routing_cost,
    route_to_topology,
    select_routing_strategy,
)
from flagquantum.core.ir import CircuitIR

pytestmark = pytest.mark.unit

# A line with one chord: 0-1-2-3 plus 0-3. Qubits 1 and 3 are two hops apart
# whichever way a route goes, so a hop count cannot choose between the routes and
# only their price can.
_CHORDED_LINE = CouplingMap(4, ((0, 1), (1, 2), (2, 3), (0, 3)))

# The route through 1-0-3 is the one a breadth-first search finds first, because it
# visits qubit 0 before qubit 2, and it is the dear one: 0.04 to use against 0.002
# for the route through 1-2-3. Both routes cross two couplings past the gate's own,
# and both insert the same two SWAPs, so this is the program on which a hop count
# and a fault total disagree about which program to emit.
_CHORDED_CALIBRATION = {
    (0, 1): 0.02,
    (0, 3): 0.02,
    (1, 2): 0.001,
    (2, 3): 0.001,
}

# A ring of six whose links differ by a factor of thirty. Two programs on it price
# the two estimating strategies at the same SWAP count and different fault totals,
# which is what the ranking test needs.
_RING = CouplingMap.ring(6)
_RING_CALIBRATION = {
    (0, 1): 0.001,
    (1, 2): 0.001,
    (2, 3): 0.03,
    (3, 4): 0.001,
    (4, 5): 0.03,
    (0, 5): 0.01,
}


def _chorded_device() -> LinkErrorRates:
    return LinkErrorRates(_CHORDED_LINE, _CHORDED_CALIBRATION)


def _ring_device() -> LinkErrorRates:
    return LinkErrorRates(_RING, _RING_CALIBRATION)


def _two_hop_workload() -> fq.Circuit:
    """One gate across a pair that has two equally long routes and one clean one."""

    circuit = fq.Circuit(4)
    circuit.h(1)
    circuit.cx(1, 3)
    return circuit


def _ring_workload() -> fq.Circuit:
    """Two gates whose cheap routes are on opposite sides of the ring."""

    circuit = fq.Circuit(6)
    circuit.h(0)
    circuit.cx(0, 4)
    circuit.h(1)
    circuit.cx(1, 3)
    return circuit


def _fault_total(ir: CircuitIR, device: LinkErrorRates) -> float:
    """Sum the declared link error of every two-qubit operation a program runs.

    Recomputed here from the emitted program rather than read from the plan, so the
    total the router reports is checked against the program it handed back.
    """

    total = 0.0
    for instruction in ir:
        if len(instruction.wires) != 2:
            continue
        if instruction.metadata.get("is_channel"):
            continue
        total += device.error(*instruction.wires)
    return total


def test_a_hop_count_cannot_choose_the_route_a_fault_total_can() -> None:
    """The same two SWAPs on different links, chosen by price rather than length.

    The program is a single ``cx(1, 3)`` on the chorded line. Both routes are two
    hops and both cost one forward SWAP and one restoring SWAP, so every count the
    router reported before this change is identical between the two programs. What
    differs is which couplings the SWAPs and the gate run on: the breadth-first
    route crosses two links priced at 0.02, the calibrated route crosses two links
    priced at 0.001.
    """

    workload = _two_hop_workload()
    device = _chorded_device()

    hop_routed = route_to_topology(workload, _CHORDED_LINE)
    calibrated = route_to_topology(workload, device)

    assert [(instruction.name, instruction.wires) for instruction in hop_routed] == [
        ("h", (1,)),
        ("swap", (1, 0)),
        ("cx", (0, 3)),
        ("swap", (1, 0)),
    ]
    assert [(instruction.name, instruction.wires) for instruction in calibrated] == [
        ("h", (1,)),
        ("swap", (1, 2)),
        ("cx", (2, 3)),
        ("swap", (1, 2)),
    ]

    hop_plan = hop_routed.metadata["routing"]
    calibrated_plan = calibrated.metadata["routing"]
    # Everything a hop count reports is unchanged, which is the point: a SWAP
    # count cannot tell these two programs apart.
    assert (
        hop_plan["inserted_swap_count"] == calibrated_plan["inserted_swap_count"] == 2
    )
    assert hop_plan["topology_gate_count"] == calibrated_plan["topology_gate_count"]
    # The fault total can, and it says the calibrated route costs fifteen times
    # less: 2 * 0.001 for the two SWAPs plus 0.001 for the gate, against the 0.06
    # the hop-count route would have paid on links priced at 0.02.
    assert hop_plan["expected_link_faults"] is None
    assert calibrated_plan["expected_link_faults"] == pytest.approx(0.003)
    assert device.route_error(_CHORDED_LINE.shortest_path(1, 3)) == pytest.approx(0.04)


def test_the_fault_total_the_router_reports_is_the_total_the_program_pays() -> None:
    """The reported total is recomputed from the emitted program, not trusted."""

    device = _chorded_device()
    for strategy in LINK_ERROR_ROUTING_STRATEGIES:
        routed = route_to_topology(_two_hop_workload(), device, strategy=strategy)
        reported = routed.metadata["routing"]["expected_link_faults"]
        assert reported == pytest.approx(_fault_total(routed, device))


def test_a_calibrated_route_computes_the_state_the_source_program_computes() -> None:
    """The program a calibrated device is routed to still computes what was asked.

    A weight that changed the route without preserving the state would be a
    cheaper wrong answer, so the routed program is executed and compared against
    the source. Both strategies and the automatic choice are checked, because the
    automatic choice is the path a caller reaches by not naming a strategy.
    """

    workload = _two_hop_workload()
    device = _chorded_device()
    expected = workload.state()

    for strategy in (*LINK_ERROR_ROUTING_STRATEGIES, "auto"):
        compiled = workload.compile(coupling_map=device, routing_strategy=strategy)
        assert torch.allclose(compiled.state(), expected, atol=1e-6), strategy


def test_the_estimate_and_the_plan_price_the_same_program() -> None:
    """An estimate is a plan: the two fault totals agree on every seeded case."""

    rng = random.Random(20261007)
    disagreements: list[str] = []
    for trial in range(12):
        calibration = {
            edge: rng.choice([0.0, 0.0005, 0.001, 0.01, 0.05]) for edge in _RING.edges
        }
        device = LinkErrorRates(_RING, calibration)
        pairs = [(rng.randrange(6), rng.randrange(6)) for _ in range(3)]
        circuit = fq.Circuit(6)
        for control, target in pairs:
            if control != target:
                circuit.h(control)
                circuit.cx(control, target)
        if not len(circuit):
            continue
        for strategy in LINK_ERROR_ROUTING_STRATEGIES:
            label = f"trial {trial} {strategy} {pairs}"
            estimate = estimate_routing_cost(circuit, device, strategy=strategy)
            routed = route_to_topology(circuit, device, strategy=strategy)
            plan = routed.metadata["routing"]
            if estimate.planned_inserted_swap_count != plan["inserted_swap_count"]:
                disagreements.append(f"{label}: swap counts differ")
            if plan["expected_link_faults"] is None:
                disagreements.append(f"{label}: plan states no fault total")
            elif not math.isclose(
                estimate.expected_link_faults,
                plan["expected_link_faults"],
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                disagreements.append(
                    f"{label}: estimate {estimate.expected_link_faults!r} priced "
                    f"{plan['expected_link_faults']!r}"
                )
            # The third opinion is the program itself: the emitted operations,
            # priced from the same calibration, are what a device would run.
            emitted = _fault_total(routed, device)
            if not math.isclose(
                estimate.expected_link_faults, emitted, rel_tol=0.0, abs_tol=1e-12
            ):
                disagreements.append(
                    f"{label}: estimate {estimate.expected_link_faults!r} against "
                    f"the emitted program's {emitted!r}"
                )
    assert not disagreements, disagreements


def test_a_planner_that_never_reads_the_calibration_is_refused() -> None:
    """A hop-count planner is refused rather than left to ignore the rate.

    Pricing the device and routing it are separate entry points, so both refuse:
    an estimate taken by a planner that never read the rate would be quoted to a
    caller as the price of the program it would get.
    """

    device = _chorded_device()
    workload = _two_hop_workload()

    for entry, call in (
        (
            "estimate_routing_cost",
            lambda name: estimate_routing_cost(workload, device, strategy=name),
        ),
        (
            "route_to_topology",
            lambda name: route_to_topology(workload, device, strategy=name),
        ),
    ):
        for strategy in ("sabre", "sabre_layout"):
            with pytest.raises(ValueError) as excinfo:
                call(strategy)
            message = str(excinfo.value)
            assert entry in message, (entry, strategy, message)
            assert repr(strategy) in message, (entry, strategy, message)
            # The refusal names the repair, because the caller chooses it.
            assert "'restore_after_each_gate'" in message
            assert "'persistent_layout'" in message


def test_the_automatic_choice_ranks_by_faults_before_it_ranks_by_swaps() -> None:
    """The two rankings disagree here, and the fault-first one is the one that runs.

    Both estimating strategies insert eight SWAPs on this program, so a ranking
    that read only counts would tie and fall back to candidate order, which puts
    ``restore_after_each_gate`` first. In faults that strategy costs 0.097 against
    the other's 0.068, because keeping the layout lets the second gate reuse the
    cheap links the first gate paid to reach -- a difference a SWAP count cannot
    see.
    """

    workload = _ring_workload()
    device = _ring_device()
    selection = select_routing_strategy(workload, device)

    assert selection.selected_strategy == "persistent_layout"
    restore = selection.candidates["restore_after_each_gate"]
    persistent = selection.candidates["persistent_layout"]
    assert (
        restore.planned_inserted_swap_count
        == (persistent.planned_inserted_swap_count)
        == 8
    )
    assert restore.estimated_instruction_count == persistent.estimated_instruction_count
    assert restore.expected_link_faults == pytest.approx(0.097)
    assert persistent.expected_link_faults == pytest.approx(0.068)

    # The ranking the uncalibrated device uses would have chosen the other plan.
    count_ranked = min(
        LINK_ERROR_ROUTING_STRATEGIES,
        key=lambda name: (
            selection.candidates[name].planned_inserted_swap_count,
            selection.candidates[name].estimated_instruction_count,
        ),
    )
    assert count_ranked == "restore_after_each_gate"

    summary = selection.summary()
    assert summary["objective"] == (
        "minimize_expected_link_faults_then_planned_inserted_swaps_" "then_instructions"
    )
    assert summary["selected_strategy"] == "persistent_layout"
    assert summary["candidates"]["persistent_layout"]["expected_link_faults"] == (
        pytest.approx(0.068)
    )


def test_the_calibrated_choice_prices_only_the_strategies_that_read_the_rate() -> None:
    """A strategy that never read the rate is not a candidate, so it is not ranked."""

    selection = select_routing_strategy(_ring_workload(), _ring_device())

    assert tuple(selection.candidates) == LINK_ERROR_ROUTING_STRATEGIES
    for name in ("sabre", "sabre_layout"):
        assert name not in selection.candidates


def test_an_uncalibrated_device_keeps_the_ranking_and_the_candidate_set() -> None:
    """A device that states only its couplings is unaffected by the new weight."""

    selection = select_routing_strategy(_ring_workload(), _RING)

    assert tuple(selection.candidates) == DEPLOYABLE_ROUTING_STRATEGIES
    assert selection.summary()["objective"] == (
        "minimize_planned_inserted_swaps_then_instructions"
    )
    for candidate in selection.candidates.values():
        assert candidate.expected_link_faults is None


def test_a_route_is_a_property_of_the_qubit_pair_not_of_its_direction() -> None:
    """The cheapest route from a to b is the cheapest route from b to a, reversed.

    Both are answers about the same two qubits on the same undirected device, so an
    answer that depended on which one was asked for first would make the route a
    property of the question. The second query also reads a warm cache, which is
    why the reverse direction is asked twice.
    """

    rng = random.Random(4242)
    for trial in range(8):
        calibration = {
            edge: round(rng.choice([0.0, 0.001, 0.004, 0.02]), 4)
            for edge in _RING.edges
        }
        device = LinkErrorRates(_RING, calibration)
        for start in range(6):
            for goal in range(start + 1, 6):
                forward = device.lowest_error_path(start, goal)
                back = device.lowest_error_path(goal, start)
                assert back == tuple(reversed(forward)), (trial, start, goal)
                assert device.route_error(forward) == pytest.approx(
                    device.route_error(back)
                )
                assert device.lowest_error_path(goal, start) == back


def test_two_equally_clean_routes_are_not_separated_by_a_summation_order() -> None:
    """The answer is a property of the device, not of the order the terms added in.

    This device has two four-hop routes from qubit 0 to qubit 4 whose links carry
    the same four rates in mirror order: 0.4, 0.3, 0.2, 0.1 one way and 0.1, 0.2,
    0.3, 0.4 the other. The two routes are therefore exactly as clean as each
    other -- both leave the route with the same survival probability -- and the
    same four numbers, added in opposite orders, sum to values two units in the
    last place apart. Compared as raw sums, the lower route wins by that one bit;
    compared at the precision a calibration is stated to, the two are equal and
    the hop count and then the qubit sequence decide, which picks the upper route.

    Without that comparison the route a caller gets is a rounding accident that a
    refactor of the search could silently reverse, so this pins the arithmetic
    rather than the plan: the two route totals are checked to be different as raw
    sums and equal after rounding, which is what makes the assertion load-bearing.
    """

    device = LinkErrorRates(
        CouplingMap(
            8, ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (4, 7))
        ),
        {
            (0, 1): 0.4,
            (1, 2): 0.3,
            (2, 3): 0.2,
            (3, 4): 0.1,
            (0, 5): 0.1,
            (5, 6): 0.2,
            (6, 7): 0.3,
            (4, 7): 0.4,
        },
    )
    upper = (0, 1, 2, 3, 4)
    lower = (0, 5, 6, 7, 4)

    def raw_cost(route: tuple[int, ...]) -> float:
        total = 0.0
        for index in range(len(route) - 1):
            total += -math.log1p(-device.error(route[index], route[index + 1]))
        return total

    assert raw_cost(upper) != raw_cost(lower)
    assert raw_cost(lower) < raw_cost(upper)
    # Both routes are the same four hops and the same four rates, so neither is
    # cheaper once the comparison is made at the precision the rates carry.
    assert round(raw_cost(upper), 12) == round(raw_cost(lower), 12)
    assert device.route_error(upper) == pytest.approx(device.route_error(lower))
    # The tie therefore falls to the hop count, which ties too, and then to the
    # qubit sequence, which picks the route through qubit 1 in either direction.
    assert device.lowest_error_path(0, 4) == upper
    assert device.lowest_error_path(4, 0) == tuple(reversed(upper))


def test_an_unreachable_route_is_refused_rather_than_priced() -> None:
    """A pair the device cannot join has no cheapest route to report."""

    device = LinkErrorRates(CouplingMap(4, ((0, 1),)), {(0, 1): 0.01})

    with pytest.raises(ValueError) as excinfo:
        device.lowest_error_path(0, 3)

    assert "No coupling path between qubits 0 and 3" in str(excinfo.value)


def test_a_qubit_the_calibration_does_not_cover_is_refused() -> None:
    """A route endpoint has to be a qubit on the device that was calibrated."""

    device = _chorded_device()

    with pytest.raises(ValueError) as excinfo:
        device.lowest_error_path(0, 9)

    assert "not on this device of 4 qubits" in str(excinfo.value)


@pytest.mark.parametrize(
    ("calibration", "expected"),
    [
        pytest.param(
            {(0, 1): 0.01, (0, 3): 0.01, (1, 2): 0.01},
            "the device declares link (2, 3) with no error rate",
            id="missing-rate",
        ),
        pytest.param(
            {(0, 1): 0.01, (0, 3): 0.01},
            "the device declares link (1, 2) with no error rate",
            id="missing-rate-second-link",
        ),
        pytest.param(
            {
                (0, 1): 0.01,
                (0, 3): 0.01,
                (1, 2): 0.01,
                (2, 3): 0.01,
                (0, 2): 0.01,
            },
            "link (0, 2) is not a coupling the device declares",
            id="rate-on-an-undeclared-link",
        ),
        pytest.param(
            {(0, 1): -0.01, (0, 3): 0.01, (1, 2): 0.01, (2, 3): 0.01},
            "must be finite and in [0, 1)",
            id="negative-rate",
        ),
        pytest.param(
            {(0, 1): 1.0, (0, 3): 0.01, (1, 2): 0.01, (2, 3): 0.01},
            "must be finite and in [0, 1)",
            id="rate-of-one",
        ),
        pytest.param(
            {(0, 1): float("nan"), (0, 3): 0.01, (1, 2): 0.01, (2, 3): 0.01},
            "must be finite and in [0, 1)",
            id="not-a-number",
        ),
        pytest.param(
            {(0, 1): float("inf"), (0, 3): 0.01, (1, 2): 0.01, (2, 3): 0.01},
            "must be finite and in [0, 1)",
            id="infinite-rate",
        ),
        pytest.param(
            {(0, 0): 0.01, (0, 1): 0.01, (0, 3): 0.01, (1, 2): 0.01, (2, 3): 0.01},
            "a link joins two different qubits, not one",
            id="self-link",
        ),
        pytest.param(
            {(0, 1): True, (0, 3): 0.01, (1, 2): 0.01, (2, 3): 0.01},
            "must be a number",
            id="boolean-rate",
        ),
    ],
)
def test_a_calibration_the_record_cannot_use_is_refused(
    calibration: dict[tuple[int, int], Any],
    expected: str,
) -> None:
    """Every refusal names the link it is about, so the repair is a lookup.

    A link priced at nothing looks like a free route, so a missing rate cannot
    default to zero and is refused instead; a rate on a link the device does not
    declare describes a device nobody has, so it is refused too.
    """

    with pytest.raises(ValueError) as excinfo:
        LinkErrorRates(_CHORDED_LINE, calibration)

    assert expected in str(excinfo.value)


def test_a_calibration_of_an_ordered_device_is_refused() -> None:
    """An ordered device prices a wrong-way gate in SWAPs, not in link error.

    One rate per link would leave the direction repair unpriced, which would report
    a program cheaper than the one the device has to run, so the pairing is refused
    rather than approximated.
    """

    ordered = DirectedCouplingMap(2, ((0, 1),))

    with pytest.raises(ValueError) as excinfo:
        LinkErrorRates(ordered, {(0, 1): 0.01})  # type: ignore[arg-type]

    assert "describe an undirected device" in str(excinfo.value)


def test_a_calibration_of_something_that_is_not_a_coupling_map_is_refused() -> None:
    """A device record is not an edge list: the rates are keyed by declared links."""

    with pytest.raises(TypeError) as excinfo:
        LinkErrorRates([(0, 1)], {(0, 1): 0.01})  # type: ignore[arg-type]

    assert "require a CouplingMap" in str(excinfo.value)


@pytest.mark.parametrize("capacity", [1, -1])
def test_a_path_cache_too_small_to_hold_a_route_is_refused(capacity: int) -> None:
    """One slot cannot hold the two orientations of one route, so it is not a cache."""

    with pytest.raises(ValueError) as excinfo:
        LinkErrorRates(
            _CHORDED_LINE, _CHORDED_CALIBRATION, path_cache_capacity=capacity
        )

    assert "zero or at least two" in str(excinfo.value)
