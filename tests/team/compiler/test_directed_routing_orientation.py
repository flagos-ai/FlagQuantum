"""Routing that reads the direction of a device's links.

A ``CouplingMap`` states which wire pairs are coupled. A
``DirectedCouplingMap`` states which way round each of those pairs runs for a
control/target opcode, and until the routing cost model read that, a gate placed
against the declared direction was left for ``legalize_directed_cx`` to repair
afterwards. The router now prices and performs that repair itself, as an
orientation SWAP: a SWAP across the offending link exchanges the two operands'
states, so the gate runs in the declared direction between the two halves of the
sandwich, and a SWAP is legal across that link whichever way it is declared.

These tests pin the three claims that change makes:

* every two-wire operation a routed program emits runs where the device says it
  runs, including the direction, and the program still computes the source state;
* the price and the plan state the same ordering cost, so an estimate is a plan
  and not a bound; and
* consequently a later direction-legalization pass finds nothing left to
  reverse, which is what makes this the router's work rather than a deferred
  bill.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler import compile as compile_circuit
from flagquantum.compiler.directed_topology import DirectedCouplingMap
from flagquantum.compiler.operand_semantics import _TWO_WIRE_OPERAND_SYMMETRY
from flagquantum.compiler.routing import (
    DEPLOYABLE_ROUTING_STRATEGIES,
    ROUTING_STRATEGIES,
    CouplingMap,
    estimate_routing_cost,
    route_to_topology,
    select_routing_strategy,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.simulation.statevector.local import run_local_statevector

pytestmark = pytest.mark.unit

# A device whose only link runs from wire 1 to wire 0, so that a source asking
# for ``cx(0, 1)`` asks for exactly the direction the device does not declare.
_REVERSED_DEVICE = DirectedCouplingMap(2, ((1, 0),))
_ALIGNED_DEVICE = DirectedCouplingMap(2, ((0, 1),))
_NATIVE = ("h", "cx", "cz", "swap", "ry")


def _reversed_workload() -> fq.Circuit:
    circuit = fq.Circuit(2)
    circuit.ry(0, theta=0.37)
    circuit.cx(0, 1)
    circuit.ry(1, theta=0.59)
    return circuit


def _state(program: fq.Circuit | CircuitIR) -> torch.Tensor:
    ir = program.to_ir() if isinstance(program, fq.Circuit) else program
    return run_local_statevector(
        ir, batch_size=1, device=torch.device("cpu"), dtype=torch.complex128
    )


def _two_wire(program: CircuitIR) -> list[Instruction]:
    return [item for item in program if len(item.wires) == 2]


def _direction_violations(
    program: CircuitIR, device: DirectedCouplingMap
) -> list[tuple[str, tuple[int, ...]]]:
    """Report emitted two-wire gates the ordered device cannot run where they are."""

    violations: list[tuple[str, tuple[int, ...]]] = []
    for instruction in _two_wire(program):
        if (
            not device.has_weak_edge(*instruction.wires)
            or _TWO_WIRE_OPERAND_SYMMETRY.get(instruction.name) is False
            and not (device.has_edge(*instruction.wires))
        ):
            violations.append((instruction.name, instruction.wires))
    return violations


@pytest.mark.parametrize("strategy", ROUTING_STRATEGIES)
def test_every_strategy_repairs_a_reversed_gate_with_an_orientation_swap(
    strategy: str,
) -> None:
    """A reversed gate is executed between two SWAPs, with its operands swapped.

    The repair is only a repair if the gate in the middle runs the other way
    round: the opening SWAP exchanges the operands' states, so emitting the gate
    in its source order there would compute the source program's reverse and the
    closing SWAP would undo the exchange without ever having repaired anything.
    The check therefore reads the executed gate's wires, not just the SWAP count.
    """

    source = _reversed_workload()
    routed = route_to_topology(source, _REVERSED_DEVICE, strategy=strategy)

    assert _direction_violations(routed, _REVERSED_DEVICE) == []
    plan = routed.metadata["routing"]
    assert plan["direction_semantics"] == "directed_cx"
    assert plan["direction_swap_count"] == 1
    # One orientation SWAP pair: the opening half and its closing half.
    assert plan["inserted_swap_count"] == 2
    assert plan["planned_inserted_swap_count"] == 2

    emitted = list(routed.instructions)
    gate_index = next(index for index, item in enumerate(emitted) if item.name == "cx")
    assert emitted[gate_index].wires == (1, 0)
    assert emitted[gate_index - 1].name == "swap"
    assert emitted[gate_index - 1].metadata["routing_phase"] == "orientation"
    assert set(emitted[gate_index - 1].wires) == {0, 1}
    # The closing half is the next SWAP on that link: a per-gate strategy closes
    # the sandwich immediately and a layout-carrying one closes it at the end of
    # the program, so the phase is what identifies the restore either way.
    closing = next(
        item
        for item in emitted[gate_index + 1 :]
        if item.name == "swap" and set(item.wires) == {0, 1}
    )
    assert "restore" in closing.metadata["routing_phase"]

    # The repair changes where the gate runs and not what the program computes.
    assert torch.allclose(_state(routed), _state(source), atol=1e-9)


@pytest.mark.parametrize("strategy", ROUTING_STRATEGIES)
def test_a_gate_already_running_the_declared_way_is_left_alone(strategy: str) -> None:
    """An ordered device costs nothing extra for a gate that already obeys it."""

    source = _reversed_workload()
    routed = route_to_topology(source, _ALIGNED_DEVICE, strategy=strategy)

    plan = routed.metadata["routing"]
    assert plan["direction_swap_count"] == 0
    assert plan["inserted_swap_count"] == 0
    # The gate is executed where it stands, so it is a topology gate and not a
    # gate the device had to be talked into carrying.
    assert plan["topology_gate_count"] == 1
    assert plan["routed_gate_count"] == 0
    assert _direction_violations(routed, _ALIGNED_DEVICE) == []
    assert torch.allclose(_state(routed), _state(source), atol=1e-9)


@pytest.mark.parametrize("strategy", ROUTING_STRATEGIES)
@pytest.mark.parametrize("opcode", ("cz", "swap", "rxx"))
def test_an_operand_symmetric_opcode_needs_only_the_link(
    strategy: str, opcode: str
) -> None:
    """An opcode whose unitary ignores operand order has no way to be reversed.

    The device declares ``1 -> 0``, and a symmetric opcode on ``(0, 1)`` crosses
    that link, so it is executable where it stands. Repairing it would add two
    SWAPs to a gate whose result a SWAP cannot change the direction of.

    Executable *where it stands* is the clause that is easy to get wrong in the
    other direction: requiring the declared direction for these opcodes instead
    of accepting either would leave the gate where it is and insert nothing, so
    the emitted program would be identical and only the report would change. That
    is why the counts are read here and not just the SWAP count -- the two entries
    agree that this gate needed no routing at all.
    """

    circuit = fq.Circuit(2)
    if opcode == "swap":
        circuit.swap(0, 1)
    elif opcode == "rxx":
        circuit.rxx(0, 1, theta=0.4)
    else:
        circuit.cz(0, 1)

    routed = route_to_topology(circuit, _REVERSED_DEVICE, strategy=strategy)
    plan = routed.metadata["routing"]
    assert plan["direction_swap_count"] == 0
    assert plan["inserted_swap_count"] == 0
    assert plan["topology_gate_count"] == 1
    assert plan["routed_gate_count"] == 0
    estimate = estimate_routing_cost(circuit, _REVERSED_DEVICE, strategy=strategy)
    assert estimate.routed_gate_count == 0
    assert [item.name for item in routed if len(item.wires) == 2] == [opcode]
    assert torch.allclose(_state(routed), _state(circuit), atol=1e-9)


# Every native opcode whose unitary does depend on which operand is the control.
# Read from the classification itself so a newly classified opcode is exercised
# here on the run that adds it.
_CONTROL_TARGET_BUILDERS: tuple[tuple[str, float | None], ...] = (
    ("cx", None),
    ("cy", None),
    ("crx", 0.31),
    ("cry", 0.42),
    ("crz", 0.53),
)


@pytest.mark.parametrize("strategy", ROUTING_STRATEGIES)
@pytest.mark.parametrize(("opcode", "theta"), _CONTROL_TARGET_BUILDERS)
def test_the_repair_works_for_every_control_target_opcode(
    strategy: str, opcode: str, theta: float | None
) -> None:
    """Exchanging the operands and exchanging the states repair any of them.

    A SWAP conjugates a two-wire gate into the same gate with its operands the
    other way round, whatever the gate is. That is why this repair needs no
    rewrite table per opcode: the same three instructions orient a CY, a CRX, a
    CRY and a CRZ. The H-conjugation path used elsewhere in this compiler covers
    CX alone, so an opcode reaching an ordered device through this router is
    handled where it would otherwise have to be refused.
    """

    assert _TWO_WIRE_OPERAND_SYMMETRY[opcode] is False
    circuit = fq.Circuit(2)
    if theta is None:
        getattr(circuit, opcode)(0, 1)
    else:
        getattr(circuit, opcode)(0, 1, theta=theta)

    routed = route_to_topology(circuit, _REVERSED_DEVICE, strategy=strategy)
    plan = routed.metadata["routing"]
    assert plan["direction_swap_count"] == 1
    assert plan["inserted_swap_count"] == 2
    assert _direction_violations(routed, _REVERSED_DEVICE) == []
    executed = [item for item in routed if item.name == opcode]
    assert [item.wires for item in executed] == [(1, 0)]
    assert torch.allclose(_state(routed), _state(circuit), atol=1e-9)


@pytest.mark.parametrize("strategy", ROUTING_STRATEGIES)
def test_the_price_of_an_ordered_device_is_the_plan_it_produces(strategy: str) -> None:
    """The estimate and the materialized plan agree on the ordering cost too.

    A price that omits the repair would let the automatic choice compare a cheap
    plan against a dearer one that is priced honestly, and the whole point of
    counting the repair is that the comparison sees it.
    """

    source = _reversed_workload()
    estimate = estimate_routing_cost(source, _REVERSED_DEVICE, strategy=strategy)
    plan = route_to_topology(source, _REVERSED_DEVICE, strategy=strategy).metadata[
        "routing"
    ]

    assert estimate.direction_semantics == "directed_cx"
    assert estimate.direction_swap_count == plan["direction_swap_count"] == 1
    assert (
        estimate.planned_inserted_swap_count
        == plan["inserted_swap_count"]
        == plan["planned_inserted_swap_count"]
        == 2
    )
    summary = estimate.summary()
    assert summary["direction_swap_count"] == 1
    assert summary["direction_semantics"] == "directed_cx"


def test_the_automatic_choice_reads_the_ordering_on_every_candidate() -> None:
    """Every priced candidate states the ordering cost it was charged for."""

    selection = select_routing_strategy(_reversed_workload(), _REVERSED_DEVICE)

    assert selection.selected_strategy in DEPLOYABLE_ROUTING_STRATEGIES
    chosen = selection.candidates[selection.selected_strategy]
    assert chosen.direction_swap_count == 1
    for name, candidate in selection.candidates.items():
        assert candidate.direction_semantics == "directed_cx", name
        # The ordering cost is inside the total, never beside it.
        assert 2 * candidate.direction_swap_count <= (
            candidate.planned_inserted_swap_count
        ), name
    summary = selection.summary()
    assert (
        summary["candidates"][selection.selected_strategy]["direction_swap_count"] == 1
    )


def test_the_automatic_choice_ranks_on_the_total_it_charged_for_the_ordering() -> None:
    """A charged-for repair may not be credited back when the candidates rank.

    The three candidates are priced 8, 8 and 10 inserted SWAPs on this program,
    and the cheapest two charge 2 orientation pairs each while the dearest charges
    4. Those numbers are what make this workload the load-bearing one for the
    ranking key: subtracting the repair from a candidate's total would price the
    dearest plan at 10 - 8 = 2 against 8 - 4 = 4 for both cheaper ones, so the
    choice would promote the plan that inserts two more SWAPs than either of the
    others. Reading the ordering is what makes the comparison honest, so the
    ordering cannot be discounted out of the comparison it was added to.
    """

    program = CircuitIR(
        3,
        (
            Instruction("cx", (2, 0)),
            Instruction("cx", (2, 0)),
            Instruction("cx", (1, 2)),
            Instruction("cx", (1, 2)),
        ),
        dtype="complex128",
    )
    device = DirectedCouplingMap(3, ((0, 1), (1, 2)))
    selection = select_routing_strategy(program, device)

    priced = {
        name: (
            candidate.planned_inserted_swap_count,
            candidate.estimated_instruction_count,
        )
        for name, candidate in selection.candidates.items()
    }
    assert priced == {
        "restore_after_each_gate": (8, 12),
        "persistent_layout": (8, 12),
        "sabre": (10, 14),
    }
    assert selection.selected_strategy == min(priced, key=lambda name: priced[name])
    # The winner is one of the two candidates that charge least for the ordering,
    # and the candidate the discounting key would have promoted is not it.
    chosen = selection.candidates[selection.selected_strategy]
    assert chosen.direction_swap_count == 2
    assert chosen.planned_inserted_swap_count == 8
    assert selection.candidates["sabre"].planned_inserted_swap_count == 10

    plan = route_to_topology(program, device, strategy=selection.selected_strategy)
    routing = plan.metadata["routing"]
    assert routing["inserted_swap_count"] == 8
    assert routing["planned_inserted_swap_count"] == 8
    assert routing["direction_swap_count"] == 2


def test_compile_routes_an_ordered_device_without_rebuilding_it() -> None:
    """The compiler entry reaches the ordered device instead of flattening it.

    ``compile`` used to coerce anything that was not a ``CouplingMap`` into one,
    which for a ``DirectedCouplingMap`` meant an iteration failure. Coercing it
    would be worse than failing: an edge sequence cannot carry the direction
    rule, so the routing plan would report ``logical_wire_order_preserved`` for a
    device that declares otherwise.
    """

    compiled = compile_circuit(
        _reversed_workload(), coupling_map=_REVERSED_DEVICE, routing_strategy="auto"
    )
    plan = compiled.metadata["routing"]
    assert plan["direction_semantics"] == "directed_cx"
    assert plan["direction_swap_count"] == 1
    assert plan["coupling_edges"] == ((1, 0),)
    assert _direction_violations(compiled, _REVERSED_DEVICE) == []
    assert compiled.metadata["routing_strategy_selection"]["selected_strategy"] == (
        plan["strategy"]
    )
    assert torch.allclose(_state(compiled), _state(_reversed_workload()), atol=1e-9)


def test_an_ordered_device_refuses_an_opcode_it_cannot_classify() -> None:
    """An unclassified two-wire opcode fails closed instead of being placed.

    A ``CouplingMap`` needs only to know that two wires are coupled, so it can
    route an opcode it has no semantics for. An ordered device cannot: whether
    the operands may be exchanged is exactly the question, and a guess would
    either insert two SWAPs no rule asked for or execute the gate the wrong way
    round. The refusal names the opcode and the table that would classify it.
    """

    matrix = torch.eye(4, dtype=torch.complex128)
    program = CircuitIR(
        n_wires=2, instructions=(Instruction("unclassified2q", (0, 1), matrix=matrix),)
    )

    for strategy in ROUTING_STRATEGIES:
        with pytest.raises(ValueError) as refusal:
            route_to_topology(program, _REVERSED_DEVICE, strategy=strategy)
        message = str(refusal.value)
        assert "unclassified2q" in message
        assert "OPERATOR_SCHEMAS" in message

    # The undirected device carries the same opcode, which is what makes this a
    # property of the ordering rather than of the opcode.
    undirected = CouplingMap(2, ((0, 1),))
    assert [item.name for item in route_to_topology(program, undirected)] == [
        "unclassified2q"
    ]


@pytest.mark.parametrize("strategy", ROUTING_STRATEGIES)
def test_the_routed_ordered_program_has_nothing_left_to_legalize(strategy: str) -> None:
    """The router does not hand the direction bill to a later pass.

    ``legalize_directed_cx`` is the pass that rewrites a reversed CX into an
    H-conjugated one. On a program this router produced it must find nothing to
    reverse, because the router already placed every gate in the declared
    direction; a nonzero count would mean routing left work it claims to price.
    """

    from flagquantum.compiler.direction_legalization import legalize_directed_cx
    from tests.team.compiler.test_direction_operand_semantics import _snapshot

    routed = route_to_topology(
        _reversed_workload(), _REVERSED_DEVICE, strategy=strategy
    )
    legalized = legalize_directed_cx(
        routed,
        coupling_map=_REVERSED_DEVICE,
        snapshot=_snapshot(_NATIVE),
        native_opcodes=_NATIVE,
    )

    assert legalized.reversed_cx_count == 0
    assert [(item.name, item.wires) for item in legalized.program] == [
        (item.name, item.wires) for item in routed
    ]


def test_a_reversed_gate_that_needs_routing_is_repaired_where_it_lands() -> None:
    """A gate reaches its own edge by SWAPs and is then oriented on that edge.

    The two repairs are separate: the hop series moves the operands onto a
    coupled pair and the orientation SWAP runs the gate the declared way once
    they are there. They are counted separately for that reason.
    """

    device = DirectedCouplingMap(3, ((0, 1), (1, 2)))
    circuit = fq.Circuit(3)
    circuit.cx(0, 1)
    circuit.cx(2, 0)

    routed = route_to_topology(circuit, device, strategy="restore_after_each_gate")
    plan = routed.metadata["routing"]

    assert _direction_violations(routed, device) == []
    assert plan["topology_gate_count"] == 2
    assert plan["routed_gate_count"] == 1
    # The second gate is the reversed one and it also had to reach wire 0 from
    # wire 2, so it costs one hop pair and one orientation pair.
    assert plan["direction_swap_count"] == 1
    assert plan["inserted_swap_count"] == 4
    assert torch.allclose(_state(routed), _state(circuit), atol=1e-9)
    phases = [
        item.metadata.get("routing_phase") for item in routed if item.name == "swap"
    ]
    assert "forward" in phases
    assert "orientation" in phases


def test_the_ordered_device_is_read_through_every_accepted_argument_form() -> None:
    """A device instance, an edge sequence, and an ordered device all route.

    The ordered device is the only form that carries the direction rule, and the
    other two must keep behaving exactly as they did, so this pins the three
    against each other on one program.
    """

    program = _reversed_workload()
    edges: tuple[tuple[int, int], ...] = ((1, 0),)

    from_map = route_to_topology(program, CouplingMap(2, edges))
    from_edges = route_to_topology(program, edges)
    ordered = route_to_topology(program, DirectedCouplingMap(2, edges))

    for undirected in (from_map, from_edges):
        plan: dict[str, Any] = undirected.metadata["routing"]
        assert plan["direction_semantics"] == "logical_wire_order_preserved"
        assert plan["direction_swap_count"] == 0
        assert plan["inserted_swap_count"] == 0

    assert ordered.metadata["routing"]["direction_swap_count"] == 1
    assert ordered.metadata["routing"]["inserted_swap_count"] == 2
    assert _direction_violations(ordered, DirectedCouplingMap(2, edges)) == []
    # Both devices accept the link and both place the gate on it. Only the
    # ordered one cares which way it runs, so the undirected plans are legal on
    # the graph and not runnable as an ordered device declares the graph -- that
    # difference is the ordering this row prices.
    graph = CouplingMap(2, edges)
    for routed in (from_map, from_edges):
        assert all(graph.has_edge(*item.wires) for item in _two_wire(routed))
        assert _direction_violations(routed, DirectedCouplingMap(2, edges)) != []


def test_the_target_legalization_composition_still_divides_the_orientation_work() -> (
    None
):
    """Where the orientation repair happens is pinned, not assumed.

    ``route_to_topology`` reads, prices, and performs the direction repair, and
    that is the router this row's cost model describes. A caller who reaches an
    ordered device through ``legalize_circuit_topology`` instead does not get that
    router: the composed target-legalization path routes on the weak projection and
    leaves every reversed gate for ``legalize_directed_cx``, the H-conjugation pass
    that predates this work. Both produce a program the ordered device can run, so
    this is a division of labour rather than a defect, but it is a division a
    reader of the row would otherwise have to infer, and an inferred division is
    one that a later refactor can move without anything going red.

    The measurement is therefore explicit in three parts: the topology stage
    inserts no SWAP for a gate that is already on a coupled pair, it emits that
    gate still running against the declared direction, and the direction pass is
    the one that repairs it.
    """

    from flagquantum.compiler.direction_legalization import legalize_directed_cx
    from flagquantum.compiler.topology_legalization import legalize_circuit_topology
    from tests.hybrid_compiler.test_target_legalization import _snapshot

    source = CircuitIR(
        2,
        (Instruction("h", (0,)), Instruction("cx", (0, 1))),
        dtype="complex128",
    )

    topology = legalize_circuit_topology(
        source, coupling_map=_REVERSED_DEVICE, snapshot=_snapshot(), strategy="auto"
    )

    # The pair is coupled, so there is nothing for a hop series to do, and the
    # stage reports the ordered device's semantics without having repaired the
    # gate those semantics describe.
    assert topology.strategy == "persistent_layout"
    assert topology.direction_semantics == "directed_cx"
    assert topology.inserted_swap_count == 0
    assert _direction_violations(topology.program, _REVERSED_DEVICE) != []

    repaired = legalize_directed_cx(
        topology.program,
        coupling_map=_REVERSED_DEVICE,
        snapshot=_snapshot(),
        native_opcodes=("h", "cx", "rx", "amplitude_damping"),
    )

    assert repaired.reversed_cx_count == 1
    assert _direction_violations(repaired.program, _REVERSED_DEVICE) == []
    # H-conjugation rewrites the gate into place rather than swapping around it:
    # the repair replaces one operation with five and inserts no SWAP at all,
    # which is why the two entries' costs are not counts of the same thing.
    assert len(repaired.program.instructions) - len(topology.program.instructions) == 4
