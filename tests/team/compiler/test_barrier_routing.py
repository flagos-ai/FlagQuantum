"""The router's reading of a barrier across every strategy and both devices.

``barrier`` is not a Core opcode: it carries no unitary and states an order. The
consequence the routing contract has to hold is therefore not "the directive
survives" but "the directive costs nothing and buys nothing":

* it needs no coupling, so a two-wire barrier over wires the device has no path
  between is routable, and the same wires carrying a real gate still are not;
* it cannot be made more executable, so no strategy inserts a SWAP for it and no
  strategy counts it as topology work;
* it is still part of the program, so it is emitted exactly once and remapped
  onto the wires the layout in force puts it on.

The last two points are what separates this contract from "delete the barrier":
every test below that asserts a zero is paired with the same program shape
carrying a real two-wire gate, so a router that simply dropped or ignored the
directive could not pass.
"""

from __future__ import annotations

import pytest
import torch

from flagquantum.compiler.directed_topology import (
    DirectedCouplingMap,
    route_to_directed_topology,
)
from flagquantum.compiler.ordering_barrier import is_ordering_barrier
from flagquantum.compiler.routing import (
    ROUTING_STRATEGIES,
    CouplingMap,
    estimate_routing_cost,
    route_to_topology,
)
from flagquantum.compiler.sabre import plan_sabre_swaps
from flagquantum.core.ir import CircuitIR, Instruction

pytestmark = pytest.mark.unit

# A line is the smallest device on which a two-wire operation can be non-local,
# and the second device is deliberately disconnected so the two components are
# separated by a missing coupling rather than by distance.
_LINE4 = CouplingMap.line(4)
_GAP4 = CouplingMap(4, ((0, 1), (2, 3)))

_TRIANGLE_AND_TAIL = CouplingMap(5, ((0, 1), (1, 2), (0, 2), (2, 3), (3, 4)))
_DIRECTED_TRIANGLE_AND_TAIL = DirectedCouplingMap(
    5,
    (
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
        (0, 2),
        (2, 0),
        (2, 3),
        (3, 2),
        (3, 4),
        (4, 3),
    ),
)

# ``barrier`` is not a declared opcode, and IR admits an undeclared opcode only
# when the instruction carries a matrix or one of the two markers. A barrier is
# an ordering directive, so it arrives through the dynamic marker.
_BARRIER_SPAN = (0, 1, 2, 3)


def _barrier(*qubits: int) -> Instruction:
    return Instruction("barrier", qubits, {}, None, {"is_dynamic": True})


def _barrier_with_a_matrix(*qubits: int) -> Instruction:
    return Instruction(
        "barrier",
        qubits,
        {},
        torch.eye(1 << len(qubits), dtype=torch.complex128),
        {"is_dynamic": True},
    )


def _measure(qubit: int) -> Instruction:
    return Instruction(
        "measure",
        (qubit,),
        {},
        None,
        {"is_dynamic": True, "classical_bit": qubit},
    )


def _spans(program: CircuitIR) -> list[tuple[int, ...]]:
    return [item.wires for item in program.instructions if item.name == "barrier"]


def _emitted_order(program: CircuitIR) -> list[int]:
    """Return the source index of every emitted non-SWAP operation, in order."""

    indices: list[int] = []
    for instruction in program.instructions:
        if instruction.metadata.get("routing_phase") is not None:
            continue
        index = instruction.metadata.get("source_instruction_index")
        assert isinstance(index, int), "an emitted operation lost its source index"
        indices.append(index)
    return indices


# --------------------------------------------------------------------------- #
# The reading itself
# --------------------------------------------------------------------------- #


def test_the_directive_is_read_from_the_missing_unitary_not_the_name() -> None:
    """An instruction that names the barrier while carrying a matrix is a gate."""

    assert is_ordering_barrier(_barrier(0, 2, 4))
    assert is_ordering_barrier(_barrier(0, 1))
    assert not is_ordering_barrier(_barrier_with_a_matrix(0, 2, 4))
    assert not is_ordering_barrier(Instruction("h", (0,)))


# --------------------------------------------------------------------------- #
# No coupling is required, at any width
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("strategy", ROUTING_STRATEGIES)
def test_a_device_wide_barrier_needs_no_coupling(strategy: str) -> None:
    """A four-wire directive over a device with no four-wire coupling routable."""

    source = CircuitIR(
        4,
        (_barrier(*_BARRIER_SPAN), Instruction("cx", (0, 3)), _measure(0)),
    )
    routed = route_to_topology(source, _LINE4, strategy=strategy)

    span = _spans(routed)
    assert len(span) == 1, "routing dropped or duplicated the ordering directive"
    assert len(span[0]) == len(_BARRIER_SPAN), "the directive changed arity"
    assert len(set(span[0])) == len(span[0]), "the directive collapsed onto one wire"
    assert all(qubit < _LINE4.n_qubits for qubit in span[0])
    # A line carries no three-wire coupling, so the refusal this test replaces
    # would have fired had the directive been read as a device operation.
    assert not _LINE4.has_edge(0, 2)


@pytest.mark.parametrize("strategy", ROUTING_STRATEGIES)
def test_a_two_wire_barrier_needs_no_coupling_path(strategy: str) -> None:
    """The disconnected device separates wires 0 and 3, and the directive crosses."""

    routed = route_to_topology(
        CircuitIR(4, (_barrier(0, 3),)), _GAP4, strategy=strategy
    )
    routing = routed.metadata["routing"]

    assert _spans(routed) == [(0, 3)]
    assert routing["inserted_swap_count"] == 0
    assert routing["topology_gate_count"] == 0


@pytest.mark.parametrize("strategy", ROUTING_STRATEGIES)
def test_the_refusal_above_is_not_vacuous(strategy: str) -> None:
    """The same two wires carrying a real gate are still refused."""

    with pytest.raises(ValueError):
        route_to_topology(
            CircuitIR(4, (Instruction("cx", (0, 3)),)), _GAP4, strategy=strategy
        )


# --------------------------------------------------------------------------- #
# No topology work is charged, at any width
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("strategy", ROUTING_STRATEGIES)
@pytest.mark.parametrize(
    "wires",
    [(0, 1), (0, 2), (0, 3), (0, 1, 2), (0, 1, 2, 3)],
)
def test_a_barrier_alone_costs_nothing(strategy: str, wires: tuple[int, ...]) -> None:
    routed = route_to_topology(
        CircuitIR(4, (_barrier(*wires),)), _LINE4, strategy=strategy
    )
    routing = routed.metadata["routing"]

    assert routing["topology_gate_count"] == 0
    assert routing["routed_gate_count"] == 0
    assert routing["inserted_swap_count"] == 0
    assert _spans(routed) == [wires], "the directive was not emitted unchanged"
    assert len(routed.instructions) == 1


@pytest.mark.parametrize("strategy", ROUTING_STRATEGIES)
def test_the_zero_above_is_not_vacuous(strategy: str) -> None:
    """A real two-wire gate over the same wires does pay for the topology."""

    routed = route_to_topology(
        CircuitIR(4, (Instruction("cx", (0, 3)),)), _LINE4, strategy=strategy
    )
    routing = routed.metadata["routing"]

    assert routing["topology_gate_count"] == 1
    assert routing["routed_gate_count"] == 1
    assert routing["inserted_swap_count"] > 0


@pytest.mark.parametrize("strategy", ROUTING_STRATEGIES)
def test_a_barrier_adds_nothing_to_the_cost_of_the_gate_beside_it(
    strategy: str,
) -> None:
    """The differential form: the directive is free next to a real gate."""

    gate_only = route_to_topology(
        CircuitIR(4, (Instruction("cx", (0, 3)),)), _LINE4, strategy=strategy
    )
    with_barrier = route_to_topology(
        CircuitIR(
            4,
            (_barrier(0, 3), Instruction("cx", (0, 3))),
        ),
        _LINE4,
        strategy=strategy,
    )

    assert (
        with_barrier.metadata["routing"]["inserted_swap_count"]
        == gate_only.metadata["routing"]["inserted_swap_count"]
    )
    assert (
        with_barrier.metadata["routing"]["topology_gate_count"]
        == gate_only.metadata["routing"]["topology_gate_count"]
    )


@pytest.mark.parametrize("strategy", ROUTING_STRATEGIES)
def test_a_barrier_keeps_its_place_between_the_operations_on_its_wires(
    strategy: str,
) -> None:
    """Free does not mean unobserved: the directive still separates its wires."""

    source = CircuitIR(
        4,
        (
            Instruction("cx", (0, 3)),
            _barrier(0, 3),
            Instruction("cx", (0, 3)),
        ),
    )
    routed = route_to_topology(source, _LINE4, strategy=strategy)

    assert _emitted_order(routed) == [0, 1, 2]
    span = _spans(routed)
    assert len(span) == 1
    assert len(span[0]) == 2
    assert span[0][0] != span[0][1], "the directive collapsed onto one wire"
    assert all(qubit < _LINE4.n_qubits for qubit in span[0])


# --------------------------------------------------------------------------- #
# The planner does not give the directive a SWAP to place
# --------------------------------------------------------------------------- #


def test_the_sabre_planner_inserts_no_swap_for_a_barrier() -> None:
    """The directive never blocks the front layer, so no SWAP is chosen for it."""

    barrier_only = plan_sabre_swaps(CircuitIR(4, (_barrier(0, 3),)), _LINE4)
    gate_only = plan_sabre_swaps(CircuitIR(4, (Instruction("cx", (0, 3)),)), _LINE4)

    assert barrier_only.swaps == ()
    assert gate_only.swaps != (), (
        "the planner finds no SWAP for a real gate either, so the assertion "
        "above holds without the directive being exempt"
    )


# --------------------------------------------------------------------------- #
# The estimate the compiler publishes agrees with what routing pays
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("strategy", ["restore_after_each_gate", "persistent_layout"])
def test_the_cost_estimate_agrees_with_the_routed_plan_for_a_barrier(
    strategy: str,
) -> None:
    """An estimate that charges for a directive the router does not route is a lie."""

    source = CircuitIR(4, (_barrier(0, 3), Instruction("cx", (0, 3)), _measure(0)))
    routed = route_to_topology(source, _LINE4, strategy=strategy)
    estimate = estimate_routing_cost(source, _LINE4, strategy=strategy)
    routing = routed.metadata["routing"]

    assert estimate.topology_gate_count == routing["topology_gate_count"] == 1
    assert estimate.planned_inserted_swap_count == routing["inserted_swap_count"]
    assert estimate.estimated_instruction_count == len(routed.instructions)
    assert estimate.routed_gate_count == routing["routed_gate_count"]


# --------------------------------------------------------------------------- #
# The directive is emitted, on the wires the layout puts it on
# --------------------------------------------------------------------------- #


def _barrier_first() -> CircuitIR:
    """A device-wide barrier ahead of the only routed gate."""

    return CircuitIR(
        4,
        (
            _barrier(*_BARRIER_SPAN),
            Instruction("cx", (0, 3)),
            *(_measure(qubit) for qubit in range(4)),
        ),
    )


@pytest.mark.parametrize("strategy", ROUTING_STRATEGIES)
def test_a_routed_barrier_names_the_qubits_the_layout_puts_it_on(
    strategy: str,
) -> None:
    """The span is recomputed from the layout, not copied from the source."""

    routed = route_to_topology(_barrier_first(), _LINE4, strategy=strategy)
    initial = routed.metadata["routing"]["initial_logical_to_physical"]

    assert _spans(routed) == [tuple(initial[qubit] for qubit in _BARRIER_SPAN)]


def test_the_barrier_span_assertion_above_is_not_vacuous() -> None:
    """One strategy starts from a non-identity layout, so the span must move."""

    routed = route_to_topology(_barrier_first(), _LINE4, strategy="sabre_layout")
    initial = routed.metadata["routing"]["initial_logical_to_physical"]

    assert initial != tuple(range(4)), (
        "every strategy starts from the identity layout, so this test cannot "
        "distinguish a recomputed span from a copied one"
    )
    assert _spans(routed) == [tuple(initial[qubit] for qubit in _BARRIER_SPAN)]


def _final_measurements_after_a_device_wide_barrier() -> CircuitIR:
    """The program Qiskit's barrier pass exists to protect.

    The gate is off-device for ``_LINE4``, so routing has to move it, and every
    measurement is written after the barrier that covers all four wires.
    """

    return CircuitIR(
        4,
        (
            Instruction("cx", (0, 3)),
            _barrier(*_BARRIER_SPAN),
            *(_measure(qubit) for qubit in range(4)),
        ),
    )


def _measurements_before_the_routed_gate(program: CircuitIR) -> bool:
    """Whether any measurement ended up ahead of the gate it was written after."""

    names = [item.name for item in program.instructions]
    gate = names.index("cx")

    return any(name == "measure" and index < gate for index, name in enumerate(names))


@pytest.mark.parametrize("strategy", ROUTING_STRATEGIES)
def test_a_device_wide_barrier_keeps_the_measurements_after_it(strategy: str) -> None:
    """The directive still separates the phases it was written to separate.

    This is the property Qiskit's ``BarrierBeforeFinalMeasurements`` depends on,
    measured through routing instead of inferred from the directive being present.
    """

    routed = route_to_topology(
        _final_measurements_after_a_device_wide_barrier(), _LINE4, strategy=strategy
    )
    names = [item.name for item in routed.instructions]
    barrier_index = names.index("barrier")

    assert names.index("cx") < barrier_index
    assert not _measurements_before_the_routed_gate(routed)


def test_the_barrier_property_above_is_not_vacuous() -> None:
    """One strategy does move a measurement across the gate without the barrier.

    Without this anchor the parametrized assertion above could hold because no
    strategy ever reorders a measurement, rather than because the barrier held.
    """

    passing = [
        strategy
        for strategy in ROUTING_STRATEGIES
        if _measurements_before_the_routed_gate(
            route_to_topology(
                CircuitIR(
                    4,
                    (
                        Instruction("cx", (0, 3)),
                        *(_measure(qubit) for qubit in range(4)),
                    ),
                ),
                _LINE4,
                strategy=strategy,
            )
        )
    ]

    assert passing, (
        "no strategy moves a final measurement ahead of the gate it follows, so "
        "the barrier assertion above holds without the barrier"
    )


# --------------------------------------------------------------------------- #
# The exemption is the missing unitary, not the opcode name
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("strategy", ROUTING_STRATEGIES)
def test_a_barrier_named_instruction_that_carries_a_matrix_is_still_a_gate(
    strategy: str,
) -> None:
    source = CircuitIR(
        5,
        (_barrier_with_a_matrix(0, 2, 4),),
    )
    with pytest.raises(ValueError, match="no verified physical connectivity rule"):
        route_to_topology(source, _TRIANGLE_AND_TAIL, strategy=strategy)


def test_the_directed_router_shares_the_directive_exemption() -> None:
    """One rule serves both routers, so the exemption is not duplicated."""

    routed = route_to_directed_topology(
        CircuitIR(5, (_barrier(0, 1, 2), Instruction("cx", (0, 1)))),
        _DIRECTED_TRIANGLE_AND_TAIL,
    )

    assert _spans(routed) == [(0, 1, 2)]
