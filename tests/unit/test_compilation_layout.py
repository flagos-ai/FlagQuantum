import random

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler import (
    CouplingMap,
    Layout,
    apply_layout,
    final_layout,
    remove_layout_restore,
    route_to_topology,
)
from flagquantum.compiler.directed_topology import (
    DirectedCouplingMap,
    route_to_directed_topology,
)
from flagquantum.core.ir import CircuitIR, Instruction, MeasurementNode, ObservableNode
from flagquantum.deployment.routing_evidence import (
    DeploymentRoutingEvidenceError,
    validate_deployment_routing_plan,
)

pytestmark = pytest.mark.unit


def _weakly_directed(coupling_map: CouplingMap) -> tuple[tuple[int, int], ...]:
    """Return both directions of every edge, so a route may use an edge either way."""

    return tuple(
        (left, right)
        for edge in coupling_map.edges
        for left, right in (edge, (edge[1], edge[0]))
    )


def _random_two_wire_program(seed: int, n_wires: int, gate_count: int) -> CircuitIR:
    rng = random.Random(seed)
    instructions = []
    for _ in range(gate_count):
        left, right = rng.sample(range(n_wires), 2)
        instructions.append(Instruction("cx", (left, right)))
        instructions.append(Instruction("ry", (left,), params={"theta": 0.37}))
    return CircuitIR(n_wires, tuple(instructions))


def _amplitude_permutation(layout: Layout) -> torch.Tensor:
    """Return the amplitude index array that reads a relabelled state in the original order.

    Wire 0 is the most significant basis bit. Moving logical wire ``l`` onto
    physical wire ``p`` moves the bit at position ``l`` to position ``p``, so the
    permutation moves the bit in the opposite direction of the layout.
    """

    inverse = Layout(layout.physical_to_logical)
    n_wires = inverse.n_wires
    index = torch.arange(1 << n_wires)
    moved = torch.zeros_like(index)
    for physical in range(n_wires):
        bit = (index >> (n_wires - physical - 1)) & 1
        moved |= bit << (n_wires - inverse.logical_of(physical) - 1)
    return moved


def test_layout_validates_a_complete_permutation() -> None:
    layout = Layout((2, 0, 1))

    assert layout.n_wires == 3
    assert layout.logical_to_physical == (2, 0, 1)
    assert layout.physical_to_logical == (1, 2, 0)
    assert layout.physical_of(0) == 2
    # Physical wire 2 holds logical 0, physical 1 holds logical 2.
    assert layout.logical_of(2) == 0
    assert layout.logical_of(1) == 2
    assert layout.map_wires((0, 2)) == (2, 1)
    assert str(layout) == "Layout((2, 0, 1))"

    with pytest.raises(ValueError, match="permutation"):
        Layout((0, 0, 1))
    with pytest.raises(ValueError, match="permutation"):
        Layout((0, 2))
    with pytest.raises(ValueError, match="at least one logical wire"):
        Layout(())
    with pytest.raises(ValueError, match="integer physical wires"):
        Layout((0, 1, 2.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="outside the layout"):
        layout.physical_of(3)
    with pytest.raises(ValueError, match="outside the layout"):
        layout.logical_of(-1)


def test_layout_applies_physical_swaps_to_the_positions_not_the_names() -> None:
    layout = Layout((2, 0, 1))

    # A SWAP exchanges the wires the two physical positions hold, so logical 0
    # moves from physical 2 to physical 1.
    assert layout.apply_swaps(((2, 1),)) == Layout((1, 0, 2))
    with pytest.raises(ValueError, match="outside a 3-wire layout"):
        layout.apply_swaps(((0, 3),))


def test_layout_reorders_per_wire_values_into_logical_order() -> None:
    layout = Layout((2, 0, 1))
    # Physical wire 0 carries logical 1, physical 1 carries logical 2, and
    # physical 2 carries logical 0.
    assert layout.to_logical_order("abc") == ("c", "a", "b")

    with pytest.raises(ValueError, match="one value per physical wire"):
        layout.to_logical_order("ab")


def test_apply_layout_relabels_every_wire_the_program_names() -> None:
    program = CircuitIR(
        3,
        (Instruction("cx", (0, 2)), Instruction("ry", (1,), params={"theta": 0.5})),
        observables=(ObservableNode("z", (0,)),),
        measurements=(MeasurementNode("probability", (1, 2)),),
    )
    layout = Layout((2, 0, 1))

    moved = apply_layout(program, layout)

    assert moved.n_wires == program.n_wires
    assert [instruction.wires for instruction in moved] == [(2, 1), (0,)]
    assert [node.wires for node in moved.observables] == [(2,)]
    assert [node.wires for node in moved.measurements] == [(0, 1)]
    assert moved.metadata["layout"] == {
        "n_wires": 3,
        "logical_to_physical": (2, 0, 1),
        "physical_to_logical": (1, 2, 0),
    }
    assert program.metadata == {}


@pytest.mark.parametrize("n_wires", (4, 5))
def test_apply_layout_preserves_the_state_up_to_the_returned_layout(
    n_wires: int,
) -> None:
    program = _random_two_wire_program(
        seed=n_wires, n_wires=n_wires, gate_count=4 * n_wires
    )
    layout = Layout(tuple(reversed(range(n_wires))))

    moved = apply_layout(program, layout)

    assert torch.allclose(
        fq.Circuit.from_ir(program).state(),
        fq.Circuit.from_ir(moved).state()[..., _amplitude_permutation(layout)],
        atol=1e-6,
    )


def test_apply_layout_fails_closed_on_a_layout_that_does_not_cover_the_circuit() -> (
    None
):
    program = CircuitIR(3, (Instruction("cx", (0, 2)),))

    with pytest.raises(ValueError, match="must cover exactly the circuit"):
        apply_layout(program, Layout((0, 1)))
    with pytest.raises(TypeError, match="must be a Layout"):
        apply_layout(program, (0, 1, 2))  # type: ignore[arg-type]


def test_layout_carries_idle_physical_slots_of_a_wider_device() -> None:
    layout = Layout((4, 5, 6, 7), 8)

    assert layout.n_wires == 4
    assert layout.physical_slot_count == 8
    assert layout.has_idle_slots is True
    # Physical wire 4 carries logical 0, and physical wires 0..3 carry nothing.
    assert layout.physical_to_logical == (None, None, None, None, 0, 1, 2, 3)
    assert layout.logical_of(4) == 0
    assert layout.map_wires((0, 3)) == (4, 7)
    assert layout.to_logical_order("abcdefgh") == ("e", "f", "g", "h")
    assert str(layout) == "Layout((4, 5, 6, 7), 8)"

    # The device width is what makes a placement legal, not the program width.
    assert Layout((4, 5, 6, 7), 8).physical_slot_count == 8
    with pytest.raises(ValueError, match="only 3 physical slots"):
        Layout((4, 5, 6, 7), 3)
    with pytest.raises(ValueError, match="outside its 8 physical slots"):
        Layout((0, 8), 8)
    with pytest.raises(ValueError, match="its own physical wire"):
        Layout((4, 4), 8)
    with pytest.raises(ValueError, match="must be an integer"):
        Layout((0, 1), 8.0)  # type: ignore[arg-type]
    # Without a declared width there is nowhere to leave idle, so the placement
    # still has to be a permutation of the program's own wires.
    with pytest.raises(ValueError, match="permutation"):
        Layout((4, 5, 6, 7))
    with pytest.raises(ValueError, match="has only -1 physical slots"):
        Layout((0, 1, 2), -1)


def test_layout_distinguishes_an_idle_slot_from_a_wire_outside_the_device() -> None:
    layout = Layout((4,), 6)

    with pytest.raises(ValueError, match="physical wire 0 holds no logical wire"):
        layout.logical_of(0)
    with pytest.raises(ValueError, match="physical wire 6 is outside the layout"):
        layout.logical_of(6)
    with pytest.raises(ValueError, match="outside the layout"):
        layout.physical_of(1)
    assert layout.to_logical_order("abcdef") == ("e",)
    with pytest.raises(ValueError, match="for 6 wires"):
        layout.to_logical_order("abc")


def test_layout_swaps_a_logical_wire_into_and_out_of_an_idle_slot() -> None:
    layout = Layout((4, 5), 6)

    # A SWAP against an idle slot moves the wire the live position holds, which is
    # how an allocated route reaches its workspace.
    moved = layout.apply_swaps(((0, 4),))
    assert moved == Layout((0, 5), 6)
    assert moved.physical_to_logical == (0, None, None, None, None, 1)
    assert moved.apply_swaps(((0, 4),)) == layout
    with pytest.raises(ValueError, match="outside a 6-wire layout"):
        layout.apply_swaps(((0, 6),))


def test_apply_layout_refuses_a_layout_that_leaves_slots_idle() -> None:
    program = CircuitIR(3, (Instruction("cx", (0, 2)),))

    # A program cannot name a wire outside its own width, so placing it on a
    # layout with idle slots would leave the assignment unreadable rather than
    # partly applied.
    with pytest.raises(ValueError, match="must cover exactly the circuit"):
        apply_layout(program, Layout((4, 5, 6), 8))
    with pytest.raises(ValueError, match="must cover exactly the circuit"):
        apply_layout(program, Layout((0, 1)))
    assert apply_layout(program, Layout((2, 0, 1))).n_wires == 3


def test_final_layout_reports_the_device_width_of_an_allocating_route() -> None:
    device = DirectedCouplingMap(8, _weakly_directed(CouplingMap.line(8)))
    program = CircuitIR(
        4,
        (
            Instruction("cx", (0, 1)),
            Instruction("cx", (1, 2)),
            Instruction("cx", (2, 3)),
        ),
        measurements=(MeasurementNode("samples", (0, 1, 2, 3)),),
    )

    routed = route_to_directed_topology(program, device, initial_layout=(4, 5, 6, 7))
    layout = final_layout(routed)

    # The routed program is widened to the device, so the layout has to report the
    # device width and say which of its slots hold nothing.
    assert routed.n_wires == 8
    assert layout == Layout((4, 5, 6, 7), 8)
    assert layout.physical_slot_count == 8
    assert layout.has_idle_slots is True
    assert layout.logical_of(4) == 0
    assert layout.to_logical_order(tuple(range(8)))[:4] == (4, 5, 6, 7)
    assert layout.map_wires((0, 1, 2, 3)) == (4, 5, 6, 7)

    # A route that never allocates stays on the three wires the program owns, even
    # on a wider device, so it reports no idle slot for the wires it never used.
    narrow = final_layout(
        route_to_topology(
            CircuitIR(2, (Instruction("cx", (0, 1)),)), CouplingMap.line(5)
        )
    )
    assert narrow == Layout((0, 1))
    assert narrow.has_idle_slots is False


def test_remove_layout_restore_reports_the_device_width_of_an_allocating_route() -> (
    None
):
    device = DirectedCouplingMap(8, _weakly_directed(CouplingMap.line(8)))
    program = CircuitIR(
        4,
        tuple(
            Instruction("cx", wires)
            for wires in ((0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3))
        ),
        measurements=(MeasurementNode("samples", (0, 1, 2, 3)),),
    )

    routed = route_to_directed_topology(program, device, initial_layout=(1, 2, 4, 5))
    dropped = remove_layout_restore(routed)
    layout = final_layout(dropped)

    assert dropped.metadata["routing"]["removed_layout_restore_swap_count"] > 0
    assert dropped.metadata["routing"]["mapping_restored"] is False
    assert layout.physical_slot_count == 8
    assert layout.has_idle_slots is True
    assert layout.n_wires == 4
    assert sorted(layout.logical_to_physical) != list(range(4))
    # The restore walked every logical wire home, so the layout the reduced
    # program reports has to be the one its SWAP evidence reached.
    assert layout.logical_to_physical == tuple(
        dropped.metadata["routing"]["final_logical_to_physical"]
    )
    # Renaming the restore away has to move every statement of the output layout,
    # or the metadata would carry two answers to one question.
    assert dropped.metadata["routing"]["logical_result_physical_slots"] == (
        layout.logical_to_physical
    )
    assert dropped.metadata["routing"]["final_physical_to_logical"] == (
        layout.physical_to_logical
    )
    # The routed program named its output on the layout the route declared, and
    # the reduced program leaves the same outputs on its own layout, so the output
    # node names the slots the pre-restore layout uses.
    assert [node.wires for node in dropped.measurements] == [
        tuple(layout.logical_to_physical[logical] for logical in range(program.n_wires))
    ]


def test_final_layout_fails_closed_when_the_reported_widths_disagree() -> None:
    device = DirectedCouplingMap(8, _weakly_directed(CouplingMap.line(8)))
    program = CircuitIR(
        4,
        (Instruction("cx", (0, 1)), Instruction("cx", (1, 2))),
        measurements=(MeasurementNode("samples", (0, 1, 2, 3)),),
    )
    routed = route_to_directed_topology(program, device, initial_layout=(1, 2, 4, 5))

    def tampered(**changes: object) -> CircuitIR:
        routing = dict(routed.metadata["routing"]) | changes
        return CircuitIR(
            routed.n_wires,
            routed.instructions,
            metadata=dict(routed.metadata) | {"routing": routing},
        )

    # A width or a wire count the program cannot hold is read as tampering rather
    # than as a layout.
    with pytest.raises(ValueError, match="physical_slot_count as 3"):
        final_layout(tampered(physical_slot_count=3))
    with pytest.raises(ValueError, match="does not fit a 8-wire program"):
        final_layout(tampered(logical_wire_count=9))
    with pytest.raises(ValueError, match="physical_slot_count as 9"):
        final_layout(tampered(physical_slot_count=9))

    # A route that never allocates reports no width, so there its placement alone
    # has to fit the program.
    narrow = route_to_topology(
        CircuitIR(3, (Instruction("cx", (0, 2)),)), CouplingMap.line(3)
    )
    routing = dict(narrow.metadata["routing"]) | {
        "final_logical_to_physical": (0, 1, 3)
    }
    with pytest.raises(ValueError, match="outside its 3 physical slots"):
        final_layout(
            CircuitIR(
                narrow.n_wires,
                narrow.instructions,
                metadata=dict(narrow.metadata) | {"routing": routing},
            )
        )


def test_remove_layout_restore_preserves_the_state_of_an_allocating_route() -> None:
    program = CircuitIR(
        4,
        (
            Instruction("cx", (0, 1)),
            Instruction("cx", (1, 2)),
            Instruction("cx", (2, 3)),
            Instruction("cx", (3, 0)),
            Instruction("ry", (2,), params={"theta": 0.9}),
        ),
        measurements=(MeasurementNode("samples", (0, 1, 2, 3)),),
    )
    n_wires = 8
    device = DirectedCouplingMap(
        8, _weakly_directed(CouplingMap(8, [(2, 4), *CouplingMap.line(8).edges]))
    )

    routed = route_to_directed_topology(program, device, initial_layout=(1, 2, 4, 5))
    dropped = remove_layout_restore(routed)
    layout = final_layout(dropped)

    # The workspace is cleaned, so every idle slot is back in |0> and the reduced
    # program has to reproduce the routed state on the slots it occupies.
    idle = [
        physical
        for physical, logical in enumerate(layout.physical_to_logical)
        if logical is None
    ]
    assert len(idle) == n_wires - program.n_wires
    assert torch.allclose(
        _occupied_state(
            routed, tuple(routed.metadata["routing"]["final_logical_to_physical"]), idle
        ),
        _occupied_state(dropped, layout.logical_to_physical, idle),
        atol=1e-6,
    )
    assert dropped.metadata["routing"]["removed_layout_restore_swap_count"] > 0


def _occupied_state(
    program: CircuitIR,
    placement: tuple[int, ...],
    idle: list[int],
) -> torch.Tensor:
    """Read a widened program's state on the slots it occupies, in logical order.

    Wire 0 is the most significant basis bit, so dropping an idle wire means
    selecting index 0 on it, and the surviving axes then have to be permuted from
    ascending physical order into logical order.
    """

    tensor = fq.Circuit.from_ir(program).state().reshape((2,) * program.n_wires)
    for wire in sorted(idle, reverse=True):
        tensor = tensor.select(wire, 0)
    ascending = sorted(placement)
    return tensor.permute(
        [ascending.index(placement[logical]) for logical in range(len(placement))]
    ).reshape(-1)


def test_apply_layout_rejects_a_layout_that_is_not_a_permutation() -> None:
    program = CircuitIR(3, (Instruction("cx", (0, 2)),))

    # A permutation over a different wire count cannot silently reach the
    # circuit: the layout is validated on construction.
    with pytest.raises(ValueError, match="permutation"):
        apply_layout(program, Layout((0, 1, 3)))


def test_final_layout_reads_the_reported_output_layout() -> None:
    program = _random_two_wire_program(seed=0, n_wires=9, gate_count=36)

    with pytest.raises(ValueError, match="no routing metadata"):
        final_layout(program)

    routed = route_to_topology(program, CouplingMap.grid(3, 3), strategy="sabre_layout")
    layout = final_layout(routed)

    assert layout.n_wires == 9
    # Routing always walks every logical wire home, so the output layout is the
    # identity and a measurement keeps naming a logical wire.
    assert layout.logical_to_physical == tuple(range(9))
    assert layout == Layout(tuple(range(9)))


def test_final_layout_fails_closed_on_routing_metadata_that_reports_no_layout() -> None:
    program = CircuitIR(2, (Instruction("cx", (0, 1)),))
    broken = CircuitIR(
        program.n_wires,
        program.instructions,
        metadata={"routing": {"final_logical_to_physical": None}},
    )

    with pytest.raises(ValueError, match="does not report final_logical_to_physical"):
        final_layout(broken)
    with pytest.raises(ValueError, match="for 1 wires, but the circuit has 2"):
        final_layout(
            CircuitIR(
                program.n_wires,
                program.instructions,
                metadata={"routing": {"final_logical_to_physical": (0,)}},
            )
        )
    with pytest.raises(ValueError, match="invalid final_logical_to_physical"):
        final_layout(
            CircuitIR(
                program.n_wires,
                program.instructions,
                metadata={"routing": {"final_logical_to_physical": (0, 0)}},
            )
        )


def test_remove_layout_restore_drops_only_the_appended_restore_phase() -> None:
    program = _random_two_wire_program(seed=0, n_wires=9, gate_count=36)

    routed = route_to_topology(program, CouplingMap.grid(3, 3), strategy="sabre_layout")
    routing = routed.metadata["routing"]
    dropped = remove_layout_restore(routed)
    dropped_routing = dropped.metadata["routing"]

    removed = routing["inserted_swap_count"] - dropped_routing["inserted_swap_count"]
    assert removed > 0
    assert dropped_routing["removed_layout_restore_swap_count"] == removed
    assert dropped_routing["final_logical_to_physical"] == tuple(
        routing["pre_restore_logical_to_physical"]
    )
    assert dropped_routing["final_logical_to_physical"] != tuple(range(9))
    assert dropped_routing["mapping_restored"] is False
    assert (
        dropped_routing["inserted_swap_count"]
        == dropped_routing["planned_inserted_swap_count"]
    )
    assert not any(
        instruction.metadata.get("routing_phase") == "final_restore"
        for instruction in dropped
    )
    assert len(dropped) == len(routed) - removed
    # Removing the restore keeps every non-restore instruction, in order.
    assert [
        instruction
        for instruction in routed
        if instruction.metadata.get("routing_phase") != "final_restore"
    ] == list(dropped)


@pytest.mark.parametrize(
    ("rows", "columns", "seed"),
    ((3, 3, 0), (3, 3, 1), (4, 4, 0), (5, 5, 1)),
)
def test_remove_layout_restore_preserves_the_state_under_the_output_layout(
    rows: int,
    columns: int,
    seed: int,
) -> None:
    n_wires = rows * columns
    program = _random_two_wire_program(
        seed=seed, n_wires=n_wires, gate_count=4 * n_wires
    )

    routed = route_to_topology(
        program, CouplingMap.grid(rows, columns), strategy="sabre_layout"
    )
    dropped = remove_layout_restore(routed)
    layout = final_layout(dropped)

    assert torch.allclose(
        fq.Circuit.from_ir(routed).state(),
        fq.Circuit.from_ir(dropped).state()[..., _amplitude_permutation(layout)],
        atol=1e-6,
    )
    # A backend running the reduced program only needs the reported layout to
    # read the logical outcome, so the saving must be real.
    assert dropped.metadata["routing"]["removed_layout_restore_swap_count"] > 0


def test_remove_layout_restore_moves_measurements_and_observables_onto_the_layout() -> (
    None
):
    program = CircuitIR(
        4,
        (
            Instruction("cx", (0, 3)),
            Instruction("cx", (1, 2)),
            Instruction("cx", (0, 1)),
        ),
        observables=(ObservableNode("z", (0,)), ObservableNode("z", (2,))),
        measurements=(MeasurementNode("probability", (0, 1, 2, 3)),),
    )

    routed = route_to_topology(program, CouplingMap.grid(1, 4), strategy="sabre_layout")
    dropped = remove_layout_restore(routed)
    layout = final_layout(dropped)

    assert [node.wires for node in dropped.measurements] == [
        layout.map_wires((0, 1, 2, 3))
    ]
    assert [node.wires for node in dropped.observables] == [
        layout.map_wires((0,)),
        layout.map_wires((2,)),
    ]


def test_remove_layout_restore_is_the_identity_on_a_program_without_a_restore() -> None:
    program = _random_two_wire_program(seed=0, n_wires=4, gate_count=8)

    with pytest.raises(ValueError, match="no routing metadata"):
        remove_layout_restore(program)

    # ``sabre`` restores by replaying its forward SWAPs in reverse, so the phase
    # that walks the layout home is still present and is still removable.
    routed = route_to_topology(program, CouplingMap.line(4), strategy="sabre")
    dropped = remove_layout_restore(routed)
    assert dropped.metadata["routing"]["removed_layout_restore_swap_count"] > 0

    # Removing it twice is the identity: there is no restore left to remove.
    assert remove_layout_restore(dropped) is dropped


def test_remove_layout_restore_leaves_restore_after_each_gate_untouched() -> None:
    program = _random_two_wire_program(seed=0, n_wires=4, gate_count=8)

    routed = route_to_topology(
        program, CouplingMap.line(4), strategy="restore_after_each_gate"
    )

    # That strategy restores after every gate, so its SWAPs are interleaved with
    # the program and there is no appended restore to remove. Removing none of
    # them is the only safe answer; stripping a ``gate_restore`` SWAP would move
    # a logical wire off the wire the next gate expects.
    assert any(
        instruction.metadata.get("routing_phase") == "gate_restore"
        for instruction in routed
    )
    assert not any(
        instruction.metadata.get("routing_phase") == "final_restore"
        for instruction in routed
    )
    assert remove_layout_restore(routed) == routed


def test_remove_layout_restore_reports_output_the_deployment_contract_refuses() -> None:
    program = _random_two_wire_program(seed=0, n_wires=9, gate_count=36)
    coupling_map = CouplingMap.grid(3, 3)

    routed = route_to_topology(program, coupling_map, strategy="persistent_layout")
    dropped = remove_layout_restore(routed)

    assert dropped is not routed
    # The routed plan is acceptable, so the refusal below is about the removed
    # restore and not about the strategy or the schema.
    validate_deployment_routing_plan(
        routed.metadata["routing"], n_wires=9, coupling_map=coupling_map
    )
    with pytest.raises(
        DeploymentRoutingEvidenceError,
        match="must restore the final logical permutation",
    ):
        validate_deployment_routing_plan(
            dropped.metadata["routing"], n_wires=9, coupling_map=coupling_map
        )


def test_remove_layout_restore_fails_closed_when_metadata_contradicts_the_swaps() -> (
    None
):
    program = _random_two_wire_program(seed=0, n_wires=4, gate_count=8)

    routed = route_to_topology(program, CouplingMap.line(4), strategy="sabre")
    routing = dict(routed.metadata["routing"])
    routing["pre_restore_logical_to_physical"] = tuple(range(4))
    tampered = CircuitIR(
        routed.n_wires,
        routed.instructions,
        metadata=dict(routed.metadata) | {"routing": routing},
    )

    with pytest.raises(ValueError, match="does not agree with the SWAP evidence"):
        remove_layout_restore(tampered)


def test_remove_layout_restore_fails_closed_on_a_non_swap_routed_instruction() -> None:
    program = _random_two_wire_program(seed=0, n_wires=4, gate_count=8)
    routed = route_to_topology(program, CouplingMap.line(4), strategy="sabre")

    # The phase marker is the removal's only evidence of which instructions the
    # restore appended, so a marker on a non-SWAP must be rejected rather than
    # dropped.
    instructions = list(routed.instructions[:-1]) + [
        Instruction(
            "cx",
            routed.instructions[-1].wires,
            metadata={"routing_phase": "final_restore"},
        )
    ]
    tampered = CircuitIR(
        routed.n_wires,
        tuple(instructions),
        metadata=routed.metadata,
    )

    with pytest.raises(ValueError, match="invalid SWAP evidence"):
        remove_layout_restore(tampered)
