import random

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler import CouplingMap, route_to_topology
from flagquantum.compiler.sabre import plan_sabre_layout, plan_sabre_swaps
from flagquantum.core.ir import CircuitIR, Instruction, MeasurementNode

pytestmark = pytest.mark.unit


def _random_two_wire_program(seed: int, n_wires: int, gate_count: int) -> CircuitIR:
    rng = random.Random(seed)
    instructions = []
    for _ in range(gate_count):
        left, right = rng.sample(range(n_wires), 2)
        instructions.append(Instruction("cx", (left, right)))
    return CircuitIR(n_wires, tuple(instructions))


def _replayed_output_layout(
    routed: CircuitIR,
    initial_layout: tuple[int, ...],
) -> tuple[int, ...]:
    """Replay the routed SWAPs and return the logical-to-physical layout.

    Counting the SWAPs is not enough to show the output layout was restored: the
    numbered SWAPs alone must transform ``initial_layout`` into the reported final
    layout, which is what a backend executing the routed circuit will see.
    """

    logical_to_physical = list(initial_layout)
    physical_to_logical = [0] * routed.n_wires
    for logical, physical in enumerate(logical_to_physical):
        physical_to_logical[physical] = logical
    for instruction in routed:
        if instruction.metadata.get("routing_phase") is None:
            continue
        left, right = instruction.wires
        left_logical = physical_to_logical[left]
        right_logical = physical_to_logical[right]
        physical_to_logical[left], physical_to_logical[right] = (
            right_logical,
            left_logical,
        )
        logical_to_physical[left_logical] = right
        logical_to_physical[right_logical] = left
    return tuple(logical_to_physical)


@pytest.mark.parametrize(
    ("rows", "columns", "seed", "sabre_swaps", "layout_swaps", "moves_layout"),
    (
        (3, 3, 0, 28, 24, True),
        # The search keeps the identity layout when no candidate improves on it,
        # so a layout pass can never cost more than plain SABRE routing.
        (3, 3, 1, 26, 26, False),
        (4, 4, 0, 98, 93, True),
        (4, 4, 1, 108, 96, True),
        (5, 5, 0, 220, 199, True),
        (5, 5, 2, 234, 191, True),
        (6, 6, 0, 410, 388, True),
        (6, 6, 1, 436, 361, True),
    ),
)
def test_sabre_layout_is_never_worse_than_sabre_routing(
    rows: int,
    columns: int,
    seed: int,
    sabre_swaps: int,
    layout_swaps: int,
    moves_layout: bool,
) -> None:
    coupling = CouplingMap.grid(rows, columns)
    program = _random_two_wire_program(
        seed=seed,
        n_wires=rows * columns,
        gate_count=4 * rows * columns,
    )

    sabre = route_to_topology(program, coupling, strategy="sabre")
    layout = route_to_topology(program, coupling, strategy="sabre_layout")

    assert sabre.metadata["routing"]["inserted_swap_count"] == sabre_swaps
    assert layout.metadata["routing"]["inserted_swap_count"] == layout_swaps
    assert layout.metadata["routing"]["inserted_swap_count"] <= sabre_swaps
    assert (
        layout.metadata["routing"]["initial_logical_to_physical"]
        != tuple(range(rows * columns))
    ) is moves_layout
    assert all(
        len(instruction.wires) != 2 or coupling.has_edge(*instruction.wires)
        for instruction in layout
    )


def test_sabre_layout_moves_the_initial_layout_and_restores_the_output_layout() -> None:
    coupling = CouplingMap.grid(4, 4)
    program = _random_two_wire_program(seed=0, n_wires=16, gate_count=64)

    routed = route_to_topology(program, coupling, strategy="sabre_layout")
    routing = routed.metadata["routing"]

    assert routing["strategy"] == "sabre_layout"
    assert routing["schema"] == "flagquantum_routing_plan_v1"
    assert routing["initial_logical_to_physical"] != tuple(range(16))
    assert sorted(routing["initial_logical_to_physical"]) == list(range(16))
    assert (
        routing["pre_restore_logical_to_physical"]
        != routing["initial_logical_to_physical"]
    )
    assert routing["final_logical_to_physical"] == tuple(range(16))
    assert routing["mapping_restored"] is True
    assert routing["planned_inserted_swap_count"] == routing["inserted_swap_count"]
    assert routing["path_cache"]["delta"]["hits"] == 0
    # The reported counts describe the emitted instructions, and replaying the
    # emitted SWAPs reproduces the reported final layout on their own.
    assert routing["inserted_swap_count"] == sum(
        instruction.metadata.get("routing_phase") is not None for instruction in routed
    )
    assert _replayed_output_layout(
        routed, routing["initial_logical_to_physical"]
    ) == tuple(range(16))
    assert all(
        len(instruction.wires) != 2 or coupling.has_edge(*instruction.wires)
        for instruction in routed
    )


def test_sabre_layout_preserves_state_and_parameter_gradients() -> None:
    def circuit(theta: torch.Tensor, phi: torch.Tensor) -> fq.Circuit:
        program = fq.Circuit(4)
        program.h(3).ry(0, theta=theta).cx(0, 3)
        program.rz(2, theta=phi).cx(1, 3).ry(2, theta=theta + phi)
        return program

    theta = torch.tensor(0.31, requires_grad=True)
    phi = torch.tensor(-0.23, requires_grad=True)
    routed_theta = theta.detach().clone().requires_grad_(True)
    routed_phi = phi.detach().clone().requires_grad_(True)

    reference = circuit(theta, phi).state()
    routed_ir = route_to_topology(
        circuit(routed_theta, routed_phi),
        CouplingMap.grid(2, 2),
        strategy="sabre_layout",
    )
    routed = fq.Circuit.from_ir(routed_ir).state()
    weights = torch.arange(reference.numel(), dtype=reference.real.dtype).reshape(
        reference.shape
    )
    reference_loss = (reference.real * weights).sum() + (
        reference.imag * weights.flip(-1)
    ).sum()
    routed_loss = (routed.real * weights).sum() + (routed.imag * weights.flip(-1)).sum()
    reference_loss.backward()
    routed_loss.backward()

    assert torch.allclose(routed, reference, atol=1e-6)
    assert torch.allclose(routed_theta.grad, theta.grad, atol=1e-6)
    assert torch.allclose(routed_phi.grad, phi.grad, atol=1e-6)


def test_sabre_layout_preserves_state_on_a_non_line_topology() -> None:
    coupling = CouplingMap(6, ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5), (1, 4)))
    program = _random_two_wire_program(seed=6, n_wires=6, gate_count=24)

    routed = route_to_topology(program, coupling, strategy="sabre_layout")

    assert all(
        len(instruction.wires) != 2 or coupling.has_edge(*instruction.wires)
        for instruction in routed
    )
    assert torch.allclose(
        fq.Circuit.from_ir(routed).state(),
        fq.Circuit.from_ir(program).state(),
        atol=1e-6,
    )


def test_sabre_layout_remaps_channels_and_restores_measurement_layout() -> None:
    coupling = CouplingMap.grid(2, 3)
    channel = Instruction(
        "test_two_wire_channel", (0, 5), metadata={"is_channel": True}
    )
    source = CircuitIR(
        6,
        (
            Instruction("cx", (0, 5)),
            channel,
            Instruction("cx", (1, 4)),
        ),
        measurements=(MeasurementNode("sample", (0, 5), shots=10),),
    )

    routing = route_to_topology(source, coupling, strategy="sabre_layout").metadata[
        "routing"
    ]

    assert routing["skipped_channel_count"] == 1
    assert routing["topology_gate_count"] == 2
    # Counted against the source wires, as every routing strategy does: (0, 5) is
    # not a grid edge, (1, 4) is.
    assert routing["routed_gate_count"] == 1
    assert routing["final_logical_to_physical"] == tuple(range(6))


def test_sabre_layout_is_deterministic_and_places_every_instruction_once() -> None:
    coupling = CouplingMap.grid(3, 3)
    program = _random_two_wire_program(seed=4, n_wires=9, gate_count=40)

    first = route_to_topology(program, coupling, strategy="sabre_layout")
    second = route_to_topology(program, coupling, strategy="sabre_layout")

    assert first == second
    source_indices = [
        instruction.metadata["source_instruction_index"]
        for instruction in first
        if instruction.metadata.get("routing_phase") is None
    ]
    assert sorted(source_indices) == list(range(len(program)))


def test_sabre_layout_fails_closed_on_an_undersized_coupling_map() -> None:
    program = _random_two_wire_program(seed=2, n_wires=5, gate_count=4)

    with pytest.raises(ValueError, match="fewer qubits"):
        route_to_topology(program, CouplingMap.line(4), strategy="sabre_layout")


def test_sabre_layout_fails_closed_on_a_disconnected_pair() -> None:
    program = CircuitIR(4, (Instruction("cx", (0, 3)),))

    with pytest.raises(ValueError, match="no coupling path"):
        route_to_topology(
            program,
            CouplingMap(4, ((0, 1), (2, 3))),
            strategy="sabre_layout",
        )


def test_sabre_layout_fails_closed_when_only_wires_outside_the_program_connect_it() -> (
    None
):
    # The program owns {0, 1}, and the device reaches 1 from 0 only through 2.
    device = CouplingMap(4, ((0, 2), (1, 2), (1, 3)))
    program = CircuitIR(2, (Instruction("cx", (0, 1)),))

    with pytest.raises(ValueError, match="ancilla wires"):
        route_to_topology(program, device, strategy="sabre_layout")


def test_sabre_layout_routes_a_program_narrower_than_the_device() -> None:
    device = CouplingMap.grid(4, 4)
    program = _random_two_wire_program(seed=3, n_wires=6, gate_count=30)

    routed = route_to_topology(program, device, strategy="sabre_layout")

    assert routed.n_wires == program.n_wires
    assert all(
        len(instruction.wires) != 2
        or instruction.metadata.get("is_channel")
        or device.has_edge(*instruction.wires)
        for instruction in routed
    )
    assert torch.allclose(
        fq.Circuit.from_ir(routed).state(),
        fq.Circuit.from_ir(program).state(),
        atol=1e-6,
    )


def test_sabre_layout_on_a_wider_device_matches_its_induced_subgraph() -> None:
    device = CouplingMap.grid(4, 4)
    program = _random_two_wire_program(seed=3, n_wires=6, gate_count=30)
    induced = CouplingMap(
        program.n_wires,
        tuple(edge for edge in device.edges if edge[1] < program.n_wires),
    )

    wider = route_to_topology(program, device, strategy="sabre_layout")
    subgraph = route_to_topology(program, induced, strategy="sabre_layout")

    assert wider.metadata["routing"]["initial_logical_to_physical"] == (
        subgraph.metadata["routing"]["initial_logical_to_physical"]
    )
    assert tuple((item.name, item.wires) for item in wider) == tuple(
        (item.name, item.wires) for item in subgraph
    )


def test_sabre_layout_fails_closed_on_an_invalid_initial_layout() -> None:
    program = _random_two_wire_program(seed=2, n_wires=4, gate_count=4)

    with pytest.raises(ValueError, match="permutation"):
        plan_sabre_swaps(program, CouplingMap.grid(2, 2), initial_layout=(0, 0, 1, 2))


def test_sabre_layout_search_reports_the_identity_layout_when_rounds_are_disabled() -> (
    None
):
    program = _random_two_wire_program(seed=0, n_wires=9, gate_count=36)

    assert plan_sabre_layout(program, CouplingMap.grid(3, 3), rounds=0) == tuple(
        range(9)
    )

    with pytest.raises(ValueError, match="non-negative"):
        plan_sabre_layout(program, CouplingMap.grid(3, 3), rounds=-1)
