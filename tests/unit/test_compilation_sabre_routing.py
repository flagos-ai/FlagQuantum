import random

import pytest
import torch

import flagquantum as fq
from flagquantum.compiler import CouplingMap, route_to_topology
from flagquantum.core.ir import CircuitIR, Instruction, MeasurementNode

pytestmark = pytest.mark.unit


def _random_circuit(seed: int) -> fq.Circuit:
    rng = random.Random(seed)
    circuit = fq.Circuit(4)
    circuit.h(seed % 4).cx(0, 3)
    for _ in range(12):
        gate = rng.choice(("h", "x", "rx", "ry", "rz", "cx", "cz"))
        if gate in {"h", "x"}:
            getattr(circuit, gate)(rng.randrange(4))
        elif gate in {"rx", "ry", "rz"}:
            getattr(circuit, gate)(
                rng.randrange(4),
                theta=rng.uniform(-0.8, 0.8),
            )
        else:
            left, right = rng.sample(range(4), 2)
            getattr(circuit, gate)(left, right)
    return circuit


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


def test_sabre_random_circuits_match_original_state() -> None:
    for seed in range(8):
        circuit = _random_circuit(seed)
        coupling = CouplingMap.line(4)

        routed = route_to_topology(circuit, coupling, strategy="sabre")

        assert all(
            len(instruction.wires) != 2
            or instruction.metadata.get("is_channel")
            or coupling.has_edge(*instruction.wires)
            for instruction in routed
        )
        assert torch.allclose(
            fq.Circuit.from_ir(routed).state(),
            circuit.state(),
            atol=1e-6,
        )


def test_sabre_parameter_gradients_match_original() -> None:
    def circuit(theta: torch.Tensor, phi: torch.Tensor) -> fq.Circuit:
        program = fq.Circuit(4)
        program.h(3).ry(0, theta=theta).cx(0, 3)
        program.rz(0, theta=phi).cx(0, 2).ry(3, theta=theta + phi)
        return program

    theta = torch.tensor(0.23, requires_grad=True)
    phi = torch.tensor(-0.17, requires_grad=True)
    routed_theta = theta.detach().clone().requires_grad_(True)
    routed_phi = phi.detach().clone().requires_grad_(True)

    reference = circuit(theta, phi).state()
    routed_ir = route_to_topology(
        circuit(routed_theta, routed_phi),
        CouplingMap.line(4),
        strategy="sabre",
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


def test_sabre_remaps_channel_but_restores_measurement_layout() -> None:
    channel = Instruction(
        "test_two_wire_channel",
        (0, 3),
        metadata={"is_channel": True},
    )
    ir = CircuitIR(
        4,
        (
            Instruction("cx", (0, 3)),
            channel,
        ),
        measurements=(MeasurementNode("sample", (0, 3), shots=10),),
    )

    routed = route_to_topology(ir, CouplingMap.line(4), strategy="sabre")
    routed_channel = next(
        instruction for instruction in routed if instruction.metadata.get("is_channel")
    )

    assert routed_channel.metadata["logical_wires"] == (0, 3)
    assert routed.measurements == ir.measurements
    assert routed.metadata["routing"]["skipped_channel_count"] == 1
    assert routed.metadata["routing"]["mapping_restored"] is True


@pytest.mark.parametrize(
    "coupling",
    (
        CouplingMap.ring(4),
        CouplingMap.grid(2, 2),
        CouplingMap(4, ((0, 1), (1, 2), (1, 3))),
    ),
    ids=("ring", "grid", "irregular"),
)
def test_sabre_supports_non_line_topologies_and_gate_families(
    coupling: CouplingMap,
) -> None:
    circuit = fq.Circuit(4)
    circuit.h(0).swap(0, 3).crx(3, 2, theta=0.27)
    circuit.rxx(0, 2, theta=-0.31).rzz(3, 1, theta=0.19)

    routed = route_to_topology(circuit, coupling, strategy="sabre")

    assert all(
        len(instruction.wires) != 2
        or instruction.metadata.get("is_channel")
        or coupling.has_edge(*instruction.wires)
        for instruction in routed
    )
    assert routed.metadata["routing"]["mapping_restored"] is True
    assert torch.allclose(
        fq.Circuit.from_ir(routed).state(),
        circuit.state(),
        atol=1e-6,
    )


def test_sabre_uses_fewer_swaps_than_the_shortest_path_strategies() -> None:
    coupling = CouplingMap.grid(3, 3)
    program = _random_two_wire_program(seed=11, n_wires=9, gate_count=30)

    sabre = route_to_topology(program, coupling, strategy="sabre")
    persistent = route_to_topology(program, coupling, strategy="persistent_layout")
    restored = route_to_topology(program, coupling, strategy="restore_after_each_gate")

    routing = sabre.metadata["routing"]
    assert routing["strategy"] == "sabre"
    assert routing["inserted_swap_count"] == 30
    assert (
        routing["inserted_swap_count"]
        < persistent.metadata["routing"]["inserted_swap_count"]
    )
    assert (
        routing["inserted_swap_count"]
        < restored.metadata["routing"]["inserted_swap_count"]
    )
    assert len(sabre) == len(program) + routing["inserted_swap_count"]


def test_sabre_plan_is_reproducible_and_places_every_instruction_once() -> None:
    coupling = CouplingMap.grid(3, 3)
    program = _random_two_wire_program(seed=5, n_wires=9, gate_count=40)

    first = route_to_topology(program, coupling, strategy="sabre")
    second = route_to_topology(program, coupling, strategy="sabre")

    assert first == second
    source_indices = [
        instruction.metadata["source_instruction_index"]
        for instruction in first
        if instruction.metadata.get("routing_phase") is None
    ]
    assert sorted(source_indices) == list(range(len(program)))
    forward = [
        instruction
        for instruction in first
        if instruction.metadata.get("routing_phase") == "forward"
    ]
    restore = [
        instruction
        for instruction in first
        if instruction.metadata.get("routing_phase") == "final_restore"
    ]
    assert len(forward) == 26
    # The restore is not required to be the forward SWAPs in reverse: a restore
    # that walks the final layout home along coupling edges can be shorter. What
    # matters is that the restore is legal, that replaying it reaches the identity
    # layout, and that the reported count matches the emitted instructions.
    assert first.metadata["routing"]["inserted_swap_count"] == len(forward) + len(
        restore
    )
    assert all(coupling.has_edge(*instruction.wires) for instruction in restore)
    assert _replayed_output_layout(
        first, first.metadata["routing"]["initial_logical_to_physical"]
    ) == (0, 1, 2, 3, 4, 5, 6, 7, 8)


def test_sabre_reports_a_moved_layout_and_a_restored_output_layout() -> None:
    coupling = CouplingMap.grid(2, 2)
    program = _random_two_wire_program(seed=3, n_wires=4, gate_count=24)

    routing = route_to_topology(program, coupling, strategy="sabre").metadata["routing"]

    assert routing["pre_restore_logical_to_physical"] != (0, 1, 2, 3)
    assert routing["final_logical_to_physical"] == (0, 1, 2, 3)
    assert routing["mapping_restored"] is True
    assert routing["planned_inserted_swap_count"] == routing["inserted_swap_count"]
    assert routing["coupling_edges"] == coupling.edges


def test_sabre_fails_closed_on_an_undersized_coupling_map() -> None:
    program = _random_two_wire_program(seed=1, n_wires=5, gate_count=4)

    with pytest.raises(ValueError, match="fewer qubits"):
        route_to_topology(program, CouplingMap.line(4), strategy="sabre")


def test_sabre_fails_closed_when_the_program_needs_a_disconnected_pair() -> None:
    program = CircuitIR(4, (Instruction("cx", (0, 3)),))

    with pytest.raises(ValueError, match="no coupling path"):
        route_to_topology(
            program,
            CouplingMap(4, ((0, 1), (2, 3))),
            strategy="sabre",
        )


def test_sabre_fails_closed_when_only_wires_outside_the_program_connect_it() -> None:
    # The program owns {0, 1}, and the device reaches 1 from 0 only through 2.
    device = CouplingMap(4, ((0, 2), (1, 2), (1, 3)))
    program = CircuitIR(2, (Instruction("cx", (0, 1)),))

    with pytest.raises(ValueError, match="ancilla wires"):
        route_to_topology(program, device, strategy="sabre")


def test_sabre_routes_a_program_narrower_than_the_device() -> None:
    device = CouplingMap.grid(4, 4)
    program = _random_two_wire_program(seed=3, n_wires=6, gate_count=30)

    routed = route_to_topology(program, device, strategy="sabre")

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


def test_sabre_on_a_wider_device_routes_like_its_induced_subgraph() -> None:
    # A padding wire of a wider device is not an ancilla the planner may swap on,
    # so the plan must be a function of the subgraph the program's wires induce.
    device = CouplingMap.grid(4, 4)
    program = _random_two_wire_program(seed=3, n_wires=6, gate_count=30)
    induced = CouplingMap(
        program.n_wires,
        tuple(edge for edge in device.edges if edge[1] < program.n_wires),
    )

    wider = route_to_topology(program, device, strategy="sabre")
    subgraph = route_to_topology(program, induced, strategy="sabre")

    assert tuple((item.name, item.wires) for item in wider) == tuple(
        (item.name, item.wires) for item in subgraph
    )
    assert (
        wider.metadata["routing"]["inserted_swap_count"]
        == subgraph.metadata["routing"]["inserted_swap_count"]
    )


def test_sabre_uses_a_bounded_number_of_swaps_on_a_larger_device() -> None:
    coupling = CouplingMap.grid(4, 4)
    program = _random_two_wire_program(seed=7, n_wires=16, gate_count=60)

    routed = route_to_topology(program, coupling, strategy="sabre")

    assert routed.metadata["routing"]["inserted_swap_count"] == 92
    assert all(
        len(instruction.wires) != 2
        or instruction.metadata.get("is_channel")
        or coupling.has_edge(*instruction.wires)
        for instruction in routed
    )
    assert torch.allclose(
        fq.Circuit.from_ir(routed).state(),
        fq.Circuit.from_ir(program).state(),
        atol=1e-6,
    )
