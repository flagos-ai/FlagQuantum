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


@pytest.mark.parametrize("seed", range(8))
def test_persistent_layout_random_circuits_match_original_state(seed: int) -> None:
    circuit = _random_circuit(seed)
    coupling = CouplingMap.line(4)

    routed = route_to_topology(
        circuit,
        coupling,
        strategy="persistent_layout",
    )
    routed_state = fq.Circuit.from_ir(routed).state()

    assert all(
        len(instruction.wires) != 2
        or instruction.metadata.get("is_channel")
        or coupling.has_edge(*instruction.wires)
        for instruction in routed
    )
    assert torch.allclose(routed_state, circuit.state(), atol=1e-6)


def _gradient_circuit(theta: torch.Tensor, phi: torch.Tensor) -> fq.Circuit:
    circuit = fq.Circuit(4)
    circuit.h(3).ry(0, theta=theta).cx(0, 3)
    circuit.rz(0, theta=phi).cx(0, 2).ry(3, theta=theta + phi)
    return circuit


def test_persistent_layout_parameter_gradients_match_original() -> None:
    theta = torch.tensor(0.23, requires_grad=True)
    phi = torch.tensor(-0.17, requires_grad=True)
    routed_theta = theta.detach().clone().requires_grad_(True)
    routed_phi = phi.detach().clone().requires_grad_(True)

    reference = _gradient_circuit(theta, phi).state()
    routed_ir = route_to_topology(
        _gradient_circuit(routed_theta, routed_phi),
        CouplingMap.line(4),
        strategy="persistent_layout",
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


def test_persistent_layout_remaps_channel_but_restores_measurement_layout() -> None:
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

    routed = route_to_topology(
        ir,
        CouplingMap.line(4),
        strategy="persistent_layout",
    )
    routed_channel = next(
        instruction for instruction in routed if instruction.metadata.get("is_channel")
    )

    assert routed_channel.wires == (2, 3)
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
def test_persistent_layout_supports_non_line_topologies_and_gate_families(
    coupling: CouplingMap,
) -> None:
    circuit = fq.Circuit(4)
    circuit.h(0).swap(0, 3).crx(3, 2, theta=0.27)
    circuit.rxx(0, 2, theta=-0.31).rzz(3, 1, theta=0.19)

    routed = route_to_topology(
        circuit,
        coupling,
        strategy="persistent_layout",
    )

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


def test_compiler_reports_planned_and_retained_routing_swaps_separately() -> None:
    circuit = fq.Circuit(4)
    circuit.x(3).cx(0, 3).h(3).cx(0, 3)

    compiled = circuit.compile(
        coupling_map=CouplingMap.line(4),
        routing_strategy="restore_after_each_gate",
    )
    routing = compiled.to_ir().metadata["routing"]

    assert routing["planned_inserted_swap_count"] == 8
    assert routing["post_optimization_inserted_swap_count"] == 4
    assert routing["post_optimization_instruction_count"] == len(compiled)
    assert torch.allclose(compiled.state(), circuit.state(), atol=1e-6)
