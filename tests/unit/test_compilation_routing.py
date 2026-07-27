import pytest
import torch

import flagquantum as fq
from flagquantum.compilation.routing import CouplingMap, route_to_topology
from flagquantum.core.ir import CircuitIR, Instruction

pytestmark = pytest.mark.unit


def test_coupling_map_rejects_invalid_dimensions_and_wire_queries() -> None:
    with pytest.raises(ValueError, match="positive wire count"):
        CouplingMap(0, ())
    with pytest.raises(ValueError, match="dimensions must be positive"):
        CouplingMap.grid(0, 2)

    coupling = CouplingMap.line(3)
    with pytest.raises(ValueError, match="outside"):
        coupling.neighbors(3)
    with pytest.raises(ValueError, match="outside"):
        coupling.has_edge(-1, 0)
    with pytest.raises(ValueError, match="outside"):
        coupling.shortest_path(0, 3)


def test_routing_fails_closed_when_path_requires_unlowered_ancilla_wire() -> None:
    ir = CircuitIR(3, (Instruction("cx", (0, 2)),))
    coupling = CouplingMap(4, ((0, 3), (3, 2), (2, 1)))

    with pytest.raises(ValueError, match="explicit layout/lowering"):
        route_to_topology(ir, coupling)


def test_unknown_routing_strategy_is_rejected() -> None:
    ir = CircuitIR(2, (Instruction("cx", (0, 1)),))

    with pytest.raises(ValueError, match="routing strategy"):
        route_to_topology(
            ir,
            CouplingMap.line(2),
            strategy="unknown",
        )


def test_reverse_direction_and_multiple_routed_gates_preserve_state() -> None:
    circuit = fq.Circuit(4)
    circuit.x(3).h(1).cx(3, 0).ry(2, theta=0.31).cx(0, 2)
    coupling = CouplingMap.line(4)

    compiled = circuit.compile(coupling_map=coupling)

    for instruction in compiled.to_ir():
        if len(instruction.wires) == 2:
            assert coupling.has_edge(*instruction.wires)
    routed_gates = tuple(
        instruction
        for instruction in compiled.to_ir()
        if instruction.metadata.get("routed")
    )
    assert tuple(gate.metadata["logical_wires"] for gate in routed_gates) == (
        (3, 0),
        (0, 2),
    )
    routing = compiled.to_ir().metadata["routing"]
    assert routing == {
        "schema": "flagquantum_routing_plan_v1",
        "strategy": "restore_after_each_gate",
        "coupling_n_wires": 4,
        "coupling_edges": ((0, 1), (1, 2), (2, 3)),
        "initial_logical_to_physical": (0, 1, 2, 3),
        "pre_restore_logical_to_physical": (0, 1, 2, 3),
        "final_logical_to_physical": (0, 1, 2, 3),
        "mapping_restored": True,
        "direction_semantics": "logical_wire_order_preserved",
        "topology_gate_count": 2,
        "routed_gate_count": 2,
        "inserted_swap_count": 6,
        "planned_inserted_swap_count": 6,
        "post_optimization_inserted_swap_count": 6,
        "post_optimization_instruction_count": 11,
        "skipped_channel_count": 0,
        "persistent_layout_supported": True,
        "path_cache": {
            "before": {
                "capacity": 4096,
                "size": 0,
                "hits": 0,
                "misses": 0,
                "evictions": 0,
            },
            "after": {
                "capacity": 4096,
                "size": 4,
                "hits": 0,
                "misses": 2,
                "evictions": 0,
            },
            "delta": {
                "hits": 0,
                "misses": 2,
                "evictions": 0,
            },
        },
    }
    assert torch.allclose(compiled.state(), circuit.state(), atol=1e-6)


def test_persistent_layout_reuses_mapping_and_restores_final_permutation() -> None:
    circuit = fq.Circuit(4)
    circuit.x(3).cx(0, 3).h(0).cx(0, 3)
    coupling = CouplingMap.line(4)

    restored = route_to_topology(
        circuit,
        coupling,
        strategy="restore_after_each_gate",
    )
    persistent = route_to_topology(
        circuit,
        coupling,
        strategy="persistent_layout",
    )
    routing = persistent.metadata["routing"]

    assert routing["pre_restore_logical_to_physical"] == (2, 0, 1, 3)
    assert routing["final_logical_to_physical"] == (0, 1, 2, 3)
    assert routing["mapping_restored"] is True
    assert routing["inserted_swap_count"] == 4
    assert (
        routing["inserted_swap_count"]
        < restored.metadata["routing"]["inserted_swap_count"]
    )
    assert torch.allclose(
        fq.Circuit.from_ir(persistent).state(),
        circuit.state(),
        atol=1e-6,
    )

    compiled = circuit.compile(
        coupling_map=coupling,
        routing_strategy="persistent_layout",
    )
    assert compiled.to_ir().metadata["routing"]["strategy"] == "persistent_layout"
    assert torch.allclose(compiled.state(), circuit.state(), atol=1e-6)
