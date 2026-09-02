from __future__ import annotations

import itertools

import pytest
import torch

import flagquantum as fq
from flagquantum._compiler.exporters.circuit_ir import seal_circuit_ir_round_trip
from flagquantum._compiler.passes.manager import PassManager
from flagquantum._compiler.passes.placement_routing import (
    DirectedCouplingGraph,
    PlacementRoutingPass,
)
from flagquantum._compiler.passes.static_canonicalization import (
    StaticCanonicalizationPass,
)
from flagquantum._compiler.passes.target_decomposition import (
    DecomposeToTargetGateSetPass,
)
from flagquantum._compiler.testing.differential import lower_module_for_differential

pytestmark = pytest.mark.unit


def _route(
    source: fq.CircuitIR,
    graph: DirectedCouplingGraph,
    *,
    initial_layout: tuple[int, ...] | None = None,
):
    sealed = seal_circuit_ir_round_trip(source)
    assert sealed.ok and sealed.artifact is not None
    routing = PlacementRoutingPass(graph, initial_layout=initial_layout)
    pipeline = PassManager(
        (
            StaticCanonicalizationPass(),
            DecomposeToTargetGateSetPass(),
            routing,
        )
    ).run(sealed.artifact.imported.module)
    lowered = lower_module_for_differential(sealed.artifact, pipeline.module)
    return sealed.artifact, pipeline, lowered


def _assert_state_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    pivot = int(torch.argmax(torch.abs(expected.reshape(-1))).item())
    phase = actual.reshape(-1)[pivot] / expected.reshape(-1)[pivot]
    torch.testing.assert_close(actual, phase * expected, atol=1e-10, rtol=0)


def test_directed_graph_is_canonical_and_rejects_invalid_edges() -> None:
    graph = DirectedCouplingGraph(3, ((2, 1), (0, 1)))

    assert graph.edges == ((0, 1), (2, 1))
    assert graph.shortest_path(0, 2) == (0, 1, 2)
    with pytest.raises(ValueError, match="unique"):
        DirectedCouplingGraph(2, ((0, 1), (0, 1)))
    with pytest.raises(ValueError, match="outside"):
        DirectedCouplingGraph(2, ((0, 2),))
    with pytest.raises(ValueError, match="self"):
        DirectedCouplingGraph(2, ((0, 0),))


def test_non_adjacent_cx_routes_and_restores_logical_order() -> None:
    source = fq.CircuitIR(
        3,
        (
            fq.Instruction("ry", (0,), {"theta": 0.31}),
            fq.Instruction("rx", (2,), {"theta": -0.27}),
            fq.Instruction("cx", (0, 2)),
            fq.Instruction("rz", (1,), {"theta": 0.19}),
        ),
        dtype="complex128",
    )
    _, pipeline, lowered = _route(source, DirectedCouplingGraph(3, ((0, 1), (1, 2))))

    assert pipeline.ok, pipeline.diagnostics
    assert lowered.ok and lowered.circuit_ir is not None
    routing = pipeline.pass_results[-1]
    assert routing.statistics["initial_logical_to_physical"] == (0, 1, 2)
    assert routing.statistics["final_logical_to_physical"] == (0, 1, 2)
    assert routing.statistics["physical_swap_count"] == 2
    assert set(item.name for item in lowered.circuit_ir.instructions) <= {
        "rx",
        "ry",
        "rz",
        "cx",
    }
    assert all(
        item.name != "cx" or (item.wires in {(0, 1), (1, 2)})
        for item in lowered.circuit_ir.instructions
    )
    _assert_state_equal(fq.run(lowered.circuit_ir).state, fq.run(source).state)


def test_reverse_cx_direction_is_synthesized_without_operand_relabeling() -> None:
    source = fq.CircuitIR(
        2,
        (
            fq.Instruction("ry", (0,), {"theta": 0.43}),
            fq.Instruction("rx", (1,), {"theta": -0.37}),
            fq.Instruction("cx", (0, 1)),
        ),
        dtype="complex128",
    )
    _, pipeline, lowered = _route(source, DirectedCouplingGraph(2, ((1, 0),)))

    assert pipeline.ok, pipeline.diagnostics
    assert lowered.ok and lowered.circuit_ir is not None
    assert pipeline.pass_results[-1].statistics["reversed_cx_count"] == 1
    assert all(
        item.name != "cx" or item.wires == (1, 0)
        for item in lowered.circuit_ir.instructions
    )
    _assert_state_equal(fq.run(lowered.circuit_ir).state, fq.run(source).state)


def test_explicit_initial_layout_is_restored_before_lowering() -> None:
    source = fq.CircuitIR(
        3,
        (
            fq.Instruction("ry", (0,), {"theta": 0.23}),
            fq.Instruction("rx", (1,), {"theta": 0.41}),
            fq.Instruction("cx", (0, 1)),
        ),
        dtype="complex128",
    )
    _, pipeline, lowered = _route(
        source,
        DirectedCouplingGraph(3, ((0, 1), (1, 0), (1, 2), (2, 1))),
        initial_layout=(2, 0, 1),
    )

    assert pipeline.ok, pipeline.diagnostics
    assert lowered.ok and lowered.circuit_ir is not None
    statistics = pipeline.pass_results[-1].statistics
    assert statistics["initial_logical_to_physical"] == (2, 0, 1)
    assert statistics["final_logical_to_physical"] == (0, 1, 2)
    _assert_state_equal(fq.run(lowered.circuit_ir).state, fq.run(source).state)


def test_routing_preserves_trainable_parameter_gradient() -> None:
    theta = torch.tensor(0.29, dtype=torch.float64, requires_grad=True)
    source = fq.CircuitIR(
        3,
        (
            fq.Instruction("ry", (0,), {"theta": theta}),
            fq.Instruction("cx", (0, 2)),
            fq.Instruction("ry", (2,), {"theta": 0.17}),
        ),
        dtype="complex128",
    )
    _, pipeline, lowered = _route(source, DirectedCouplingGraph(3, ((0, 1), (1, 2))))

    assert pipeline.ok and lowered.ok and lowered.circuit_ir is not None
    source_loss = fq.Circuit.from_ir(source).expectation_z(2).sum()
    source_gradient = torch.autograd.grad(source_loss, theta, retain_graph=True)[0]
    routed_loss = fq.Circuit.from_ir(lowered.circuit_ir).expectation_z(2).sum()
    routed_gradient = torch.autograd.grad(routed_loss, theta)[0]
    torch.testing.assert_close(routed_loss, source_loss, atol=1e-6, rtol=0)
    torch.testing.assert_close(routed_gradient, source_gradient, atol=1e-6, rtol=0)


def test_disconnected_topology_fails_closed_without_partial_module() -> None:
    source = fq.CircuitIR(3, (fq.Instruction("cx", (0, 2)),))
    artifact, pipeline, lowered = _route(source, DirectedCouplingGraph(3, ((0, 1),)))

    assert not pipeline.ok
    assert pipeline.module is artifact.imported.module
    assert "no coupling path" in pipeline.diagnostics[0].message
    assert lowered.ok
    assert lowered.circuit_ir == source


@pytest.mark.parametrize("initial_layout", ((0, 1), (0, 0, 2), (0, 1, 3)))
def test_invalid_or_mismatched_layout_fails_closed(
    initial_layout: tuple[int, ...],
) -> None:
    source = fq.CircuitIR(3, (fq.Instruction("rx", (0,), {"theta": 0.2}),))
    artifact, pipeline, _ = _route(
        source,
        DirectedCouplingGraph(3, ((0, 1), (1, 2))),
        initial_layout=initial_layout,
    )

    assert not pipeline.ok
    assert pipeline.module is artifact.imported.module


def test_routing_is_identity_deterministic_and_direct_legal_path_is_unchanged() -> None:
    source = fq.CircuitIR(
        2,
        (
            fq.Instruction("rx", (0,), {"theta": 0.2}),
            fq.Instruction("cx", (0, 1)),
        ),
    )
    graph = DirectedCouplingGraph(2, ((0, 1),))
    artifact, first, _ = _route(source, graph)
    second = PassManager(
        (
            StaticCanonicalizationPass(),
            DecomposeToTargetGateSetPass(),
            PlacementRoutingPass(graph),
        )
    ).run(artifact.imported.module)

    assert first.ok and second.ok
    assert first.module.program_identity == second.module.program_identity
    assert first.pass_results[-1].changed is False
    assert first.pass_results[-1].module is first.pass_results[-2].module


def test_rerouting_an_existing_routing_scope_keeps_value_ids_unique() -> None:
    source = fq.CircuitIR(
        3,
        (
            fq.Instruction("ry", (0,), {"theta": 0.2}),
            fq.Instruction("cx", (0, 2)),
        ),
        dtype="complex128",
    )
    artifact, first, _ = _route(source, DirectedCouplingGraph(3, ((0, 1), (1, 2))))
    second = PassManager(
        (PlacementRoutingPass(DirectedCouplingGraph(3, ((1, 0), (2, 1)))),)
    ).run(first.module)
    lowered = lower_module_for_differential(artifact, second.module)

    assert first.ok and second.ok
    assert lowered.ok and lowered.circuit_ir is not None
    _assert_state_equal(fq.run(lowered.circuit_ir).state, fq.run(source).state)


@pytest.mark.parametrize(
    ("initial_layout", "control", "target"),
    tuple(
        (layout, control, target)
        for layout in itertools.permutations(range(3))
        for control, target in itertools.permutations(range(3), 2)
    ),
)
def test_all_three_qubit_layouts_and_cx_orders_preserve_state(
    initial_layout: tuple[int, ...], control: int, target: int
) -> None:
    source = fq.CircuitIR(
        3,
        (
            fq.Instruction("ry", (control,), {"theta": 0.31}),
            fq.Instruction("rx", (target,), {"theta": -0.23}),
            fq.Instruction("cx", (control, target)),
        ),
        dtype="complex128",
    )
    graph = DirectedCouplingGraph(3, ((0, 1), (2, 1)))
    _, pipeline, lowered = _route(source, graph, initial_layout=initial_layout)

    assert pipeline.ok, pipeline.diagnostics
    assert lowered.ok and lowered.circuit_ir is not None
    assert pipeline.pass_results[-1].statistics["final_logical_to_physical"] == (
        0,
        1,
        2,
    )
    assert all(
        item.name != "cx" or item.wires in {(0, 1), (2, 1)}
        for item in lowered.circuit_ir.instructions
    )
    _assert_state_equal(fq.run(lowered.circuit_ir).state, fq.run(source).state)
