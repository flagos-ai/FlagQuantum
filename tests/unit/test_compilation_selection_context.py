import pytest

from flagquantum.compilation.models import CircuitAnalysis
from flagquantum.compilation.selection_context import (
    build_runtime_selection_context,
)

pytestmark = pytest.mark.unit


def _analysis(n_wires: int = 4) -> CircuitAnalysis:
    return CircuitAnalysis(
        n_wires=n_wires,
        n_instructions=0,
        depth=0,
        gate_counts={},
        wire_usage=(0,) * n_wires,
        max_gate_width=0,
        two_qubit_gates=0,
        multi_qubit_gates=0,
    )


def test_selection_context_normalizes_topology_and_tn_alias() -> None:
    context = build_runtime_selection_context(
        _analysis(),
        bsz=1,
        world_size=0,
        local_world_size=8,
        node_count=0,
        complex_bytes=8,
        max_bond=None,
        max_intermediate_size=None,
        state_mode="tn",
        prefer_jax=False,
        prefer_distributed=None,
        require_gradients=True,
        require_deployment=False,
    )

    assert (context.world_size, context.local_world_size, context.node_count) == (
        1,
        1,
        1,
    )
    assert context.prefer_distributed is False
    assert context.requested_state_mode == "tensor_network"
    assert context.objective == "training+simulation+native_preferred"
    assert context.sharded_dense_bytes == context.dense_bytes


def test_selection_context_records_explicit_distributed_objective_and_peak() -> None:
    context = build_runtime_selection_context(
        _analysis(),
        bsz=2,
        world_size=8,
        local_world_size=4,
        node_count=None,
        complex_bytes=16,
        max_bond=32,
        max_intermediate_size=12345,
        state_mode="mps",
        prefer_jax=True,
        prefer_distributed=True,
        require_gradients=False,
        require_deployment=True,
    )

    assert (context.world_size, context.local_world_size, context.node_count) == (
        8,
        4,
        2,
    )
    assert context.objective == (
        "inference+deployment_ready+jax_preferred+distributed_scale_out"
    )
    assert context.tensor_network_peak_bytes == 12345
    assert (
        context.sharded_dense_bytes
        == (context.dense_bytes + context.world_size - 1) // context.world_size
    )
