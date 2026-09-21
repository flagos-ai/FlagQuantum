"""Tensor-network residency has one definition across selection and planning.

These tests pin the coupling, not the numbers: the selector proxy, the selection
context, and the execution policy must agree because they now share one function.
They do not claim the shared function bounds the executor's measured peak; see
``test_tensor_network_proxy_is_not_an_upper_bound_on_the_measured_peak``.
"""

import pytest

import flagquantum as fq
from flagquantum.runtime.planner import (
    OutputTarget,
    analyze,
    estimate_tensor_network_bytes,
    estimate_tensor_network_working_set_bytes,
    interaction_width,
    plan_advanced,
    select_backend_by_cost,
)
from flagquantum.runtime.planner.execution_policy import (
    estimate_execution_state_bytes,
    recommend_execution_mode,
)
from flagquantum.runtime.planner.selection_context import (
    build_runtime_selection_context,
)

pytestmark = pytest.mark.unit


def _all_to_all(n_wires: int) -> fq.Circuit:
    circuit = fq.Circuit(n_wires)
    for left in range(n_wires):
        for right in range(left + 1, n_wires):
            circuit.cx(left, right)
    return circuit


def test_interaction_width_is_the_width_the_selector_reports() -> None:
    circuit = _all_to_all(4)

    decision = select_backend_by_cost(
        circuit,
        target="expectation",
        memory_limit_bytes=1 << 30,
    )

    assert decision.interaction_width_proxy == interaction_width(circuit) == 3


def test_working_set_estimate_reproduces_the_selector_proxy() -> None:
    circuit = _all_to_all(4)
    width = interaction_width(circuit)

    cases: tuple[tuple[OutputTarget, int, bool], ...] = (
        ("expectation", 1, False),
        ("few_amplitudes", 8, False),
        ("local_observables", 4, True),
    )
    for target, target_count, require_gradients in cases:
        decision = select_backend_by_cost(
            circuit,
            target=target,
            target_count=target_count,
            require_gradients=require_gradients,
            memory_limit_bytes=1 << 30,
        )
        tensor_network = next(
            item for item in decision.candidates if item.backend == "tensor_network"
        )

        assert tensor_network.estimated_memory_bytes == (
            estimate_tensor_network_working_set_bytes(
                circuit.n_wires,
                contraction_width=width,
                target_count=target_count,
                require_gradients=require_gradients,
            )
        )


def test_sparse_target_keeps_the_materialized_state_as_a_floor() -> None:
    circuit = _all_to_all(4)

    plan = plan_advanced(circuit, state_mode="tensor_network", target="expectation")

    # A contraction that returns one expectation can hold more intermediate
    # amplitudes than it returns, so a sparse target is not allowed to shrink the
    # estimate below the state a full contraction would materialize.
    assert plan.state_bytes == max(
        estimate_tensor_network_bytes(circuit.n_wires),
        estimate_tensor_network_working_set_bytes(
            circuit.n_wires,
            contraction_width=interaction_width(circuit),
        ),
    )
    assert plan.state_bytes >= estimate_tensor_network_bytes(circuit.n_wires)


def test_full_state_target_keeps_the_materialized_state_as_a_floor() -> None:
    circuit = fq.Circuit(4)
    circuit.h(0)

    plan = plan_advanced(circuit, state_mode="tensor_network")

    assert interaction_width(circuit) == 0
    assert plan.state_bytes == estimate_tensor_network_bytes(circuit.n_wires)
    assert plan.state_bytes == estimate_execution_state_bytes(
        "tensor_network",
        n_wires=circuit.n_wires,
        bsz=1,
        complex_bytes=8,
        max_bond=None,
        contraction_width=0,
    )


def test_a_wide_contraction_is_not_reported_as_only_its_output() -> None:
    circuit = _all_to_all(4)

    plan = plan_advanced(circuit, state_mode="tensor_network")

    assert plan.state_bytes == estimate_tensor_network_working_set_bytes(
        circuit.n_wires,
        contraction_width=interaction_width(circuit),
    )
    assert plan.state_bytes > estimate_tensor_network_bytes(circuit.n_wires)


def test_working_set_is_monotone_in_the_contraction_width() -> None:
    chain = fq.Circuit(6)
    for wire in range(5):
        chain.cx(wire, wire + 1)
    product = fq.Circuit(6)
    product.h(0)

    estimates = [
        estimate_tensor_network_working_set_bytes(6, contraction_width=width)
        for width in (interaction_width(product), interaction_width(chain))
    ]

    assert estimates[0] < estimates[1]


def test_omitted_width_assumes_a_fully_connected_program() -> None:
    chain = fq.Circuit(6)
    for wire in range(5):
        chain.cx(wire, wire + 1)
    width = interaction_width(chain)

    assert estimate_tensor_network_working_set_bytes(
        6, contraction_width=width
    ) < estimate_tensor_network_working_set_bytes(6, contraction_width=None)
    assert estimate_tensor_network_working_set_bytes(
        6, contraction_width=None
    ) == estimate_tensor_network_working_set_bytes(6, contraction_width=6)


def test_selection_context_peak_matches_the_execution_estimate() -> None:
    circuit = _all_to_all(4)
    width = interaction_width(circuit)
    analysis = analyze(circuit.to_ir())

    context = build_runtime_selection_context(
        analysis,
        bsz=1,
        world_size=1,
        local_world_size=None,
        node_count=None,
        complex_bytes=8,
        max_bond=None,
        max_intermediate_size=None,
        state_mode="tensor_network",
        prefer_jax=False,
        prefer_distributed=None,
        require_gradients=False,
        require_deployment=False,
        contraction_width=width,
    )

    assert context.tensor_network_peak_bytes == estimate_execution_state_bytes(
        "tensor_network",
        n_wires=circuit.n_wires,
        bsz=1,
        complex_bytes=8,
        max_bond=None,
        contraction_width=width,
    )


def test_fail_closed_capacity_check_uses_the_working_set() -> None:
    circuit = _all_to_all(4)
    working_set = estimate_tensor_network_working_set_bytes(
        circuit.n_wires,
        contraction_width=interaction_width(circuit),
    )
    dense_state = estimate_tensor_network_bytes(circuit.n_wires)
    limit = (dense_state + working_set) // 2

    assert dense_state <= limit < working_set

    plan = plan_advanced(
        circuit,
        state_mode="tensor_network",
        target="expectation",
        memory_limit_bytes=limit,
    )

    assert plan.state_bytes == working_set > limit
    assert plan.recommended_mode == "mps"
    # The output-state number on its own would have admitted the run.
    assert (
        recommend_execution_mode(
            "tensor_network",
            world_size=1,
            has_noise=False,
            state_bytes=dense_state,
            memory_limit_bytes=limit,
        )
        == "tensor_network"
    )
