"""Tests for general tensor-network circuit contraction."""

import torch

import flagquantum as fq
import flagquantum.simulation.tensor as tensor_runtime


def test_tensor_network_bell_state_matches_statevector():
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)

    tn, plan = fq.run_native(circuit, mode="tensor_network", return_plan=True)

    assert plan.state_mode == "tensor_network"
    assert plan.recommended_mode == "tensor_network"
    assert tn.summary()["state_mode"] == "tensor_network"
    assert tn.summary()["n_nodes"] == 4
    assert torch.allclose(tn.to_statevector(), circuit.state(), atol=1e-6)
    assert torch.allclose(tn.expectation_z(), circuit.expectation_z(), atol=1e-6)


def test_tensor_network_alias_and_top_level_runner():
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 2).ry(1, theta=0.2)

    by_alias = fq.run_native(circuit, mode="tn")
    by_function = fq.run_tensor_network(circuit)

    assert isinstance(by_alias, fq.TensorNetworkState)
    assert torch.allclose(by_alias.state(), circuit.state(), atol=1e-6)
    assert torch.allclose(by_function.state(), circuit.state(), atol=1e-6)


def test_compiled_tn_program_reuses_topology_with_new_tensor_slots():
    circuit = fq.Circuit(2).ry(0, 0.2).cx(0, 1)

    first = fq.run_tensor_network(circuit)
    program = circuit._backend_programs[("tensor_network", 1)]
    second = fq.run_tensor_network(circuit)
    rebound = program.bind(tuple(node.tensor for node in first.plan.nodes))

    assert isinstance(program, tensor_runtime.CompiledTNProgram)
    assert circuit._backend_programs[("tensor_network", 1)] is program
    assert rebound.path is program.path
    assert first.plan.nodes is not second.plan.nodes


def test_tensor_network_parameter_gradient_matches_statevector():
    theta = torch.tensor(0.3, requires_grad=True)
    circuit = fq.Circuit(1)
    circuit.rx(0, theta=theta)

    tn = fq.run_native(circuit, mode="tensor_network")
    loss = tn.expectation_z(0).sum()
    loss.backward()

    assert theta.grad is not None
    assert torch.allclose(theta.grad, -torch.sin(theta.detach()), atol=1e-6)


def test_tensor_network_sampling_counts_and_pauli_expectation():
    circuit = fq.Circuit(2)
    circuit.x(0).x(1)
    tn = fq.run_tensor_network(circuit)

    assert torch.allclose(tn.expectation_ps(z=[0, 1]), torch.ones(1), atol=1e-6)
    assert tn.counts(8, generator=torch.Generator().manual_seed(1)) == [{"11": 8}]
    assert tn.counts(8, generator=torch.Generator().manual_seed(1), format="int") == [
        {3: 8}
    ]


def test_tensor_network_expectation_uses_direct_contraction():
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)
    tn = fq.run_tensor_network(circuit)

    def blocked_state(*args, **kwargs):
        raise AssertionError("direct expectation must not materialize statevector")

    tn.state = blocked_state
    tn.to_statevector = blocked_state

    assert torch.allclose(tn.expectation_z(), torch.zeros(1, 2), atol=1e-6)
    assert torch.allclose(tn.expectation_ps(x=[0, 1]), torch.ones(1), atol=1e-6)
    assert torch.allclose(tn.expectation_ps(y=[0, 1]), -torch.ones(1), atol=1e-6)
    assert torch.allclose(tn.expectation_ps(z=[0, 1]), torch.ones(1), atol=1e-6)


def test_tensor_network_expectation_plan_is_public_and_matches_circuit():
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 2).ry(1, theta=0.2)
    plan = fq.build_tensor_network(circuit)
    expectation_plan = fq.build_tensor_network_expectation(plan, x=[0, 2], z=[1])

    assert isinstance(expectation_plan, fq.TensorNetworkExpectationPlan)
    assert expectation_plan.observable_wires == (0, 1, 2)
    assert expectation_plan.summary()["contraction"] == "expectation"
    assert torch.allclose(
        fq.tensor_network_expectation_ps(plan, x=[0, 2], z=[1]),
        circuit.expectation_ps(x=[0, 2], z=[1]),
        atol=1e-6,
    )


def test_build_tensor_network_exposes_contraction_plan():
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)

    plan = fq.build_tensor_network(circuit)

    assert isinstance(plan, fq.TensorNetworkContractionPlan)
    assert plan.n_wires == 2
    assert plan.n_nodes == 4
    assert len(plan.path) == plan.n_nodes
    assert torch.allclose(plan.contract(), circuit.state(), atol=1e-6)


def test_tensor_network_greedy_path_matches_einsum_contract():
    circuit = fq.Circuit(4)
    circuit.h(0).cx(0, 3).ry(1, theta=0.2).rzz(2, 3, theta=-0.4)
    plan = fq.build_tensor_network(circuit)

    greedy_path = plan.greedy_path()
    cost = plan.contraction_cost()

    assert greedy_path
    assert isinstance(greedy_path[0], fq.PairContractionStep)
    assert cost["estimated_cost"] > 0
    assert cost["peak_size"] > 0
    assert torch.allclose(
        plan.contract(strategy="greedy"), plan.contract(strategy="einsum"), atol=1e-6
    )
    assert torch.allclose(plan.contract(strategy="greedy"), circuit.state(), atol=1e-6)
    assert plan.summary()["greedy_cost"] == cost["estimated_cost"]


def test_tensor_network_memory_greedy_profile_and_contract():
    circuit = fq.Circuit(5)
    circuit.h(0).cx(0, 4).ry(1, theta=0.2).rzz(2, 3, theta=-0.4).cx(1, 3)
    plan = fq.build_tensor_network(circuit)

    profile = plan.contraction_profile("memory_greedy")
    greedy_profile = plan.contraction_profile("greedy")

    assert isinstance(profile, fq.TensorNetworkContractionProfile)
    assert profile.strategy == "memory_greedy"
    assert profile.n_steps == len(plan.memory_greedy_path())
    assert profile.estimated_cost > 0
    assert profile.peak_size <= greedy_profile.peak_size
    assert profile.summary()["total_intermediate_size"] >= profile.peak_size
    assert plan.summary()["memory_greedy_peak_size"] == profile.peak_size
    assert torch.allclose(
        plan.contract(strategy="memory_greedy"),
        plan.contract(strategy="greedy"),
        atol=1e-6,
    )
    assert torch.allclose(
        plan.contract(strategy="memory_greedy"), circuit.state(), atol=1e-6
    )


def test_tensor_network_beam_profile_cache_and_auto_slicing():
    circuit = fq.Circuit(5)
    circuit.h(0).cx(0, 4).ry(1, theta=0.2).rzz(2, 3, theta=-0.4).cx(1, 3)
    plan = fq.build_tensor_network(circuit)

    beam_profile = plan.contraction_profile("beam", beam_width=4)
    cached_profile = plan.contraction_profile("beam", beam_width=4)
    greedy_profile = plan.contraction_profile("greedy")
    target_peak = max(1, greedy_profile.peak_size // 2)
    sliced_profile = plan.contraction_profile(
        "auto_sliced", max_intermediate_size=target_peak
    )
    beam_sliced_profile = plan.contraction_profile(
        "beam_sliced", max_intermediate_size=target_peak, beam_width=4
    )

    assert beam_profile is cached_profile
    assert beam_profile.strategy == "beam"
    assert beam_profile.n_steps == len(plan.beam_path(beam_width=4))
    assert beam_profile.peak_size <= greedy_profile.peak_size
    assert sliced_profile.strategy == "auto_sliced"
    assert sliced_profile.n_slices >= 1
    assert beam_sliced_profile.strategy == "beam_sliced"
    assert beam_sliced_profile.n_slices >= 1
    assert beam_sliced_profile.peak_size <= beam_profile.peak_size
    assert torch.allclose(
        plan.contract(strategy="beam"), plan.contract(strategy="greedy"), atol=1e-6
    )
    assert torch.allclose(
        plan.contract(strategy="auto_sliced", max_intermediate_size=target_peak),
        plan.contract(strategy="greedy"),
        atol=1e-6,
    )
    assert torch.allclose(
        plan.contract(
            strategy="beam_sliced", max_intermediate_size=target_peak, beam_width=4
        ),
        plan.contract(strategy="greedy"),
        atol=1e-6,
    )


def test_tensor_network_optimal_small_path_matches_einsum():
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)
    plan = fq.build_tensor_network(circuit)

    optimal_profile = plan.contraction_profile("optimal")

    assert optimal_profile.strategy == "optimal"
    assert optimal_profile.n_steps == len(plan.optimal_path())
    assert optimal_profile.peak_size <= plan.contraction_profile("greedy").peak_size
    assert torch.allclose(
        plan.contract(strategy="optimal"), plan.contract(strategy="einsum"), atol=1e-6
    )
    assert torch.allclose(plan.contract(strategy="optimal"), circuit.state(), atol=1e-6)


def test_tensor_network_sliced_contract_matches_greedy_contract():
    circuit = fq.Circuit(4)
    circuit.h(0).cx(0, 3).ry(1, theta=0.2).rzz(2, 3, theta=-0.4)
    plan = fq.build_tensor_network(circuit)
    greedy_cost = plan.contraction_cost()
    slicing = plan.slicing_plan(
        max_intermediate_size=max(1, greedy_cost["peak_size"] // 2)
    )

    assert isinstance(slicing, fq.TensorNetworkSlicingPlan)
    assert slicing.n_slices >= 1
    assert slicing.peak_size <= greedy_cost["peak_size"]
    assert torch.allclose(
        plan.contract(
            strategy="sliced",
            max_intermediate_size=max(1, greedy_cost["peak_size"] // 2),
        ),
        plan.contract(strategy="greedy"),
        atol=1e-6,
    )


def test_tensor_network_expectation_greedy_path_matches_einsum():
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 2).ry(1, theta=0.2)
    expectation_plan = fq.build_tensor_network_expectation(circuit, x=[0, 2], z=[1])

    assert expectation_plan.greedy_path()
    assert expectation_plan.contraction_cost()["estimated_cost"] > 0
    assert torch.allclose(
        expectation_plan.contract(strategy="greedy"),
        expectation_plan.contract(strategy="einsum"),
        atol=1e-6,
    )
    assert torch.allclose(
        expectation_plan.contract(strategy="greedy").real,
        circuit.expectation_ps(x=[0, 2], z=[1]),
        atol=1e-6,
    )


def test_tensor_network_expectation_memory_greedy_profile_matches_greedy():
    circuit = fq.Circuit(4)
    circuit.h(0).cx(0, 3).ry(1, theta=0.2).rzz(2, 3, theta=-0.4)
    expectation_plan = fq.build_tensor_network_expectation(circuit, x=[0, 3], z=[1])

    profile = expectation_plan.contraction_profile("memory_greedy")

    assert isinstance(profile, fq.TensorNetworkContractionProfile)
    assert profile.strategy == "memory_greedy"
    assert profile.n_steps == len(expectation_plan.memory_greedy_path())
    assert expectation_plan.summary()["memory_greedy_peak_size"] == profile.peak_size
    assert torch.allclose(
        expectation_plan.contract(strategy="memory_greedy"),
        expectation_plan.contract(strategy="greedy"),
        atol=1e-6,
    )


def test_tensor_network_expectation_beam_matches_greedy():
    circuit = fq.Circuit(4)
    circuit.h(0).cx(0, 3).ry(1, theta=0.2).rzz(2, 3, theta=-0.4)
    expectation_plan = fq.build_tensor_network_expectation(circuit, x=[0, 3], z=[1])

    profile = expectation_plan.contraction_profile("beam", beam_width=4)

    assert profile.strategy == "beam"
    assert profile.n_steps == len(expectation_plan.beam_path(beam_width=4))
    assert torch.allclose(
        expectation_plan.contract(strategy="beam"),
        expectation_plan.contract(strategy="greedy"),
        atol=1e-6,
    )

    greedy_profile = expectation_plan.contraction_profile("greedy")
    target_peak = max(1, greedy_profile.peak_size // 2)
    beam_sliced = expectation_plan.contraction_profile(
        "beam_sliced",
        max_intermediate_size=target_peak,
        beam_width=4,
    )
    assert beam_sliced.strategy == "beam_sliced"
    assert torch.allclose(
        expectation_plan.contract(
            strategy="beam_sliced", max_intermediate_size=target_peak, beam_width=4
        ),
        expectation_plan.contract(strategy="greedy"),
        atol=1e-6,
    )


def test_tensor_network_expectation_optimal_matches_greedy():
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)
    expectation_plan = fq.build_tensor_network_expectation(circuit, z=[0, 1])

    profile = expectation_plan.contraction_profile("optimal")

    assert profile.strategy == "optimal"
    assert profile.n_steps == len(expectation_plan.optimal_path())
    assert torch.allclose(
        expectation_plan.contract(strategy="optimal"),
        expectation_plan.contract(strategy="greedy"),
        atol=1e-6,
    )


def test_tensor_network_expectation_sliced_contract_matches_greedy():
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 2).ry(1, theta=0.2)
    expectation_plan = fq.build_tensor_network_expectation(circuit, x=[0, 2], z=[1])
    greedy_cost = expectation_plan.contraction_cost()
    slicing = expectation_plan.slicing_plan(
        max_intermediate_size=max(1, greedy_cost["peak_size"] // 2)
    )

    assert slicing.n_slices >= 1
    assert torch.allclose(
        expectation_plan.contract(
            strategy="sliced",
            max_intermediate_size=max(1, greedy_cost["peak_size"] // 2),
        ),
        expectation_plan.contract(strategy="greedy"),
        atol=1e-6,
    )


def test_tensor_network_run_accepts_sliced_strategy_options():
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 2).ry(1, theta=0.2)
    plan = fq.build_tensor_network(circuit)
    target_peak = max(1, plan.contraction_cost()["peak_size"] // 2)

    tn = fq.run_native(
        circuit,
        mode="tensor_network",
        contraction_strategy="sliced",
        max_intermediate_size=target_peak,
    )

    assert tn.contraction_strategy == "sliced"
    assert tn.max_intermediate_size == target_peak
    assert torch.allclose(tn.state(), circuit.state(), atol=1e-6)


def test_greedy_contraction_path_is_reused_for_dynamic_parameters():
    tensor_runtime._CONTRACTION_PATH_CACHE.clear()
    theta = torch.tensor(0.2, requires_grad=True)
    first = fq.run_tensor_network(fq.Circuit(2).ry(0, theta).cx(0, 1))
    first.expectation_z(0).sum().backward()
    cache_size = len(tensor_runtime._CONTRACTION_PATH_CACHE)

    phi = torch.tensor(-0.3, requires_grad=True)
    second = fq.run_tensor_network(fq.Circuit(2).ry(0, phi).cx(0, 1))
    second.expectation_z(0).sum().backward()

    assert cache_size > 0
    assert len(tensor_runtime._CONTRACTION_PATH_CACHE) == cache_size
    assert theta.grad is not None and phi.grad is not None


def test_compiled_contraction_stage_plan_prebuckets_matching_operations():
    nodes = (
        tensor_runtime.TensorNetworkNode(torch.ones(2, 2), (0, 1)),
        tensor_runtime.TensorNetworkNode(torch.ones(2, 2), (1, 2)),
        tensor_runtime.TensorNetworkNode(torch.ones(2, 2), (0, 3)),
        tensor_runtime.TensorNetworkNode(torch.ones(2, 2), (3, 4)),
    )
    path = (
        (0, 1, (0, 2)),
        (0, 1, (0, 4)),
        (0, 1, (2, 4)),
    )

    plan = tensor_runtime._compile_contraction_stages(nodes, path)

    assert plan.output_labels == (2, 4)
    assert len(plan.stages) == 2
    assert len(plan.stages[0].buckets) == 1
    assert len(plan.stages[0].buckets[0].operations) == 2
    assert plan.stages[0].buckets[0].batched_equation.startswith("Z")


def test_small_tn_reuses_one_state_contraction_for_many_z_observables():
    theta = torch.tensor(0.2, requires_grad=True)
    state = fq.run_tensor_network(
        fq.Circuit(4).ry(0, theta).cx(0, 1).cx(1, 2).cx(2, 3),
        dense_observable_wires=12,
    )

    values = state.expectation_z((0, 1, 2, 3))
    values.sum().backward()

    assert values.shape == (1, 4)
    assert state._state_cache is not None
    assert state.summary()["observable_execution"] == "dense_state"
    assert theta.grad is not None


def test_compiled_z_observable_batch_avoids_full_state_materialization():
    theta = torch.tensor(0.2, requires_grad=True)
    state = fq.run_tensor_network(
        fq.Circuit(3).ry(0, theta).cx(0, 1).cx(1, 2),
        dense_observable_wires=0,
    )

    values = state.expectation_z((0, 1, 2))
    values.sum().backward()

    assert values.shape == (1, 3)
    assert state._state_cache is None
    assert state.summary()["observable_execution"] == "memory_first_direct_batch"
    assert isinstance(
        state._observable_programs[(0, 1, 2)],
        tensor_runtime.CompiledTNObservableProgram,
    )
    assert theta.grad is not None


def test_compiled_z_observable_program_is_reused_across_runs():
    circuit = fq.Circuit(3).ry(0, 0.2).cx(0, 1).cx(1, 2)

    first = fq.run_tensor_network(circuit, dense_observable_wires=0)
    first_values = first.expectation_z((0, 1, 2))
    second = fq.run_tensor_network(circuit, dense_observable_wires=0)
    second_values = second.expectation_z((0, 1, 2))

    assert first.summary()["observable_program_cache_hit"] is False
    assert second.summary()["observable_program_cache_hit"] is True
    assert second.summary()["observable_peak_size"] is not None
    assert torch.allclose(first_values, second_values)


def test_plan_accepts_tensor_network_state_mode():
    circuit = fq.Circuit(4)
    circuit.h(0).cx(0, 3)

    plan = circuit.plan(state_mode="tensor_network")

    assert plan.state_mode == "tensor_network"
    assert plan.recommended_mode == "tensor_network"
    assert plan.state_bytes == fq.estimate_tensor_network_bytes(4)
