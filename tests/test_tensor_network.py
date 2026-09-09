"""Tests for general tensor-network circuit contraction."""

import pytest
import torch

import flagquantum as fq
import flagquantum.runtime as fqr
import flagquantum.runtime.executors.tensor_network.execution as fqxd
import flagquantum.runtime.executors.tensor_network.plan_cache as distributed_plan_cache
import flagquantum.runtime.planner as fqxp
import flagquantum.simulation.tensor_network as fqtn
import flagquantum.simulation.tensor_network.entrypoints as tensor_execution
import flagquantum.simulation.tensor_network.observables as tensor_observables
from flagquantum.algorithms import Hamiltonian, pauli_term
from flagquantum.runtime.executors.tensor_network.joint_planning import (
    DistributedTNWorkingSetPolicy,
)
from flagquantum.runtime.planner import (
    build_tn_working_set_calibration,
    estimate_tensor_network_bytes,
)
from flagquantum.simulation.tensor_network.contraction import (
    _build_slicing_plan,
    _cost_for_sliced_labels,
    _internal_slice_candidates,
    _parallel_slice_labels,
)
from flagquantum.simulation.tensor_network.entrypoints import (
    build_tensor_network,
    build_tensor_network_expectation,
    build_tensor_network_hamiltonian_expectation,
    build_tensor_network_hamiltonian_expectations,
)
from flagquantum.simulation.tensor_network.models import (
    CompiledTNObservableProgram,
    CompiledTNProgram,
    PairContractionStep,
    TensorNetworkContractionPlan,
    TensorNetworkContractionProfile,
    TensorNetworkExpectationPlan,
    TensorNetworkNode,
    TensorNetworkSlicingPlan,
)
from flagquantum.simulation.tensor_network.path_search import (
    _CONTRACTION_PATH_CACHE,
    _contract_nodes_quality_multistart,
)
from flagquantum.simulation.tensor_network.stages import (
    compile_contraction_stages as _compile_contraction_stages,
)
from flagquantum.simulation.tensor_network.state import (
    TensorNetworkState,
    tensor_network_expectation_ps,
)


def test_tensor_network_bell_state_matches_statevector():
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)

    tn, plan = fqr.run_native(circuit, mode="tensor_network", return_plan=True)

    assert plan.state_mode == "tensor_network"
    assert plan.recommended_mode == "tensor_network"
    assert tn.summary()["state_mode"] == "tensor_network"
    assert tn.summary()["n_nodes"] == 4
    assert torch.allclose(tn.to_statevector(), circuit.state(), atol=1e-6)
    assert torch.allclose(tn.expectation_z(), circuit.expectation_z(), atol=1e-6)


def test_hamiltonian_mpo_compression_is_reused_for_static_observable(monkeypatch):
    tensor_observables._PAULI_MPO_CORE_CACHE.clear()
    calls = 0
    original = tensor_observables._compress_pauli_sum_mpo

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(tensor_observables, "_compress_pauli_sum_mpo", counted)
    target = Hamiltonian([pauli_term(-1.0, "ZZ", (0, 1)), pauli_term(-0.7, "X", (0,))])
    first = fq.Circuit(2).ry(0, 0.1)
    second = fq.Circuit(2).ry(0, 0.2)

    first_value = build_tensor_network_hamiltonian_expectation(first, target).contract()
    second_value = build_tensor_network_hamiltonian_expectation(
        second, target
    ).contract()

    assert calls == 1
    assert torch.isfinite(first_value).all()
    assert torch.isfinite(second_value).all()


def test_persistent_distributed_plan_round_trip(tmp_path):
    plan = build_tensor_network(fq.Circuit(3).h(0).cx(0, 2))
    nodes, outputs = tensor_execution._amplitude_projection(plan, "101")
    slicing = _build_slicing_plan(nodes, outputs)
    steps = _contract_nodes_quality_multistart(nodes, outputs)
    key = distributed_plan_cache._persistent_plan_key(
        nodes,
        outputs,
        max_intermediate_size=None,
        max_intermediate_bytes=None,
        sliced_labels=None,
    )
    path = tmp_path / "plan.json"

    distributed_plan_cache._write_persistent_plan(
        path,
        cache_key=key,
        slicing=slicing,
        steps=steps,
    )
    restored = distributed_plan_cache._load_persistent_plan(path, expected_key=key)

    assert restored == (slicing, steps)
    assert isinstance(restored[0].contraction_path, tuple)
    assert all(
        isinstance(step, PairContractionStep) for step in restored[0].contraction_path
    )
    assert (
        distributed_plan_cache._load_persistent_plan(path, expected_key="different")
        is None
    )
    path.write_text("{truncated", encoding="utf-8")
    assert distributed_plan_cache._load_persistent_plan(path, expected_key=key) is None


def test_tensor_network_wrapper_delegates_local_numerics(monkeypatch):
    expected = object()
    calls = []

    def replacement(circuit_or_ir, **options):
        calls.append((circuit_or_ir, options))
        return expected

    monkeypatch.setattr(tensor_execution, "run_local_tensor_network", replacement)
    circuit = fq.Circuit(2, dtype=torch.complex128).h(0)

    assert fqtn.run_tensor_network(circuit) is expected
    assert calls == [
        (
            circuit,
            {
                "bsz": circuit.bsz,
                "device": circuit.device,
                "dtype": circuit.dtype,
                "contraction_strategy": "greedy",
                "max_intermediate_size": None,
                "sliced_labels": None,
                "dense_observable_wires": 0,
            },
        )
    ]


def test_tensor_network_observable_wrapper_delegates_plan_builder(monkeypatch):
    expected = object()
    calls = []

    def replacement(plan_or_circuit, **axes):
        calls.append((plan_or_circuit, axes))
        return expected

    monkeypatch.setattr(
        tensor_execution, "_build_tensor_network_expectation", replacement
    )
    circuit = fq.Circuit(2).h(0)

    assert build_tensor_network_expectation(circuit, x=(0,), z=(1,)) is expected
    assert calls == [(circuit, {"x": (0,), "y": None, "z": (1,)})]


def test_tensor_network_alias_and_top_level_runner():
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 2).ry(1, theta=0.2)

    by_alias = fqr.run_native(circuit, mode="tn")
    by_function = fqtn.run_tensor_network(circuit)

    assert isinstance(by_alias, TensorNetworkState)
    assert torch.allclose(by_alias.state(), circuit.state(), atol=1e-6)
    assert torch.allclose(by_function.state(), circuit.state(), atol=1e-6)


def test_compiled_tn_program_reuses_topology_with_new_tensor_slots():
    circuit = fq.Circuit(2).ry(0, 0.2).cx(0, 1)

    first = fqtn.run_tensor_network(circuit)
    program = circuit._backend_programs[("tensor_network", 1)]
    second = fqtn.run_tensor_network(circuit)
    rebound = program.bind(tuple(node.tensor for node in first.plan.nodes))

    assert isinstance(program, CompiledTNProgram)
    assert circuit._backend_programs[("tensor_network", 1)] is program
    assert rebound.path is program.path
    assert first.plan.nodes is not second.plan.nodes


def test_tensor_network_parameter_gradient_matches_statevector():
    theta = torch.tensor(0.3, requires_grad=True)
    circuit = fq.Circuit(1)
    circuit.rx(0, theta=theta)

    tn = fqr.run_native(circuit, mode="tensor_network")
    loss = tn.expectation_z(0).sum()
    loss.backward()

    assert theta.grad is not None
    assert torch.allclose(theta.grad, -torch.sin(theta.detach()), atol=1e-6)


def test_tensor_network_complex128_parameter_matrix_is_generated_in_float64():
    theta = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    circuit = fq.Circuit(1, dtype=torch.complex128).ry(0, theta=theta)

    plan = build_tensor_network(circuit, dtype=torch.complex128)
    ry = next(node.tensor for node in plan.nodes if node.name.endswith(":ry"))
    expected = torch.cos(theta.detach() / 2)

    assert ry.dtype == torch.complex128
    assert torch.equal(ry[0, 0, 0].real, expected)
    assert abs(ry[0, 0, 0].real.item() - 0.9987502694129944) > 1e-9


def test_tensor_network_sampling_counts_and_pauli_expectation():
    circuit = fq.Circuit(2)
    circuit.x(0).x(1)
    tn = fqtn.run_tensor_network(circuit)

    assert torch.allclose(tn.expectation_ps(z=[0, 1]), torch.ones(1), atol=1e-6)
    assert tn.counts(8, generator=torch.Generator().manual_seed(1)) == [{"11": 8}]
    assert tn.counts(8, generator=torch.Generator().manual_seed(1), format="int") == [
        {3: 8}
    ]


def test_tensor_network_expectation_uses_direct_contraction():
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)
    tn = fqtn.run_tensor_network(circuit)

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
    plan = build_tensor_network(circuit)
    expectation_plan = build_tensor_network_expectation(plan, x=[0, 2], z=[1])

    assert isinstance(expectation_plan, TensorNetworkExpectationPlan)
    assert expectation_plan.observable_wires == (0, 1, 2)
    assert expectation_plan.summary()["contraction"] == "expectation"
    assert torch.allclose(
        tensor_network_expectation_ps(plan, x=[0, 2], z=[1]),
        circuit.expectation_ps(x=[0, 2], z=[1]),
        atol=1e-6,
    )


def test_build_tensor_network_exposes_contraction_plan():
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)

    plan = build_tensor_network(circuit)

    assert isinstance(plan, TensorNetworkContractionPlan)
    assert plan.n_wires == 2
    assert plan.n_nodes == 4
    assert len(plan.path) == plan.n_nodes
    assert torch.allclose(plan.contract(), circuit.state(), atol=1e-6)


def test_tensor_network_greedy_path_matches_einsum_contract():
    circuit = fq.Circuit(4)
    circuit.h(0).cx(0, 3).ry(1, theta=0.2).rzz(2, 3, theta=-0.4)
    plan = build_tensor_network(circuit)

    greedy_path = plan.greedy_path()
    cost = plan.contraction_cost()

    assert greedy_path
    assert isinstance(greedy_path[0], PairContractionStep)
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
    plan = build_tensor_network(circuit)

    profile = plan.contraction_profile("memory_greedy")
    greedy_profile = plan.contraction_profile("greedy")

    assert isinstance(profile, TensorNetworkContractionProfile)
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
    assert torch.allclose(
        plan.contract(strategy="quality_greedy"),
        plan.contract(strategy="greedy"),
        atol=1e-6,
    )


def test_tensor_network_memory_greedy_avoids_outer_products_while_connected():
    nodes = (
        TensorNetworkNode(torch.ones(2), (0,), name="left"),
        TensorNetworkNode(torch.ones(2, 2), (0, 1), name="bridge"),
        TensorNetworkNode(torch.ones(2), (1,), name="right"),
        TensorNetworkNode(torch.ones(2), (2,), name="component_a"),
        TensorNetworkNode(torch.ones(2), (2,), name="component_b"),
    )
    plan = TensorNetworkContractionPlan(
        n_wires=0,
        bsz=1,
        nodes=nodes,
        output_labels=(),
        path=(),
    )
    steps = plan.quality_greedy_path()
    for step in steps[:-1]:
        assert set(step.left_labels) & set(step.right_labels)


def test_tensor_network_quality_multistart_is_deterministic_and_not_worse():
    circuit = fq.Circuit(6)
    for qubit in range(6):
        circuit.ry(qubit, theta=0.1 * (qubit + 1))
    for left, right in ((0, 1), (2, 3), (4, 5), (1, 2), (3, 4)):
        circuit.cx(left, right)
    plan = build_tensor_network_expectation(circuit, z=range(6))

    baseline = plan.contraction_profile("quality_greedy")
    first = plan.contraction_profile("quality_multistart")
    second = plan.contraction_profile("quality_multistart")

    assert first is second
    assert (first.peak_size, first.estimated_cost) <= (
        baseline.peak_size,
        baseline.estimated_cost,
    )
    assert plan.quality_multistart_path() == plan.quality_multistart_path()


def test_tensor_network_quality_reconfiguration_is_bounded_and_not_worse():
    nodes = tuple(
        [
            TensorNetworkNode(
                torch.ones(2, 2, 2),
                (3 * index, 3 * index + 1, 3 * index + 2),
                name=f"site:{index}",
            )
            for index in range(6)
        ]
        + [
            TensorNetworkNode(
                torch.ones(2, 2),
                (3 * index + 1, 3 * (index + 1)),
                name=f"bond:{index}",
            )
            for index in range(5)
        ]
        + [
            TensorNetworkNode(
                torch.ones(2),
                (3 * index + 2,),
                name=f"vector:{index}",
            )
            for index in range(6)
        ]
    )
    plan = TensorNetworkContractionPlan(0, 1, nodes, (), ())
    baseline = plan.contraction_profile("quality_multistart")
    reconfigured = plan.contraction_profile("quality_reconfigured")
    assert (reconfigured.peak_size, reconfigured.estimated_cost) <= (
        baseline.peak_size,
        baseline.estimated_cost,
    )
    assert plan.quality_reconfigured_path() == plan.quality_reconfigured_path()


def test_tensor_network_beam_profile_cache_and_auto_slicing():
    circuit = fq.Circuit(5)
    circuit.h(0).cx(0, 4).ry(1, theta=0.2).rzz(2, 3, theta=-0.4).cx(1, 3)
    plan = build_tensor_network(circuit)

    beam_profile = plan.contraction_profile("beam", beam_width=4)
    cached_profile = plan.contraction_profile("beam", beam_width=4)
    greedy_profile = plan.contraction_profile("greedy")
    target_peak = max(greedy_profile.output_size, greedy_profile.peak_size // 2)
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
    plan = build_tensor_network(circuit)

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
    plan = build_tensor_network(circuit)
    greedy_cost = plan.contraction_cost()
    output_size = plan.contraction_profile("greedy").output_size
    target_peak = max(output_size, greedy_cost["peak_size"] // 2)
    slicing = plan.slicing_plan(max_intermediate_size=target_peak)

    assert isinstance(slicing, TensorNetworkSlicingPlan)
    assert slicing.n_slices >= 1
    assert slicing.peak_size <= greedy_cost["peak_size"]
    assert slicing.budget_satisfied
    assert slicing.baseline_estimated_cost == greedy_cost["estimated_cost"]
    assert slicing.recomputation_factor >= 0.0
    assert slicing.summary()["budget_satisfied"] is True
    assert slicing.element_size_bytes == plan.nodes[0].tensor.element_size()
    assert slicing.peak_bytes == slicing.peak_size * slicing.element_size_bytes
    assert torch.allclose(
        plan.contract(
            strategy="sliced",
            max_intermediate_size=target_peak,
        ),
        plan.contract(strategy="greedy"),
        atol=1e-6,
    )


def test_tensor_network_slicing_budget_fails_closed():
    node = TensorNetworkNode(
        torch.ones(2, 2),
        (0, 1),
        name="output",
    )
    plan = TensorNetworkContractionPlan(
        n_wires=1,
        bsz=1,
        nodes=(node,),
        output_labels=(0, 1),
        path=(),
    )

    with pytest.raises(ValueError, match="max_intermediate_size must be >= 1"):
        plan.slicing_plan(max_intermediate_size=0)
    with pytest.raises(ValueError, match="cannot satisfy"):
        plan.slicing_plan(max_intermediate_size=1)
    with pytest.raises(ValueError, match="max_intermediate_bytes must be >= 1"):
        plan.slicing_plan(max_intermediate_bytes=0)
    with pytest.raises(ValueError, match="cannot hold one"):
        plan.slicing_plan(max_intermediate_bytes=1)


def test_tensor_network_dry_run_uses_meta_for_huge_intermediate():
    dimension = 2**32
    nodes = (
        TensorNetworkNode(
            torch.empty(dimension, device="meta"),
            (0,),
            name="left",
        ),
        TensorNetworkNode(
            torch.empty(dimension, device="meta"),
            (1,),
            name="right",
        ),
    )
    plan = TensorNetworkContractionPlan(
        n_wires=0,
        bsz=1,
        nodes=nodes,
        output_labels=(0, 1),
        path=(),
    )

    profile = plan.contraction_profile("greedy")

    assert profile.peak_size == 2**64
    assert profile.estimated_cost == 2**64


def test_tensor_network_slicing_accepts_byte_budget_and_quality_path():
    circuit = fq.Circuit(4)
    circuit.h(0).cx(0, 3).ry(1, theta=0.2).rzz(2, 3, theta=-0.4)
    plan = build_tensor_network(circuit)
    baseline = plan.contraction_profile("quality_multistart")
    element_size = max(node.tensor.element_size() for node in plan.nodes)
    target_elements = max(baseline.output_size, baseline.peak_size // 2)

    slicing = plan.slicing_plan(
        max_intermediate_bytes=target_elements * element_size,
        contraction_strategy="quality_multistart",
    )

    assert slicing.target_peak_bytes == target_elements * element_size
    assert slicing.target_peak_size == target_elements
    assert slicing.peak_bytes <= slicing.target_peak_bytes
    assert slicing.baseline_estimated_cost == baseline.estimated_cost
    assert slicing.budget_satisfied
    assert slicing.reduction_method == "kahan_compensated"
    assert slicing.summary()["reduction_method"] == "kahan_compensated"
    profile = plan.contraction_profile(
        "quality_sliced",
        max_intermediate_bytes=target_elements * element_size,
    )
    assert profile.peak_size == slicing.peak_size
    assert profile.n_slices == slicing.n_slices
    assert torch.allclose(
        plan.contract(
            strategy="quality_sliced",
            max_intermediate_bytes=target_elements * element_size,
        ),
        plan.contract(strategy="greedy"),
        atol=1e-6,
    )


def test_quality_slicing_batch_selects_peak_labels_to_meet_budget():
    nodes = (
        TensorNetworkNode(
            torch.empty(2, 2, 2, 2),
            (0, 1, 2, 3),
            name="left",
        ),
        TensorNetworkNode(
            torch.empty(2, 2, 2, 2),
            (0, 1, 4, 5),
            name="right",
        ),
        TensorNetworkNode(
            torch.empty(2, 2, 2, 2),
            (2, 3, 6, 7),
            name="top",
        ),
        TensorNetworkNode(
            torch.empty(2, 2, 2, 2),
            (4, 5, 6, 7),
            name="bottom",
        ),
    )
    plan = TensorNetworkContractionPlan(0, 1, nodes, (), ())
    baseline = plan.contraction_profile("quality_multistart")

    slicing = plan.slicing_plan(
        max_intermediate_size=max(1, baseline.peak_size // 4),
    )

    assert slicing.budget_satisfied
    assert slicing.peak_size <= baseline.peak_size // 4
    assert slicing.n_slices >= 4


def test_quality_sliced_compensated_reduction_preserves_autograd():
    theta = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    circuit = fq.Circuit(4)
    circuit.h(0).cx(0, 3).ry(1, theta=theta).rzz(2, 3, theta=-0.4)
    ket_plan = build_tensor_network(circuit, dtype=torch.complex128)
    assert {node.tensor.dtype for node in ket_plan.nodes} == {torch.complex128}
    plan = build_tensor_network_expectation(ket_plan, z=[0, 1, 2, 3])
    baseline = plan.contraction_profile("quality_multistart")
    target_peak = max(baseline.output_size, baseline.peak_size // 2)

    sliced = plan.contract(
        strategy="quality_sliced",
        max_intermediate_size=target_peak,
    )
    reference = plan.contract(strategy="greedy")
    sliced_gradient = torch.autograd.grad(sliced.real, theta, retain_graph=True)[0]
    reference_gradient = torch.autograd.grad(reference.real, theta)[0]

    assert torch.allclose(sliced, reference, atol=1e-12, rtol=1e-10)
    assert torch.allclose(
        sliced_gradient,
        reference_gradient,
        atol=1e-12,
        rtol=1e-10,
    )


def test_cotengra_slicing_plan_executes_external_path_with_autograd():
    pytest.importorskip("cotengra")
    theta = torch.tensor(0.19, dtype=torch.float64, requires_grad=True)
    circuit = fq.Circuit(3, dtype=torch.complex128)
    circuit.h(0).ry(1, theta=theta).cx(0, 2).rzz(1, 2, theta=-0.27)
    plan = build_tensor_network_expectation(circuit, x=(0,), z=(2,))
    baseline = plan.contraction_profile("greedy")
    slicing = plan.cotengra_slicing_plan(
        target_peak_elements=max(baseline.output_size, baseline.peak_size // 2),
        max_repeats=1,
    )

    actual = plan.contract_slicing_plan(slicing)
    reference = plan.contract(strategy="greedy")
    actual_gradient = torch.autograd.grad(actual.real, theta, retain_graph=True)[0]
    reference_gradient = torch.autograd.grad(reference.real, theta)[0]

    torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(
        actual_gradient, reference_gradient, atol=1e-12, rtol=1e-10
    )


def test_tensor_network_expectation_greedy_path_matches_einsum():
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 2).ry(1, theta=0.2)
    expectation_plan = build_tensor_network_expectation(circuit, x=[0, 2], z=[1])

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
    expectation_plan = build_tensor_network_expectation(circuit, x=[0, 3], z=[1])

    profile = expectation_plan.contraction_profile("memory_greedy")

    assert isinstance(profile, TensorNetworkContractionProfile)
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
    expectation_plan = build_tensor_network_expectation(circuit, x=[0, 3], z=[1])

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
    expectation_plan = build_tensor_network_expectation(circuit, z=[0, 1])

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
    expectation_plan = build_tensor_network_expectation(circuit, x=[0, 2], z=[1])
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
    plan = build_tensor_network(circuit)
    profile = plan.contraction_profile("greedy")
    target_peak = max(profile.output_size, profile.peak_size // 2)

    tn = fqr.run_native(
        circuit,
        mode="tensor_network",
        contraction_strategy="sliced",
        max_intermediate_size=target_peak,
    )

    assert tn.contraction_strategy == "sliced"
    assert tn.max_intermediate_size == target_peak
    assert torch.allclose(tn.state(), circuit.state(), atol=1e-6)


def test_greedy_contraction_path_is_reused_for_dynamic_parameters():
    _CONTRACTION_PATH_CACHE.clear()
    theta = torch.tensor(0.2, requires_grad=True)
    first = fqtn.run_tensor_network(fq.Circuit(2).ry(0, theta).cx(0, 1))
    first.expectation_z(0).sum().backward()
    cache_size = len(_CONTRACTION_PATH_CACHE)

    phi = torch.tensor(-0.3, requires_grad=True)
    second = fqtn.run_tensor_network(fq.Circuit(2).ry(0, phi).cx(0, 1))
    second.expectation_z(0).sum().backward()

    assert cache_size > 0
    assert len(_CONTRACTION_PATH_CACHE) == cache_size
    assert theta.grad is not None and phi.grad is not None


def test_compiled_contraction_stage_plan_prebuckets_matching_operations():
    nodes = (
        TensorNetworkNode(torch.ones(2, 2), (0, 1)),
        TensorNetworkNode(torch.ones(2, 2), (1, 2)),
        TensorNetworkNode(torch.ones(2, 2), (0, 3)),
        TensorNetworkNode(torch.ones(2, 2), (3, 4)),
    )
    path = (
        (0, 1, (0, 2)),
        (0, 1, (0, 4)),
        (0, 1, (2, 4)),
    )

    plan = _compile_contraction_stages(nodes, path)

    assert plan.output_labels == (2, 4)
    assert len(plan.stages) == 2
    assert len(plan.stages[0].buckets) == 1
    assert len(plan.stages[0].buckets[0].operations) == 2
    assert plan.stages[0].buckets[0].batched_equation.startswith("Z")


def test_small_tn_reuses_one_state_contraction_for_many_z_observables():
    theta = torch.tensor(0.2, requires_grad=True)
    state = fqtn.run_tensor_network(
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
    state = fqtn.run_tensor_network(
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
        CompiledTNObservableProgram,
    )
    assert theta.grad is not None


def test_compiled_z_observable_program_is_reused_across_runs():
    circuit = fq.Circuit(3).ry(0, 0.2).cx(0, 1).cx(1, 2)

    first = fqtn.run_tensor_network(circuit, dense_observable_wires=0)
    first_values = first.expectation_z((0, 1, 2))
    second = fqtn.run_tensor_network(circuit, dense_observable_wires=0)
    second_values = second.expectation_z((0, 1, 2))

    assert first.summary()["observable_program_cache_hit"] is False
    assert second.summary()["observable_program_cache_hit"] is True
    assert second.summary()["observable_peak_size"] is not None
    assert torch.allclose(first_values, second_values)


def test_plan_accepts_tensor_network_state_mode():
    circuit = fq.Circuit(4)
    circuit.h(0).cx(0, 3)

    plan = fqxp.plan_advanced(circuit, state_mode="tensor_network")

    assert plan.state_mode == "tensor_network"
    assert plan.recommended_mode == "tensor_network"
    assert plan.state_bytes == estimate_tensor_network_bytes(4)


@pytest.mark.parametrize("bitstring", [0, 3, "101", (1, 1, 0)])
def test_tensor_network_single_amplitude_matches_dense_state(bitstring):
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 2).ry(1, theta=0.37)
    index = (
        bitstring
        if isinstance(bitstring, int)
        else int(
            bitstring if isinstance(bitstring, str) else "".join(map(str, bitstring)),
            2,
        )
    )

    amplitude = fqtn.tensor_network_amplitude(circuit, bitstring)

    assert torch.allclose(amplitude, circuit.state()[:, index], atol=1e-6)


def test_sliced_single_amplitude_budget_excludes_full_state_output():
    circuit = fq.Circuit(5)
    circuit.h(0).cx(0, 4).ry(2, theta=0.2).rzz(1, 3, theta=-0.4)

    amplitude = fqtn.tensor_network_amplitude(
        circuit,
        "10001",
        max_intermediate_size=8,
    )

    assert torch.allclose(amplitude, circuit.state()[:, 0b10001], atol=1e-6)


def test_tensor_network_single_amplitude_validates_bitstrings():
    circuit = fq.Circuit(3)

    with pytest.raises(ValueError, match="bitstring"):
        fqtn.tensor_network_amplitude(circuit, "01")
    with pytest.raises(ValueError, match="state space"):
        fqtn.tensor_network_amplitude(circuit, 8)


def test_tensor_network_single_amplitude_preserves_parameter_gradients():
    theta = torch.tensor(0.37, requires_grad=True)
    circuit = fq.Circuit(3)
    circuit.h(0).ry(1, theta=theta).cx(1, 2)

    amplitude = fqtn.tensor_network_amplitude(circuit, "011")
    loss = amplitude.abs().square().sum()
    (gradient,) = torch.autograd.grad(loss, (theta,))

    reference_theta = theta.detach().clone().requires_grad_(True)
    reference = fq.Circuit(3)
    reference.h(0).ry(1, theta=reference_theta).cx(1, 2)
    reference_loss = reference.state()[:, 0b011].abs().square().sum()
    (reference_gradient,) = torch.autograd.grad(reference_loss, (reference_theta,))

    assert torch.allclose(gradient, reference_gradient, atol=1e-6)


def test_distributed_single_amplitude_reduces_only_sparse_output():
    circuit = fq.Circuit(5)
    circuit.h(0).cx(0, 4).ry(2, theta=0.2).rzz(1, 3, theta=-0.4)

    result = fqxd.distributed_tensor_network_amplitude(
        circuit,
        "10001",
        world_size=2,
        distributed_profile="development",
        torch_backend="local_tensor",
        max_intermediate_size=8,
    )
    summary = result.summary()

    assert torch.allclose(result.value, circuit.state()[:, 0b10001], atol=1e-6)
    assert summary["output_target"] == "single_amplitude"
    assert summary["full_state_materialized"] is False
    assert (
        summary["reduction_payload_bytes"]
        == result.value.numel() * result.value.element_size()
    )
    assert summary["distribution_semantics"] == (
        "local_simulated_slice_parallel_sparse_output_reduction"
    )
    preflight = summary["working_set_preflight"]
    assert preflight["model"] == "calibrated_peak_safety_factor"
    assert preflight["budget_satisfied"] is True
    assert preflight["fail_closed"] is True
    assert preflight["predicted_working_set_bytes"] == (
        preflight["tensor_peak_bytes"] * 4
    )


def test_distributed_single_amplitude_local_simulator_preserves_gradients():
    theta = torch.tensor(0.31, requires_grad=True)
    circuit = fq.Circuit(4)
    circuit.h(0).ry(1, theta=theta).cx(1, 3)

    result = fqxd.distributed_tensor_network_amplitude(
        circuit,
        "0101",
        world_size=2,
        distributed_profile="development",
        torch_backend="local_tensor",
    )
    (gradient,) = torch.autograd.grad(result.value.abs().square().sum(), (theta,))

    reference_theta = theta.detach().clone().requires_grad_(True)
    reference = fq.Circuit(4)
    reference.h(0).ry(1, theta=reference_theta).cx(1, 3)
    reference_loss = reference.state()[:, 0b0101].abs().square().sum()
    (reference_gradient,) = torch.autograd.grad(reference_loss, (reference_theta,))

    assert torch.allclose(gradient, reference_gradient, atol=1e-6)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"max_slices": 2}, "slice_count=4"),
        ({"max_recomputation_factor": 2.0}, "recomputation_factor"),
    ],
)
def test_single_amplitude_rejects_uneconomic_slicing(options, message):
    circuit = fq.Circuit(5)
    circuit.h(0).cx(0, 4).ry(2, theta=0.2).rzz(1, 3, theta=-0.4)

    with pytest.raises(ValueError, match=message):
        fqtn.tensor_network_amplitude(
            circuit,
            "10001",
            max_intermediate_size=4,
            **options,
        )


def test_distributed_single_amplitude_rejects_before_rank_execution():
    circuit = fq.Circuit(5)
    circuit.h(0).cx(0, 4).ry(2, theta=0.2).rzz(1, 3, theta=-0.4)

    with pytest.raises(ValueError, match="slice_count=4"):
        fqxd.distributed_tensor_network_amplitude(
            circuit,
            "10001",
            world_size=2,
            distributed_profile="development",
            torch_backend="local_tensor",
            max_intermediate_size=4,
            max_slices=2,
        )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"max_working_set_bytes": 0}, "max_working_set_bytes must be positive"),
        (
            {"max_working_set_bytes": 1024, "working_set_safety_factor": 0.5},
            "working_set_safety_factor must be >= 1",
        ),
    ],
)
def test_distributed_single_amplitude_validates_working_set_policy(
    options,
    message,
):
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 2)

    with pytest.raises(ValueError, match=message):
        fqxd.distributed_tensor_network_amplitude(
            circuit,
            "101",
            world_size=1,
            distributed_profile="development",
            torch_backend="local_tensor",
            **options,
        )


def test_distributed_single_amplitude_enforces_explicit_working_set_components():
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 2)
    policy = DistributedTNWorkingSetPolicy(
        kernel_workspace_output_multiplier=2.0,
        communication_buffer_output_multiplier=3.0,
        allocator_headroom_fraction=0.25,
        minimum_allocator_headroom_bytes=4096,
    )

    result = fqxd.distributed_tensor_network_amplitude(
        circuit,
        "101",
        world_size=1,
        distributed_profile="development",
        torch_backend="local_tensor",
        working_set_policy=policy,
    )
    preflight = result.summary()["working_set_preflight"]

    assert preflight["model"] == "explicit_end_to_end_components"
    assert preflight["working_set_policy"]["allocator_headroom_fraction"] == 0.25
    assert preflight["kernel_workspace_bytes"] > 0
    assert preflight["communication_buffer_bytes"] > 0
    assert preflight["allocator_headroom_bytes"] >= 4096
    assert preflight["predicted_working_set_bytes"] == sum(
        preflight[field]
        for field in (
            "tensor_peak_bytes",
            "kernel_workspace_bytes",
            "communication_buffer_bytes",
            "allocator_headroom_bytes",
        )
    )
    with pytest.raises(ValueError, match="working-set preflight rejected"):
        fqxd.distributed_tensor_network_amplitude(
            circuit,
            "101",
            world_size=1,
            distributed_profile="development",
            torch_backend="local_tensor",
            max_working_set_bytes=1024,
            working_set_policy=policy,
        )


def test_distributed_single_amplitude_applies_scoped_memory_calibration():
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 2)
    calibration = build_tn_working_set_calibration(
        (
            {
                "predicted_working_set_bytes": predicted,
                "cuda_peak_allocated_bytes": predicted + 128,
                "cuda_peak_reserved_bytes": predicted + 256,
            }
            for predicted in (1024, 2048, 4096)
        ),
        accelerator_name="cpu",
        complex_bytes=8,
        world_size=1,
        topology_class="single_device_test",
    )

    result = fqxd.distributed_tensor_network_amplitude(
        circuit,
        "101",
        world_size=1,
        distributed_profile="development",
        torch_backend="local_tensor",
        memory_calibration=calibration,
    )
    preflight = result.summary()["working_set_preflight"]

    assert preflight["model"] == "measured_reserved_memory_calibration"
    assert preflight["memory_calibration_identity"] == calibration.identity
    with pytest.raises(ValueError, match="scope mismatch"):
        fqxd.distributed_tensor_network_amplitude(
            circuit,
            "101",
            world_size=2,
            distributed_profile="development",
            torch_backend="local_tensor",
            memory_calibration=calibration,
        )


def test_distributed_local_observable_avoids_full_state_materialization():
    circuit = fq.Circuit(6)
    circuit.h(0).cx(0, 5).ry(2, theta=0.2).rzz(1, 3, theta=-0.4)

    result = fqxd.distributed_tensor_network_expectation(
        circuit,
        x=(0,),
        z=(2, 5),
        world_size=2,
        distributed_profile="development",
        torch_backend="local_tensor",
    )
    summary = result.summary()

    assert torch.allclose(
        result.value,
        circuit.expectation_ps(x=(0,), z=(2, 5)),
        atol=1e-6,
    )
    assert summary["output_target"] == "local_observables"
    assert summary["observable_wires"] == (0, 2, 5)
    assert summary["full_state_materialized"] is False
    assert (
        summary["reduction_payload_bytes"]
        == result.value.numel() * result.value.element_size()
    )


def test_distributed_local_observable_preserves_parameter_gradients():
    theta = torch.tensor(0.31, requires_grad=True)
    circuit = fq.Circuit(4)
    circuit.h(0).ry(1, theta=theta).cx(1, 3)

    result = fqxd.distributed_tensor_network_expectation(
        circuit,
        z=(1, 3),
        world_size=2,
        distributed_profile="development",
        torch_backend="local_tensor",
    )
    (gradient,) = torch.autograd.grad(result.value.sum(), (theta,))

    reference_theta = theta.detach().clone().requires_grad_(True)
    reference = fq.Circuit(4)
    reference.h(0).ry(1, theta=reference_theta).cx(1, 3)
    reference_loss = reference.expectation_ps(z=(1, 3)).sum()
    (reference_gradient,) = torch.autograd.grad(reference_loss, (reference_theta,))

    assert torch.allclose(gradient, reference_gradient, atol=1e-6)


def test_tensor_network_amplitude_batch_matches_dense_gather():
    circuit = fq.Circuit(6)
    circuit.h(0).cx(0, 5).ry(2, theta=0.2).rzz(1, 3, theta=-0.4)
    bitstrings = ("000000", "100001", "010100", "111111")
    indices = torch.tensor([int(bits, 2) for bits in bitstrings])

    values = fqtn.tensor_network_amplitudes(circuit, bitstrings)

    assert values.shape == (1, len(bitstrings))
    assert torch.allclose(values, circuit.state()[:, indices], atol=1e-6)


def test_distributed_amplitude_batch_uses_one_shared_reduction():
    circuit = fq.Circuit(6)
    circuit.h(0).cx(0, 5).ry(2, theta=0.2).rzz(1, 3, theta=-0.4)
    bitstrings = ("000000", "100001", "010100", "111111")
    indices = torch.tensor([int(bits, 2) for bits in bitstrings])

    result = fqxd.distributed_tensor_network_amplitudes(
        circuit,
        bitstrings,
        world_size=2,
        distributed_profile="development",
        torch_backend="local_tensor",
    )
    summary = result.summary()

    assert torch.allclose(result.values, circuit.state()[:, indices], atol=1e-6)
    assert summary["target_count"] == len(bitstrings)
    assert summary["shared_contraction"] is True
    assert summary["full_state_materialized"] is False
    assert summary["reduction_payload_bytes"] == (
        result.values.numel() * result.values.element_size()
    )


def test_tensor_network_amplitude_batch_preserves_shared_gradient():
    theta = torch.tensor(0.31, requires_grad=True)
    circuit = fq.Circuit(4)
    circuit.h(0).ry(1, theta=theta).cx(1, 3)
    bitstrings = ("0000", "0101", "1111")

    values = fqtn.tensor_network_amplitudes(circuit, bitstrings)
    (gradient,) = torch.autograd.grad(values.abs().square().sum(), (theta,))

    reference_theta = theta.detach().clone().requires_grad_(True)
    reference = fq.Circuit(4)
    reference.h(0).ry(1, theta=reference_theta).cx(1, 3)
    indices = torch.tensor([int(bits, 2) for bits in bitstrings])
    reference_loss = reference.state()[:, indices].abs().square().sum()
    (reference_gradient,) = torch.autograd.grad(reference_loss, (reference_theta,))

    assert torch.allclose(gradient, reference_gradient, atol=1e-6)


def test_tensor_network_observable_batch_matches_individual_expectations():
    circuit = fq.Circuit(6)
    circuit.h(0).cx(0, 5).ry(2, theta=0.2).rzz(1, 3, theta=-0.4)
    observables = (
        {"x": (0,), "z": (2, 5)},
        {"y": (1, 3)},
        {"z": (0, 1, 2)},
        {},
    )

    values = fqtn.tensor_network_expectations(circuit, observables)
    reference = torch.stack(
        [
            circuit.expectation_ps(
                x=observable.get("x"),
                y=observable.get("y"),
                z=observable.get("z"),
            )
            for observable in observables
        ],
        dim=-1,
    )

    assert values.shape == (1, len(observables))
    assert torch.allclose(values, reference, atol=1e-6)


def test_tensor_network_observable_batch_preserves_shared_gradient():
    theta = torch.tensor(0.31, requires_grad=True)
    circuit = fq.Circuit(4)
    circuit.h(0).ry(1, theta=theta).cx(1, 3)
    observables = ({"z": (1,)}, {"x": (0,), "z": (3,)})

    values = fqtn.tensor_network_expectations(circuit, observables)
    (gradient,) = torch.autograd.grad(values.sum(), (theta,))

    reference_theta = theta.detach().clone().requires_grad_(True)
    reference = fq.Circuit(4)
    reference.h(0).ry(1, theta=reference_theta).cx(1, 3)
    reference_loss = sum(
        reference.expectation_ps(
            x=observable.get("x"),
            z=observable.get("z"),
        ).sum()
        for observable in observables
    )
    (reference_gradient,) = torch.autograd.grad(reference_loss, (reference_theta,))

    assert torch.allclose(gradient, reference_gradient, atol=1e-6)


def test_distributed_observable_batch_uses_one_shared_reduction():
    circuit = fq.Circuit(6)
    circuit.h(0).cx(0, 5).ry(2, theta=0.2).rzz(1, 3, theta=-0.4)
    observables = (
        {"x": (0,), "z": (2, 5)},
        {"y": (1, 3)},
        {"z": (0, 1, 2)},
    )

    result = fqxd.distributed_tensor_network_expectations(
        circuit,
        observables,
        world_size=2,
        distributed_profile="development",
        torch_backend="local_tensor",
    )
    reference = fqtn.tensor_network_expectations(circuit, observables)
    summary = result.summary()

    assert torch.allclose(result.values, reference, atol=1e-6)
    assert summary["observable_count"] == len(observables)
    assert summary["shared_contraction"] is True
    assert summary["full_state_materialized"] is False
    assert summary["reduction_payload_bytes"] == (
        result.values.numel() * result.values.element_size()
    )


def test_parallel_slice_planner_reduces_rank_local_cost_vs_label_order():
    circuit = fq.Circuit(12)
    for layer in range(4):
        for wire in range(12):
            circuit.ry(wire, theta=0.01)
        for wire in range(layer % 2, 11, 2):
            circuit.cx(wire, wire + 1)
    plan = build_tensor_network(circuit)
    nodes, outputs = tensor_execution._amplitude_batch_projection(
        plan,
        ("0" * 12,),
    )
    candidates = _internal_slice_candidates(nodes, outputs)

    selected = _parallel_slice_labels(nodes, outputs, 8)
    selected_cost = _cost_for_sliced_labels(
        nodes,
        outputs,
        selected,
        contraction_strategy="quality_multistart",
    )
    label_order_cost = _cost_for_sliced_labels(
        nodes,
        outputs,
        candidates[:3],
        contraction_strategy="quality_multistart",
    )

    assert len(selected) == 3
    assert selected_cost["estimated_cost"] < label_order_cost["estimated_cost"]


def test_tensor_network_hamiltonian_mpo_matches_termwise_value_and_gradient():
    parameters = torch.tensor([0.23, -0.31], dtype=torch.float64, requires_grad=True)
    reference_parameters = parameters.detach().clone().requires_grad_(True)
    hamiltonian = Hamiltonian(
        (
            pauli_term(-0.7, "ZZ", (0, 1)),
            pauli_term(0.2, "X", (0,)),
            pauli_term(-0.13, "Y", (1,)),
        )
    )

    def circuit(values):
        return (
            fq.Circuit(2, dtype=torch.complex128)
            .ry(0, values[0])
            .rx(1, values[1])
            .cx(0, 1)
        )

    plan = build_tensor_network_hamiltonian_expectation(
        circuit(parameters), hamiltonian
    )
    actual = plan.contract(strategy="greedy").real.sum()
    reference = hamiltonian.expectation(circuit(reference_parameters)).sum()
    actual_gradient = torch.autograd.grad(actual, parameters)[0]
    reference_gradient = torch.autograd.grad(reference, reference_parameters)[0]

    torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        actual_gradient, reference_gradient, atol=1e-12, rtol=1e-12
    )


def test_tensor_network_hamiltonian_block_mpo_matches_individual_expectations():
    parameter = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    circuit = fq.Circuit(2, dtype=torch.complex128).ry(0, parameter).cx(0, 1)
    observables = (
        Hamiltonian((pauli_term(0.5, "ZZ", (0, 1)),)),
        Hamiltonian((pauli_term(-0.7, "X", (0,)), pauli_term(0.2, "Z", (1,)))),
    )

    plan = build_tensor_network_hamiltonian_expectations(circuit, observables)
    actual = plan.contract(strategy="greedy").real.squeeze(0)
    reference = torch.stack(
        tuple(observable.expectation(circuit).sum() for observable in observables)
    )

    torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
