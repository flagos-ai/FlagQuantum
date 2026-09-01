import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import flagquantum as fq
import flagquantum.backends as fqb
from flagquantum.runtime.execution import run_advanced

pytestmark = [pytest.mark.distributed, pytest.mark.distributed_cpu]


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_static_gate_basis_owner_matches_tensor_index_mapping(world_size):
    from flagquantum.runtime.backends.statevector.forward import (
        _basis_owner_rank,
        _owner_and_local,
    )
    from flagquantum.runtime.backends.statevector.state import _basis_offset, _wire_mask

    plan = fq.plan_distributed_statevector(fq.Circuit(6), world_size=world_size)
    wire_sets = (
        (0, plan.sharded_wires[-1]),
        (plan.sharded_wires[-1], 1),
        tuple(reversed(plan.sharded_wires)),
    )
    rank_bits = len(plan.sharded_wires)
    for rank in range(world_size):
        global_index = torch.tensor([(1 << rank_bits) | rank])
        for wires in wire_sets:
            clear_mask = ~sum(_wire_mask(plan.n_wires, wire) for wire in wires)
            for basis in range(2 ** len(wires)):
                required = (global_index & clear_mask) | _basis_offset(
                    plan.n_wires, wires, basis
                )
                owners, _ = _owner_and_local(required, plan=plan)
                assert _basis_owner_rank(plan, rank, wires, basis) == int(owners[0])


def test_diagonal_gate_on_sharded_wire_stays_rank_local():
    from flagquantum.runtime.backends.statevector.forward import (
        _vectorized_local_diagonal_gate,
    )
    from flagquantum.runtime.backends.statevector.state import (
        _instruction_matrix,
        initialize_statevector_shard,
    )

    generator = torch.Generator().manual_seed(4401)
    dense = torch.randn(1, 32, dtype=torch.complex64, generator=generator)
    theta = torch.tensor(0.37)
    circuit = fq.Circuit(5, inputs=dense).rz(4, theta).rzz(3, 4, theta * 0.5)
    expected = circuit.state()
    plan = fq.plan_distributed_statevector(circuit, world_size=4)

    assert plan.sharded_wires == (3, 4)
    for rank in range(plan.world_size):
        shard = initialize_statevector_shard(
            plan, rank=rank, device="cpu", dtype=torch.complex64
        )
        shard = replace(shard, amplitudes=dense[:, shard.global_indices].clone())
        for instruction in circuit.to_ir().instructions:
            matrix = _instruction_matrix(
                instruction, device=torch.device("cpu"), dtype=torch.complex64
            )
            shard, scratch = _vectorized_local_diagonal_gate(
                shard,
                matrix,
                instruction.wires,
                plan=plan,
                chunk_amplitudes=3,
            )
            assert scratch >= shard.amplitudes.numel() * shard.amplitudes.element_size()
        torch.testing.assert_close(
            shard.amplitudes,
            expected[:, shard.global_indices],
            atol=1e-6,
            rtol=1e-6,
        )


def test_distributed_statevector_plan_marks_sharded_wires_and_communication():
    circuit = fq.Circuit(5)
    circuit.h(0).rz(4, theta=0.1).x(4).cx(3, 0).cx(0, 4)

    plan = fq.plan_distributed_statevector(circuit, world_size=4, bsz=2)

    assert plan.distribution == "qubit_address_sharded"
    assert plan.summary()["distribution_semantics"] == "sharded_across_ranks"
    assert plan.summary()["scalability_claim_allowed"] is False
    assert plan.summary()["claim_evidence_type"] == "plan_preflight"
    assert plan.summary()["sharding_plan_available"] is True
    assert (
        plan.summary()["statevector_training_claimability_status"] == "preflight_only"
    )
    assert plan.summary()["claimable_production_training"] is False
    assert (
        plan.summary()["statevector_training_claimability_gate"]["checks"][
            "memory_plan_has_per_rank_shard_and_comm_buffer"
        ]
        is True
    )
    assert plan.rank_address_bits == 2
    assert plan.sharded_wires == (3, 4)
    assert plan.total_state_bytes == 2 * (2**5) * 8
    assert plan.per_rank_state_bytes == 2 * (2**3) * 8
    assert [gate.communication for gate in plan.gate_plans] == [
        "local",
        "local",
        "pair_exchange",
        "local",
        "all_to_all",
    ]
    assert plan.communication_gate_count == 2
    assert plan.estimated_transfer_bytes > 0


def test_distributed_statevector_preflight_is_not_production_training_claim():
    circuit = fq.Circuit(5)
    circuit.h(0).rz(4, theta=0.1).cx(0, 4)

    summary = fq.plan_distributed_statevector(circuit, world_size=4, bsz=2).summary()

    assert summary["distribution_semantics"] == "sharded_across_ranks"
    assert summary["claim_evidence_type"] == "plan_preflight"
    assert summary["scalability_claim_allowed"] is False
    assert summary["sharding_plan_available"] is True
    assert summary["claimable_production_training"] is False
    assert (
        summary["statevector_training_claimability_gate"]["checks"][
            "forward_semantics_sharded"
        ]
        is True
    )
    assert (
        summary["statevector_training_claimability_gate"]["checks"][
            "backward_semantics_sharded"
        ]
        is False
    )
    assert (
        summary["statevector_training_claimability_gate"]["checks"][
            "parameter_gradient_available"
        ]
        is False
    )
    assert (
        summary["statevector_training_claimability_gate"]["checks"][
            "optimizer_step_preserves_sharded_ownership"
        ]
        is False
    )
    with pytest.raises(fq.DistributedScalabilityError):
        fq.require_distributed_scalability(summary)


def test_distributed_statevector_plan_builds_fusion_barriers():
    circuit = fq.Circuit(4)
    circuit.h(0).rx(1, theta=0.2).x(3).h(2)

    plan = fq.plan_distributed_statevector(
        circuit,
        world_size=2,
        max_fusion_gate_width=2,
    )

    assert plan.sharded_wires == (3,)
    assert [block.communication_barrier for block in plan.fusion_blocks] == [
        False,
        True,
        False,
    ]
    assert plan.fusion_blocks[0].gate_indices == (0, 1)
    assert plan.fusion_blocks[1].gate_indices == (2,)
    assert plan.fusion_blocks[2].gate_indices == (3,)


def test_distributed_statevector_plan_batches_communication_segments():
    circuit = fq.Circuit(4)
    circuit.h(0).x(3).rx(3, theta=0.2).h(1).cx(0, 3)

    plan = fq.plan_distributed_statevector(circuit, world_size=2)

    assert isinstance(plan.execution_segments[1], fq.StatevectorExecutionSegment)
    assert [segment.kind for segment in plan.execution_segments] == [
        "local_fusion",
        "communication_batch",
        "local_fusion",
        "communication_batch",
    ]
    assert plan.execution_segments[1].communication == "pair_exchange"
    assert plan.execution_segments[1].gate_indices == (1, 2)
    assert plan.execution_segments[1].can_overlap_with_compute
    assert plan.execution_segments[3].communication == "all_to_all"
    assert plan.summary()["execution_segment_count"] == 4
    assert plan.summary()["communication_segment_count"] == 2
    assert plan.summary()["overlap_candidate_count"] == 2


def test_distributed_statevector_pair_exchange_uses_touched_sharded_wire():
    circuit = fq.Circuit(5)
    circuit.x(3).rx(3, theta=0.2).x(4)

    plan = fq.plan_distributed_statevector(circuit, world_size=4)
    pair_segments = [
        segment
        for segment in plan.execution_segments
        if segment.communication == "pair_exchange"
    ]

    assert plan.sharded_wires == (3, 4)
    assert [segment.gate_indices for segment in pair_segments] == [(0, 1), (2,)]
    first_edges = {
        (edge.src_rank, edge.dst_rank)
        for edge in plan.topology.edges
        if edge.segment_indices == (pair_segments[0].index,)
    }
    second_edges = {
        (edge.src_rank, edge.dst_rank)
        for edge in plan.topology.edges
        if edge.segment_indices == (pair_segments[1].index,)
    }
    assert first_edges == {(0, 2), (1, 3)}
    assert second_edges == {(0, 1), (2, 3)}


def test_distributed_statevector_performance_estimate():
    circuit = fq.Circuit(5)
    circuit.x(4).rx(4, theta=0.2).cx(0, 4)
    plan = fq.plan_distributed_statevector(circuit, world_size=2)

    estimate = plan.estimate_performance(
        bandwidth_bytes_per_second=1_000.0,
        local_gate_amplitudes_per_second=1_000_000.0,
    )
    estimate_from_function = fq.estimate_distributed_statevector_performance(
        plan,
        bandwidth_bytes_per_second=1_000.0,
        local_gate_amplitudes_per_second=1_000_000.0,
    )

    assert isinstance(estimate, fq.StatevectorPerformanceEstimate)
    assert estimate.summary()["bottleneck"] == "communication"
    assert estimate.communication_seconds > estimate.compute_seconds
    assert estimate.overlapped_seconds > 0
    assert estimate_from_function == estimate


def test_distributed_statevector_topology_and_buffer_plan():
    circuit = fq.Circuit(5)
    circuit.x(4).rx(4, theta=0.2).cx(0, 4)

    plan = fq.plan_distributed_statevector(circuit, world_size=4)

    assert isinstance(plan.topology, fq.StatevectorRankTopology)
    assert isinstance(plan.topology.edges[0], fq.StatevectorCommunicationEdge)
    assert isinstance(plan.buffer_plans[0], fq.StatevectorBufferPlan)
    assert plan.topology.layout == "hypercube"
    assert plan.topology.rank_coordinates == ((0, 0), (0, 1), (1, 0), (1, 1))
    assert {edge.communication for edge in plan.topology.edges} == {
        "pair_exchange",
        "all_to_all",
    }
    assert all(buffer_plan.peak_bytes > 0 for buffer_plan in plan.buffer_plans)
    assert plan.summary()["topology_layout"] == "hypercube"
    assert plan.summary()["topology_edge_count"] == len(plan.topology.edges)
    assert plan.summary()["peak_buffer_bytes"] == max(
        buffer_plan.peak_bytes for buffer_plan in plan.buffer_plans
    )


def test_distributed_statevector_topology_tracks_multi_node_tiers():
    circuit = fq.Circuit(5)
    circuit.x(3).x(4).cx(0, 4)

    plan = fq.plan_distributed_statevector(circuit, world_size=4, local_world_size=2)
    summary = plan.summary()
    topology_summary = plan.topology.summary()

    assert summary["distribution_semantics"] == "sharded_across_ranks"
    assert summary["local_world_size"] == 2
    assert summary["node_count"] == 2
    assert topology_summary["local_world_size"] == 2
    assert topology_summary["node_count"] == 2
    assert "intra_node" in topology_summary["communication_tiers"]
    assert "inter_node" in topology_summary["communication_tiers"]
    assert summary["intra_node_communication_bytes"] > 0
    assert summary["inter_node_communication_bytes"] > 0
    assert all(
        edge.tier in {"intra_node", "inter_node"} for edge in plan.topology.edges
    )


def test_distributed_statevector_trace_and_validation_report():
    circuit = fq.Circuit(4)
    circuit.h(0).x(3).rx(3, theta=0.2).cx(0, 3)

    plan = fq.plan_distributed_statevector(circuit, world_size=2)
    trace = plan.trace()
    validation = fq.validate_distributed_statevector_plan(plan)
    trace_from_function = fq.trace_distributed_statevector_plan(plan)

    assert isinstance(trace, fq.StatevectorTraceReport)
    assert isinstance(trace.events[0], fq.StatevectorTraceEvent)
    assert trace.valid
    assert trace.errors == ()
    assert validation.valid
    assert trace_from_function == trace
    assert trace.per_rank_event_counts == (len(plan.execution_segments),) * 2
    assert (
        trace.summary()["event_count"] == len(plan.execution_segments) * plan.world_size
    )
    assert max(trace.peak_rank_buffer_bytes) == plan.summary()["peak_buffer_bytes"]

    communication_events = [
        event for event in trace.events if event.action == "exchange_statevector_slices"
    ]
    assert communication_events
    assert all(event.peer_ranks for event in communication_events)
    assert all(event.buffer_bytes > 0 for event in communication_events)


def test_distributed_statevector_dry_run_executor_report():
    circuit = fq.Circuit(4)
    circuit.h(0).x(3).rx(3, theta=0.2).cx(0, 3)

    plan = fq.plan_distributed_statevector(circuit, world_size=2)
    report = plan.execute_dry_run()
    report_from_function = fq.execute_distributed_statevector_dry_run(plan)

    assert isinstance(report, fq.StatevectorExecutorReport)
    assert isinstance(report.rank_results[0], fq.StatevectorRankResult)
    assert isinstance(
        report.rank_results[0].segment_results[0], fq.StatevectorSegmentResult
    )
    assert report.valid
    assert report.errors == ()
    assert report_from_function == report
    assert report.total_local_bytes_processed > 0
    assert report.total_communication_bytes > 0
    assert report.peak_buffer_bytes == plan.summary()["peak_buffer_bytes"]
    assert report.summary()["trace_event_count"] == len(plan.trace().events)


def test_statevector_shard_state_applies_local_gate_without_full_state():
    circuit = fq.Circuit(3)
    circuit.h(0)
    plan = fq.plan_distributed_statevector(circuit, world_size=2)
    h_matrix = torch.tensor(
        [[1.0, 1.0], [1.0, -1.0]],
        dtype=torch.complex64,
    ) / torch.sqrt(torch.tensor(2.0, dtype=torch.complex64))

    rank0 = fq.initialize_statevector_shard(plan, rank=0, device="cpu")
    rank1 = fq.initialize_statevector_shard(plan, rank=1, device="cpu")
    updated0 = fq.apply_gate_to_statevector_shard(rank0, h_matrix, (0,), plan=plan)
    updated1 = fq.apply_gate_to_statevector_shard(rank1, h_matrix, (0,), plan=plan)
    reconstructed = torch.zeros((1, plan.total_amplitudes), dtype=torch.complex64)
    reconstructed[:, updated0.global_indices] = updated0.amplitudes
    reconstructed[:, updated1.global_indices] = updated1.amplitudes

    assert isinstance(updated0, fq.StatevectorShardState)
    assert updated0.summary()["local_amplitudes"] == 4
    assert tuple(updated0.global_indices.tolist()) == (0, 2, 4, 6)
    assert tuple(updated1.global_indices.tolist()) == (1, 3, 5, 7)
    assert torch.allclose(reconstructed, circuit.state(), atol=1e-6)


def test_statevector_shard_state_rejects_cross_shard_gate():
    circuit = fq.Circuit(3)
    plan = fq.plan_distributed_statevector(circuit, world_size=2)
    x_matrix = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex64)
    rank0 = fq.initialize_statevector_shard(plan, rank=0, device="cpu")

    with pytest.raises(ValueError, match="communication is required"):
        fq.apply_gate_to_statevector_shard(rank0, x_matrix, (2,), plan=plan)


def test_statevector_shard_exchange_applies_cross_shard_gate_without_dense_state():
    circuit = fq.Circuit(3)
    circuit.x(2)
    plan = fq.plan_distributed_statevector(circuit, world_size=2)
    x_matrix = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex64)
    shards = tuple(
        fq.initialize_statevector_shard(plan, rank=rank, device="cpu")
        for rank in range(plan.world_size)
    )

    updated = fq.apply_gate_to_statevector_shards(shards, x_matrix, (2,), plan=plan)
    reconstructed = torch.zeros((1, plan.total_amplitudes), dtype=torch.complex64)
    for shard in updated:
        reconstructed[:, shard.global_indices] = shard.amplitudes

    assert tuple(shard.summary()["local_amplitudes"] for shard in updated) == (4, 4)
    assert torch.allclose(reconstructed, circuit.state(), atol=1e-6)


def test_statevector_shard_state_applies_diagonal_gate_on_sharded_wire():
    circuit = fq.Circuit(3)
    plan = fq.plan_distributed_statevector(circuit, world_size=2)
    rank0 = fq.initialize_statevector_shard(plan, rank=0, device="cpu")
    rank1 = fq.initialize_statevector_shard(plan, rank=1, device="cpu")

    phase = torch.tensor(0.25, dtype=torch.float32)
    rz = torch.diag(
        torch.stack(
            [
                torch.exp(-0.5j * phase),
                torch.exp(0.5j * phase),
            ]
        ).to(torch.complex64)
    )
    updated0 = fq.apply_gate_to_statevector_shard(rank0, rz, (2,), plan=plan)
    updated1 = fq.apply_gate_to_statevector_shard(rank1, rz, (2,), plan=plan)
    reconstructed = torch.zeros((1, plan.total_amplitudes), dtype=torch.complex64)
    reconstructed[:, updated0.global_indices] = updated0.amplitudes
    reconstructed[:, updated1.global_indices] = updated1.amplitudes

    expected = fq.Circuit(3)
    expected.rz(2, theta=phase)

    assert torch.allclose(reconstructed, expected.state(), atol=1e-6)


def test_local_distributed_statevector_simulator_matches_single_device_state():
    theta = torch.tensor(0.2)
    circuit = fq.Circuit(3)
    circuit.h(0).rx(1, theta=theta).x(2).cx(0, 2).rz(1, theta=-0.3)

    result = fq.simulate_distributed_statevector_local(
        circuit, world_size=2, device="cpu"
    )

    assert isinstance(result, fq.LocalDistributedStatevectorResult)
    assert result.summary()["executor"] == "local_cpu_distributed_simulator"
    assert result.summary()["distribution_semantics"] == "sharded_across_ranks"
    assert result.summary()["scalability_claim_allowed"] is False
    assert result.local_gate_count > 0
    assert result.distributed_gate_count > 0
    assert result.simulated_communication_count > 0
    assert (
        result.summary()["communication_execution"] == "rank_local_amplitude_exchange"
    )
    assert result.summary()["full_state_reconstruction_count"] == 1
    assert torch.allclose(result.state, circuit.state(), atol=1e-6)


def test_distributed_statevector_transport_requires_initialized_group_for_multi_rank():
    circuit = fq.Circuit(4)
    circuit.x(3).cx(0, 3)
    plan = fq.plan_distributed_statevector(circuit, world_size=2)

    report = fq.execute_distributed_statevector_transport(plan)

    assert isinstance(report, fq.StatevectorTransportReport)
    assert report.rank == 0
    assert report.world_size == 1
    assert not report.valid
    assert "torch.distributed is not initialized" in report.errors


def test_distributed_statevector_correctness_run_spec():
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 2)
    plan = fq.plan_distributed_statevector(circuit, world_size=2)

    spec = plan.correctness_run_spec(entrypoint="checks/statevector.py", backend="gloo")
    spec_from_function = fq.build_statevector_correctness_run_spec(
        plan,
        entrypoint="checks/statevector.py",
        backend="gloo",
    )

    assert isinstance(spec, fq.StatevectorCorrectnessRunSpec)
    assert spec == spec_from_function
    assert spec.command() == (
        "torchrun",
        "--nproc_per_node=2",
        "checks/statevector.py",
        "--world-size",
        "2",
        "--n-wires",
        "3",
        "--distribution",
        "qubit_address_sharded",
        "--topology",
        "hypercube",
    )
    assert ("FQ_DIST_BACKEND", "gloo") in spec.env
    assert "local_vs_distributed_expectation" in spec.expected_checks


def test_statevector_correctness_script_single_rank_smoke():
    script = (
        Path(__file__).resolve().parent / "distributed" / "statevector_correctness.py"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--world-size",
            "1",
            "--n-wires",
            "3",
            "--distribution",
            "replicated_single_rank",
            "--topology",
            "single_rank",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "statevector_correctness" in completed.stdout
    assert "world_size=1" in completed.stdout
    assert "transport_events=0" in completed.stdout


def test_distributed_statevector_plan_supports_non_power_of_two_world_size():
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 1)

    plan = fq.plan_distributed_statevector(circuit, world_size=3)

    assert plan.distribution == "contiguous_amplitude_range"
    assert plan.topology.layout == "range_partition"
    assert plan.sharded_wires == ()
    assert tuple(shard.local_amplitudes for shard in plan.shards) == (3, 3, 2)
    assert all(gate.communication == "indexed_all_to_all" for gate in plan.gate_plans)
    assert plan.topology.summary()["communications"] == ("indexed_all_to_all",)


def test_run_distributed_attaches_statevector_plan_summary():
    circuit = fq.Circuit(2)
    circuit.h(0).cx(0, 1)

    result = run_advanced(
        circuit, mode="distributed_statevector", device="cpu", world_size=1
    )

    assert isinstance(result, fq.ExecutionResult)
    assert result.plan.state_mode == "statevector"
    assert result.runtime["mode"] == "distributed_statevector"
    assert result.compatibility["source_type"] == "DistributedQuantumDevice"
    assert result.compatibility["full_state_materialized_by_adapter"] is False

    # Backend-native inspection remains available explicitly without weakening
    # the stable ExecutionResult contract of Circuit.run().
    qdev = fqb.run_native(
        circuit,
        mode="distributed_statevector",
        device="cpu",
        world_size=1,
    )

    assert qdev.distributed_statevector_plan.world_size == 1
    assert qdev.distributed_statevector_plan.validate().valid
    assert (
        qdev.distributed_statevector_summary["state_mode"] == "distributed_statevector"
    )
    assert (
        qdev.distributed_statevector_summary["distribution"] == "replicated_single_rank"
    )
    assert torch.allclose(fq.measure_allZ(qdev), circuit.expectation_z(), atol=1e-6)
