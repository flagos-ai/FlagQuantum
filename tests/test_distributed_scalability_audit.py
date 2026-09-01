import json
from pathlib import Path

import pytest

import flagquantum as fq
import flagquantum.compilation.planner as fqxp
from benchmarks.audit_results import _json_files, audit_paths
from flagquantum.runtime.audit.engine import evaluate_mps_backward_readiness

pytestmark = [pytest.mark.distributed, pytest.mark.distributed_cpu]


def _optimizer_ownership():
    return (
        {
            "rank": 0,
            "parameter_start": 0,
            "parameter_end": 2,
            "owned_parameter_count": 2,
            "writeback_route": "rank_local_parameter_owner_writeback",
        },
        {
            "rank": 1,
            "parameter_start": 2,
            "parameter_end": 4,
            "owned_parameter_count": 2,
            "writeback_route": "rank_local_parameter_owner_writeback",
        },
    )


def _with_sharded_optimizer_evidence(payload, *, training_step_count=3):
    ownership = _optimizer_ownership()
    return fq.attach_sharded_optimizer_step_evidence(
        payload,
        training_step_count=training_step_count,
        parameter_ownership=ownership,
        gradient_ownership=ownership,
        optimizer_update_ownership=ownership,
    )


@pytest.mark.parametrize("bad_training_step_count", (0, -1, "not-an-int"))
def test_sharded_optimizer_step_evidence_helper_rejects_non_positive_steps(
    bad_training_step_count,
):
    with pytest.raises(
        ValueError, match="training_step_count must be a positive integer"
    ):
        fq.attach_sharded_optimizer_step_evidence(
            {},
            training_step_count=bad_training_step_count,
            parameter_ownership=_optimizer_ownership(),
            gradient_ownership=_optimizer_ownership(),
            optimizer_update_ownership=_optimizer_ownership(),
        )


@pytest.mark.parametrize(
    ("missing_key", "expected_message"),
    (
        ("parameter_ownership", "parameter_ownership must be nonempty"),
        ("gradient_ownership", "gradient_ownership must be nonempty"),
        ("optimizer_update_ownership", "optimizer_update_ownership must be nonempty"),
    ),
)
def test_sharded_optimizer_step_evidence_helper_rejects_missing_ownership(
    missing_key,
    expected_message,
):
    kwargs = {
        "training_step_count": 1,
        "parameter_ownership": _optimizer_ownership(),
        "gradient_ownership": _optimizer_ownership(),
        "optimizer_update_ownership": _optimizer_ownership(),
    }
    kwargs[missing_key] = ()

    with pytest.raises(ValueError, match=expected_message):
        fq.attach_sharded_optimizer_step_evidence({}, **kwargs)


def _write_repo_local_audit_payload(name, payload):
    path = Path(".pytest_cache") / "flagquantum_audit_payloads" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_scalability_audit_accepts_sharded_statevector_plan():
    circuit = fq.Circuit(5)
    circuit.x(3).x(4).cx(0, 4)
    plan = fq.plan_distributed_statevector(circuit, world_size=4, local_world_size=2)

    audit = fq.audit_distributed_scalability(plan.summary())

    assert isinstance(audit, fq.DistributedScalabilityAudit)
    assert audit.valid
    assert not audit.scalability_claim_allowed
    assert audit.claim_evidence_type == "plan_preflight"
    assert not audit.release_gate_allowed
    assert audit.errors == ()


def test_scalability_audit_rejects_replicated_kernel_claim():
    payload = {
        "distribution_semantics": "rank_local_replicated_kernel",
        "scalability_claim_allowed": True,
        "world_size": 8,
    }

    audit = fq.audit_distributed_scalability(payload)

    assert not audit.valid
    assert not audit.scalability_claim_allowed
    assert (
        "scalability_claim_allowed requires distribution_semantics='sharded_across_ranks'"
        in audit.errors
    )
    assert (
        "replicated execution cannot claim single-workload scalability" in audit.errors
    )


def test_scalability_audit_accepts_single_device_fast_path_without_distributed_metadata():
    payload = {
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "world_size": 1,
        "backend": "jax",
        "mode": "mps",
    }

    audit = fq.audit_distributed_scalability(payload)

    assert audit.valid
    assert not audit.scalability_claim_allowed
    assert audit.errors == ()


def test_runtime_planner_local_fast_path_does_not_emit_distributed_claim():
    circuit = fq.Circuit(3)
    circuit.h(0).cx(0, 1).ry(2, theta=0.2)

    summary = fqxp.plan_runtime_selection(
        circuit, prefer_jax=True, require_gradients=True
    ).summary()
    candidate = summary["recommended_candidate"]
    audit = fq.audit_distributed_scalability(candidate)

    assert summary["world_size"] == 1
    assert candidate["distribution_semantics"] == "single_device_fast_path"
    assert candidate["scalability_claim_allowed"] is False
    assert candidate["release_gate_allowed"] is False
    assert candidate["available"] is True
    assert audit.valid
    assert not audit.scalability_claim_allowed
    with pytest.raises(fq.DistributedScalabilityError):
        fq.require_distributed_scalability(candidate)


def test_attach_scalability_audit_adds_machine_readable_summary():
    payload = {
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "world_size": 1,
    }

    attached = fq.attach_distributed_scalability_audit(payload)

    assert attached is not payload
    assert attached["scalability_audit"]["valid"] is True
    assert (
        attached["scalability_audit"]["distribution_semantics"]
        == "single_device_fast_path"
    )
    assert attached["scalability_audit"]["scalability_claim_allowed"] is False
    assert attached["scalability_audit"]["release_gate_allowed"] is False


def _statevector_claimable_payload(**overrides):
    payload = {
        "claim_evidence_type": "production_training_benchmark",
        "state_mode": "jax_sharded_statevector",
        "distribution_semantics": "sharded_across_ranks",
        "forward_distribution_semantics": "sharded_across_ranks",
        "backward_distribution_semantics": "sharded_across_ranks",
        "scalability_claim_allowed": True,
        "parameter_gradient_ready": True,
        "gradient_distribution_semantics": "sharded_across_ranks",
        "world_size": 2,
        "local_world_size": 2,
        "node_count": 1,
        "rank_shards": (
            {
                "rank": 0,
                "amplitude_start": 0,
                "amplitude_end": 4,
                "local_state_bytes": 64,
            },
            {
                "rank": 1,
                "amplitude_start": 4,
                "amplitude_end": 8,
                "local_state_bytes": 64,
            },
        ),
        "memory_plan": {
            "per_rank_shard_bytes": (64, 64),
            "communication_buffer_bytes": 128,
            "communication_buffer_count": 1,
        },
        "communication_plan": {
            "communication_execution": "jax_pmap_amplitude_exchange_backward_replay",
            "transport_patterns": ("pair_exchange", "all_to_all"),
        },
        "backward_uses_full_state_replay": False,
        "scalability_blockers": (),
        "gradient_blockers": (),
    }
    payload = _with_sharded_optimizer_evidence(payload)
    payload.update(overrides)
    return payload


def test_statevector_training_claimability_accepts_complete_training_payload():
    gate = fq.evaluate_statevector_training_claimability(
        _statevector_claimable_payload()
    )

    assert gate.status == "claimable_production_training"
    assert gate.claimable_production_training
    assert gate.errors == ()
    assert gate.blockers == ()
    assert all(gate.checks.values())


def test_sharded_optimizer_step_evidence_helper_satisfies_statevector_ownership_gate():
    payload = {
        "claim_evidence_type": "production_runtime",
        "state_mode": "jax_sharded_statevector",
        "distribution_semantics": "sharded_across_ranks",
        "forward_distribution_semantics": "sharded_across_ranks",
        "backward_distribution_semantics": "sharded_across_ranks",
        "scalability_claim_allowed": False,
        "parameter_gradient_ready": True,
        "gradient_distribution_semantics": "sharded_across_ranks",
        "world_size": 2,
        "local_world_size": 2,
        "node_count": 1,
        "rank_shards": (
            {
                "rank": 0,
                "amplitude_start": 0,
                "amplitude_end": 4,
                "local_state_bytes": 64,
            },
            {
                "rank": 1,
                "amplitude_start": 4,
                "amplitude_end": 8,
                "local_state_bytes": 64,
            },
        ),
        "memory_plan": {
            "per_rank_shard_bytes": (64, 64),
            "communication_buffer_bytes": 128,
            "communication_buffer_count": 1,
        },
        "communication_plan": {
            "communication_execution": "jax_pmap_amplitude_exchange_backward_replay",
            "transport_patterns": ("pair_exchange",),
        },
        "backward_uses_full_state_replay": False,
        "scalability_blockers": (),
        "gradient_blockers": (),
    }

    payload = _with_sharded_optimizer_evidence(payload, training_step_count=2)
    gate = fq.evaluate_statevector_training_claimability(payload)

    assert payload["parameter_ownership_semantics"] == "sharded_across_ranks"
    assert payload["gradient_ownership_semantics"] == "sharded_across_ranks"
    assert payload["optimizer_update_ownership_semantics"] == "sharded_across_ranks"
    assert payload["optimizer_update_semantics"] == "sharded_across_ranks"
    assert payload["training_step_count"] == 2
    assert gate.status == "claimable_production_training"
    assert gate.checks["parameter_ownership_sharded"] is True
    assert gate.checks["gradient_ownership_sharded"] is True
    assert gate.checks["optimizer_update_ownership_sharded"] is True
    assert gate.checks["training_steps_executed"] is True


def test_statevector_training_claimability_rejects_non_statevector_payload():
    gate = fq.evaluate_statevector_training_claimability(
        _statevector_claimable_payload(state_mode="distributed_mps")
    )

    assert gate.status == "blocked"
    assert not gate.claimable_production_training
    assert gate.checks["statevector_path"] is False
    assert (
        "statevector training gate requires state_mode or mode to be one of "
        "distributed_statevector, jax_sharded_statevector, statevector"
    ) in gate.errors


def test_statevector_training_claimability_rejects_unknown_state_mode_even_if_mode_is_statevector():
    gate = fq.evaluate_statevector_training_claimability(
        _statevector_claimable_payload(state_mode="tensor_network", mode="statevector")
    )

    assert gate.status == "blocked"
    assert not gate.claimable_production_training
    assert gate.checks["statevector_path"] is False


def test_statevector_training_claimability_fails_closed_for_missing_fields():
    gate = fq.evaluate_statevector_training_claimability(
        {
            "state_mode": "jax_sharded_statevector",
            "distribution_semantics": "sharded_across_ranks",
        }
    )

    assert gate.status == "blocked"
    assert not gate.claimable_production_training
    assert (
        "statevector training gate requires parameter_gradient_ready=True"
        in gate.errors
    )
    assert (
        "statevector training gate requires optimizer_update_semantics='sharded_across_ranks'"
        in gate.errors
    )
    assert (
        "statevector training gate requires memory_plan with per-rank shard and communication buffer"
        in gate.errors
    )


def test_statevector_training_claimability_rejects_forward_only_and_full_state_replay():
    gate = fq.evaluate_statevector_training_claimability(
        _statevector_claimable_payload(
            backward_distribution_semantics="incomplete",
            parameter_gradient_ready=False,
            backward_uses_full_state_replay=True,
        )
    )

    assert gate.status == "blocked"
    assert (
        "statevector training gate requires backward_distribution_semantics='sharded_across_ranks'"
        in gate.errors
    )
    assert (
        "statevector training gate requires parameter_gradient_ready=True"
        in gate.errors
    )
    assert (
        "statevector training gate rejects full-state replay for backward or parameter gradients"
        in gate.errors
    )


def test_statevector_training_claimability_rejects_nonempty_blockers():
    gate = fq.evaluate_statevector_training_claimability(
        _statevector_claimable_payload(
            scalability_blockers=("production_collective_timeout_not_validated",),
        )
    )

    assert gate.status == "blocked"
    assert not gate.claimable_production_training
    assert gate.blockers == ("production_collective_timeout_not_validated",)
    assert gate.checks["blockers_empty"] is False
    assert "statevector training gate requires blockers to be empty" in gate.errors


def test_statevector_training_claimability_rejects_missing_rank_ownership_and_topology():
    payload = _statevector_claimable_payload()
    payload.pop("rank_shards")
    payload.pop("world_size")
    payload.pop("local_world_size")
    payload.pop("node_count")

    gate = fq.evaluate_statevector_training_claimability(payload)

    assert gate.status == "blocked"
    assert gate.checks["rank_ownership_explicit"] is False
    assert gate.checks["world_topology_reported"] is False
    assert "statevector training gate requires explicit rank ownership" in gate.errors
    assert (
        "statevector training gate requires world_size, local_world_size, and node_count"
        in gate.errors
    )


def test_statevector_training_claimability_rejects_incomplete_memory_and_communication_plans():
    gate = fq.evaluate_statevector_training_claimability(
        _statevector_claimable_payload(
            memory_plan={"per_rank_shard_bytes": (64, 64)},
            communication_plan={"communication_execution": "rank_local_no_transport"},
            communication_tiers={},
        )
    )

    assert gate.status == "blocked"
    assert gate.checks["memory_plan_has_per_rank_shard_and_comm_buffer"] is False
    assert gate.checks["communication_plan_has_statevector_route"] is False
    assert (
        "statevector training gate requires memory_plan with per-rank shard and communication buffer"
        in gate.errors
    )
    assert (
        "statevector training gate requires communication_plan with pair-exchange, all-to-all, or collective route"
        in gate.errors
    )


def test_statevector_training_claimability_rejects_non_sharded_optimizer_update():
    gate = fq.evaluate_statevector_training_claimability(
        _statevector_claimable_payload(
            optimizer_update_semantics="replicated_all_reduce"
        )
    )

    assert gate.status == "blocked"
    assert gate.checks["optimizer_step_preserves_sharded_ownership"] is False
    assert (
        "statevector training gate requires optimizer_update_semantics='sharded_across_ranks'"
        in gate.errors
    )


def test_statevector_training_claimability_rejects_missing_optimizer_ownership_evidence():
    payload = _statevector_claimable_payload()
    payload.pop("parameter_ownership")
    payload.pop("gradient_ownership")
    payload.pop("optimizer_update_ownership")
    payload.pop("optimizer_step_evidence")

    gate = fq.evaluate_statevector_training_claimability(payload)

    assert gate.status == "blocked"
    assert gate.checks["parameter_ownership_sharded"] is False
    assert gate.checks["gradient_ownership_sharded"] is False
    assert gate.checks["optimizer_update_ownership_sharded"] is False
    assert (
        "statevector training gate requires parameter_ownership_semantics='sharded_across_ranks' with explicit parameter ownership"
        in gate.errors
    )
    assert (
        "statevector training gate requires gradient_ownership_semantics='sharded_across_ranks' with explicit gradient ownership"
        in gate.errors
    )
    assert (
        "statevector training gate requires optimizer_update_ownership_semantics='sharded_across_ranks' with explicit optimizer-update ownership"
        in gate.errors
    )


@pytest.mark.parametrize(
    ("semantics_key", "expected_error"),
    (
        (
            "parameter_ownership_semantics",
            "statevector training gate requires parameter_ownership_semantics='sharded_across_ranks' with explicit parameter ownership",
        ),
        (
            "gradient_ownership_semantics",
            "statevector training gate requires gradient_ownership_semantics='sharded_across_ranks' with explicit gradient ownership",
        ),
        (
            "optimizer_update_ownership_semantics",
            "statevector training gate requires optimizer_update_ownership_semantics='sharded_across_ranks' with explicit optimizer-update ownership",
        ),
    ),
)
@pytest.mark.parametrize(
    "bad_semantics", ("unknown", "replicated_per_rank", "rank_local_replicated_kernel")
)
def test_statevector_training_claimability_rejects_non_sharded_optimizer_ownership_semantics(
    semantics_key,
    expected_error,
    bad_semantics,
):
    payload = _statevector_claimable_payload(**{semantics_key: bad_semantics})

    gate = fq.evaluate_statevector_training_claimability(payload)

    assert gate.status == "blocked"
    assert not gate.claimable_production_training
    assert expected_error in gate.errors


@pytest.mark.parametrize(
    ("name", "mutate", "expected_error"),
    (
        (
            "missing_backward_gradient",
            lambda payload: payload.update(
                {
                    "backward_distribution_semantics": "incomplete",
                    "parameter_gradient_ready": False,
                }
            ),
            "statevector training gate requires backward_distribution_semantics='sharded_across_ranks'",
        ),
        (
            "missing_rank_ownership",
            lambda payload: payload.pop("rank_shards"),
            "statevector training gate requires explicit rank ownership",
        ),
        (
            "missing_memory_plan",
            lambda payload: payload.update(
                {"memory_plan": {"per_rank_shard_bytes": (64, 64)}}
            ),
            "statevector training gate requires memory_plan with per-rank shard and communication buffer",
        ),
        (
            "missing_communication_plan",
            lambda payload: payload.update(
                {
                    "communication_plan": {
                        "communication_execution": "rank_local_no_transport"
                    }
                }
            ),
            "statevector training gate requires communication_plan with pair-exchange, all-to-all, or collective route",
        ),
        (
            "nonempty_blockers",
            lambda payload: payload.update(
                {"scalability_blockers": ("production_collective_pending",)}
            ),
            "statevector training gate requires blockers to be empty",
        ),
    ),
)
def test_statevector_training_claimability_false_positive_matrix_fails_closed(
    name,
    mutate,
    expected_error,
):
    payload = _statevector_claimable_payload()
    mutate(payload)

    gate = fq.evaluate_statevector_training_claimability(payload)

    assert name
    assert gate.status == "blocked"
    assert not gate.claimable_production_training
    assert expected_error in gate.errors


def test_statevector_training_claimability_reports_preflight_only():
    payload = _statevector_claimable_payload(
        claim_evidence_type="plan_preflight",
        planner="jax_sharded_statevector_training",
        scalability_claim_allowed=False,
    )
    gate = fq.evaluate_statevector_training_claimability(payload)

    assert gate.status == "preflight_only"
    assert not gate.claimable_production_training


def test_statevector_training_claimability_reports_local_simulation():
    payload = _statevector_claimable_payload(
        claim_evidence_type="development_smoke",
        backward_execution="local_simulated_backward",
    )
    gate = fq.evaluate_statevector_training_claimability(payload)

    assert gate.status == "local_simulation"
    assert not gate.claimable_production_training


def test_require_scalability_accepts_only_release_grade_training_benchmark():
    payload = {
        "claim_evidence_type": "production_training_benchmark",
        "distribution_semantics": "sharded_across_ranks",
        "scalability_claim_allowed": True,
        "world_size": 2,
        "local_world_size": 2,
        "node_count": 1,
        "rank_shards": (
            {
                "rank": 0,
                "amplitude_start": 0,
                "amplitude_end": 4,
                "local_state_bytes": 64,
            },
            {
                "rank": 1,
                "amplitude_start": 4,
                "amplitude_end": 8,
                "local_state_bytes": 64,
            },
        ),
        "estimated_transfer_bytes": 128,
        "single_gpu_expected_oom": True,
        "capacity_baseline_device": "single_gpu_24gb",
        "capacity_failure_reason": "single_gpu_memory_budget_exceeded",
        "gradient_distribution_semantics": "sharded_across_ranks",
    }
    payload = _with_sharded_optimizer_evidence(payload)

    audit = fq.require_distributed_scalability(payload)

    assert audit.valid
    assert audit.scalability_claim_allowed
    assert audit.release_gate_allowed


def test_require_scalability_accepts_explicit_release_payload():
    payload = {
        "claim_evidence_type": "release_payload",
        "release_payload": True,
        "distribution_semantics": "sharded_across_ranks",
        "scalability_claim_allowed": True,
        "world_size": 2,
        "local_world_size": 2,
        "node_count": 1,
        "rank_placement": ({"rank": 0, "node_id": 0}, {"rank": 1, "node_id": 0}),
        "local_memory_bytes_by_rank": (64, 64),
        "communication_bytes": 128,
        "single_gpu_expected_oom": True,
        "capacity_baseline_device": "single_gpu_24gb",
        "capacity_failure_reason": "single_gpu_memory_budget_exceeded",
        "gradient_distribution_semantics": "sharded_across_ranks",
    }
    payload = _with_sharded_optimizer_evidence(payload)

    audit = fq.require_distributed_scalability(payload)

    assert audit.valid
    assert audit.claim_evidence_type == "release_payload"
    assert audit.release_gate_allowed


def test_require_scalability_rejects_release_payload_without_explicit_marker():
    payload = {
        "claim_evidence_type": "release_payload",
        "distribution_semantics": "sharded_across_ranks",
        "scalability_claim_allowed": True,
        "world_size": 2,
        "local_world_size": 2,
        "node_count": 1,
        "rank_placement": ({"rank": 0, "node_id": 0}, {"rank": 1, "node_id": 0}),
        "local_memory_bytes_by_rank": (64, 64),
        "communication_bytes": 128,
        "single_gpu_expected_oom": True,
        "capacity_baseline_device": "single_gpu_24gb",
        "capacity_failure_reason": "single_gpu_memory_budget_exceeded",
        "gradient_distribution_semantics": "sharded_across_ranks",
    }
    payload = _with_sharded_optimizer_evidence(payload)

    with pytest.raises(fq.DistributedScalabilityError) as excinfo:
        fq.require_distributed_scalability(payload)

    assert excinfo.value.audit.claim_evidence_type == "release_payload"
    assert "release_payload evidence must be explicit" in excinfo.value.audit.errors


def test_require_scalability_rejects_production_runtime_even_with_training_fields():
    payload = {
        "claim_evidence_type": "production_runtime",
        "distribution_semantics": "sharded_across_ranks",
        "scalability_claim_allowed": True,
        "world_size": 2,
        "local_world_size": 2,
        "node_count": 1,
        "rank_shards": (
            {
                "rank": 0,
                "amplitude_start": 0,
                "amplitude_end": 4,
                "local_state_bytes": 64,
            },
            {
                "rank": 1,
                "amplitude_start": 4,
                "amplitude_end": 8,
                "local_state_bytes": 64,
            },
        ),
        "communication_bytes": 128,
        "single_gpu_expected_oom": True,
        "capacity_baseline_device": "single_gpu_24gb",
        "capacity_failure_reason": "single_gpu_memory_budget_exceeded",
        "gradient_distribution_semantics": "sharded_across_ranks",
    }
    payload = _with_sharded_optimizer_evidence(payload)

    with pytest.raises(fq.DistributedScalabilityError) as excinfo:
        fq.require_distributed_scalability(payload)

    assert excinfo.value.audit.claim_evidence_type == "production_runtime"
    assert (
        "release gate requires claim_evidence_type='production_training_benchmark' or 'release_payload'"
        in excinfo.value.audit.errors
    )


def test_require_scalability_rejects_plan_preflight_payload():
    circuit = fq.Circuit(5)
    circuit.x(3).x(4).cx(0, 4)
    plan = fq.plan_distributed_statevector(circuit, world_size=4, local_world_size=2)

    with pytest.raises(fq.DistributedScalabilityError) as excinfo:
        fq.require_distributed_scalability(plan.summary())

    assert excinfo.value.audit.claim_evidence_type == "plan_preflight"
    assert (
        "release gate requires scalability_claim_allowed=True"
        in excinfo.value.audit.errors
    )


def test_require_scalability_rejects_missing_capacity_evidence():
    payload = {
        "claim_evidence_type": "production_training_benchmark",
        "distribution_semantics": "sharded_across_ranks",
        "scalability_claim_allowed": True,
        "world_size": 2,
        "local_world_size": 2,
        "node_count": 1,
        "rank_shards": (
            {
                "rank": 0,
                "amplitude_start": 0,
                "amplitude_end": 4,
                "local_state_bytes": 64,
            },
            {
                "rank": 1,
                "amplitude_start": 4,
                "amplitude_end": 8,
                "local_state_bytes": 64,
            },
        ),
        "communication_bytes": 128,
        "gradient_distribution_semantics": "sharded_across_ranks",
    }
    payload = _with_sharded_optimizer_evidence(payload)

    with pytest.raises(fq.DistributedScalabilityError) as excinfo:
        fq.require_distributed_scalability(payload)

    assert (
        "release gate requires single_gpu_expected_oom=True"
        in excinfo.value.audit.errors
    )
    assert (
        "release gate requires capacity_baseline_device" in excinfo.value.audit.errors
    )
    assert "release gate requires capacity_failure_reason" in excinfo.value.audit.errors


def test_require_scalability_rejects_false_capacity_expansion():
    payload = {
        "claim_evidence_type": "production_training_benchmark",
        "distribution_semantics": "sharded_across_ranks",
        "scalability_claim_allowed": True,
        "world_size": 2,
        "local_world_size": 2,
        "node_count": 1,
        "rank_shards": (
            {
                "rank": 0,
                "amplitude_start": 0,
                "amplitude_end": 4,
                "local_state_bytes": 64,
            },
            {
                "rank": 1,
                "amplitude_start": 4,
                "amplitude_end": 8,
                "local_state_bytes": 64,
            },
        ),
        "communication_bytes": 128,
        "single_gpu_expected_oom": False,
        "capacity_baseline_device": "single_gpu_24gb",
        "capacity_failure_reason": "fits_on_single_gpu",
        "gradient_distribution_semantics": "sharded_across_ranks",
    }
    payload = _with_sharded_optimizer_evidence(payload)

    with pytest.raises(fq.DistributedScalabilityError) as excinfo:
        fq.require_distributed_scalability(payload)

    assert (
        "release gate requires single_gpu_expected_oom=True"
        in excinfo.value.audit.errors
    )


def test_base_audit_release_gate_rejects_false_capacity_expansion():
    payload = {
        "claim_evidence_type": "production_training_benchmark",
        "distribution_semantics": "sharded_across_ranks",
        "scalability_claim_allowed": True,
        "world_size": 2,
        "local_world_size": 2,
        "node_count": 1,
        "rank_shards": (
            {
                "rank": 0,
                "amplitude_start": 0,
                "amplitude_end": 4,
                "local_state_bytes": 64,
            },
            {
                "rank": 1,
                "amplitude_start": 4,
                "amplitude_end": 8,
                "local_state_bytes": 64,
            },
        ),
        "communication_bytes": 128,
        "single_gpu_expected_oom": False,
        "capacity_baseline_device": "single_gpu_24gb",
        "capacity_failure_reason": "fits_on_single_gpu",
        "gradient_distribution_semantics": "sharded_across_ranks",
    }
    payload = _with_sharded_optimizer_evidence(payload)

    audit = fq.audit_distributed_scalability(payload)

    assert audit.valid
    assert audit.scalability_claim_allowed
    assert not audit.release_gate_allowed
    assert any(
        "release gate evidence is incomplete" in warning for warning in audit.warnings
    )


def test_distributed_transport_evidence_classifies_single_node_release_collective():
    payload = _statevector_claimable_payload(
        claim_evidence_type="release_payload",
        release_payload=True,
        single_gpu_expected_oom=True,
        capacity_baseline_device="single_gpu_24gb",
        capacity_failure_reason="single_gpu_memory_budget_exceeded",
        communication_bytes=128,
        communication_plan={
            "transport": "single_node_xla_all_to_all",
            "topology_scope": "single_node_executed_collective",
            "estimated_transfer_bytes": 128,
            "intra_node_communication_bytes": 128,
            "inter_node_communication_bytes": 0,
        },
    )

    transport = fq.evaluate_distributed_transport_evidence(payload)
    attached = fq.attach_distributed_evidence_contract(payload)

    assert transport.status == "single_node_executed_collective"
    assert transport.node_count == 1
    assert transport.errors == ()
    assert attached["transport_evidence_status"] == "single_node_executed_collective"
    assert (
        attached["distributed_evidence_contract"]["transport_evidence"]["status"]
        == "single_node_executed_collective"
    )


def test_distributed_transport_evidence_accepts_multi_node_production_route():
    payload = _statevector_claimable_payload(
        claim_evidence_type="production_training_benchmark",
        world_size=4,
        local_world_size=2,
        node_count=2,
        rank_placement=(
            {"rank": 0, "node_id": 0, "local_rank": 0},
            {"rank": 1, "node_id": 0, "local_rank": 1},
            {"rank": 2, "node_id": 1, "local_rank": 0},
            {"rank": 3, "node_id": 1, "local_rank": 1},
        ),
        communication_bytes=256,
        intra_node_communication_bytes=128,
        inter_node_communication_bytes=128,
        network_backend="nccl",
        communication_plan={
            "communication_execution": "jax_pmap_amplitude_exchange_backward_replay",
            "transport_patterns": ("pair_exchange", "all_to_all"),
            "topology_scope": "multi_node_production_transport",
            "executed_collective_route": {
                "backend": "nccl",
                "route_scope": "multi_node",
                "collective": "all_to_all",
                "nodes": (0, 1),
            },
            "estimated_transfer_bytes": 256,
            "intra_node_communication_bytes": 128,
            "inter_node_communication_bytes": 128,
        },
        single_gpu_expected_oom=True,
        capacity_baseline_device="single_gpu_24gb",
        capacity_failure_reason="single_gpu_memory_budget_exceeded",
    )

    transport = fq.evaluate_distributed_transport_evidence(payload)
    audit = fq.require_distributed_scalability(payload)
    contract = fq.evaluate_distributed_evidence_contract(payload)

    assert transport.status == "multi_node_production_transport"
    assert transport.network_backend == "nccl"
    assert transport.rank_placement_reported is True
    assert transport.communication_bytes_reported is True
    assert audit.release_gate_allowed
    assert contract.checks["multi_node_transport_evidence_complete"] is True
    assert contract.transport_evidence["status"] == "multi_node_production_transport"


def test_multi_node_release_claim_rejects_route_scope_without_backend_evidence():
    payload = _statevector_claimable_payload(
        claim_evidence_type="production_training_benchmark",
        world_size=4,
        local_world_size=2,
        node_count=2,
        rank_placement=(
            {"rank": 0, "node_id": 0, "local_rank": 0},
            {"rank": 1, "node_id": 0, "local_rank": 1},
            {"rank": 2, "node_id": 1, "local_rank": 0},
            {"rank": 3, "node_id": 1, "local_rank": 1},
        ),
        communication_bytes=256,
        intra_node_communication_bytes=128,
        inter_node_communication_bytes=128,
        communication_plan={
            "communication_execution": "jax_pmap_amplitude_exchange_backward_replay",
            "transport_patterns": ("pair_exchange", "all_to_all"),
            "executed_collective_route": {
                "route_scope": "multi_node",
                "collective": "all_to_all",
                "nodes": (0, 1),
            },
            "estimated_transfer_bytes": 256,
            "intra_node_communication_bytes": 128,
            "inter_node_communication_bytes": 128,
        },
        single_gpu_expected_oom=True,
        capacity_baseline_device="single_gpu_24gb",
        capacity_failure_reason="single_gpu_memory_budget_exceeded",
    )

    transport = fq.evaluate_distributed_transport_evidence(payload)
    contract = fq.evaluate_distributed_evidence_contract(payload)

    assert transport.status == "topology_dependent_planning"
    assert transport.network_backend == "unknown"
    assert "multi_node_production_transport_backend_required" in transport.blockers
    assert contract.status == "blocked"
    assert contract.checks["multi_node_transport_evidence_complete"] is False
    with pytest.raises(fq.DistributedScalabilityError) as excinfo:
        fq.require_distributed_scalability(payload)
    assert (
        "multi_node_production_transport_backend_required" in excinfo.value.audit.errors
    )


def test_multi_node_release_claim_accepts_payload_level_production_backend_evidence():
    payload = _statevector_claimable_payload(
        claim_evidence_type="production_training_benchmark",
        world_size=4,
        local_world_size=2,
        node_count=2,
        rank_placement=(
            {"rank": 0, "node_id": 0, "local_rank": 0},
            {"rank": 1, "node_id": 0, "local_rank": 1},
            {"rank": 2, "node_id": 1, "local_rank": 0},
            {"rank": 3, "node_id": 1, "local_rank": 1},
        ),
        communication_bytes=256,
        intra_node_communication_bytes=128,
        inter_node_communication_bytes=128,
        network_backend="gloo",
        communication_plan={
            "communication_execution": "torch_distributed_all_to_all",
            "transport_patterns": ("all_to_all",),
            "executed_collective_route": {
                "route_scope": "multi_node",
                "collective": "all_to_all",
                "nodes": (0, 1),
            },
            "estimated_transfer_bytes": 256,
            "intra_node_communication_bytes": 128,
            "inter_node_communication_bytes": 128,
        },
        single_gpu_expected_oom=True,
        capacity_baseline_device="single_gpu_24gb",
        capacity_failure_reason="single_gpu_memory_budget_exceeded",
    )

    transport = fq.evaluate_distributed_transport_evidence(payload)
    contract = fq.evaluate_distributed_evidence_contract(payload)
    audit = fq.require_distributed_scalability(payload)

    assert transport.status == "multi_node_production_transport"
    assert transport.network_backend == "gloo"
    assert contract.checks["multi_node_transport_evidence_complete"] is True
    assert contract.transport_evidence["status"] == "multi_node_production_transport"
    assert audit.release_gate_allowed


@pytest.mark.parametrize(
    ("missing_fields", "expected_blocker"),
    (
        (
            ("rank_placement",),
            "multi_node_rank_placement_required_for_transport_evidence",
        ),
        (
            (
                "communication_bytes",
                "intra_node_communication_bytes",
                "inter_node_communication_bytes",
            ),
            "multi_node_communication_bytes_required_for_transport_evidence",
        ),
    ),
)
def test_multi_node_release_claim_rejects_backend_route_without_required_evidence(
    missing_fields,
    expected_blocker,
):
    payload = _statevector_claimable_payload(
        claim_evidence_type="production_training_benchmark",
        world_size=4,
        local_world_size=2,
        node_count=2,
        rank_placement=(
            {"rank": 0, "node_id": 0, "local_rank": 0},
            {"rank": 1, "node_id": 0, "local_rank": 1},
            {"rank": 2, "node_id": 1, "local_rank": 0},
            {"rank": 3, "node_id": 1, "local_rank": 1},
        ),
        communication_bytes=256,
        intra_node_communication_bytes=128,
        inter_node_communication_bytes=128,
        communication_plan={
            "communication_execution": "jax_pmap_amplitude_exchange_backward_replay",
            "transport_patterns": ("pair_exchange", "all_to_all"),
            "executed_collective_route": {
                "backend": "nccl",
                "route_scope": "multi_node",
                "collective": "all_to_all",
                "nodes": (0, 1),
            },
            "estimated_transfer_bytes": 256,
            "intra_node_communication_bytes": 128,
            "inter_node_communication_bytes": 128,
        },
        single_gpu_expected_oom=True,
        capacity_baseline_device="single_gpu_24gb",
        capacity_failure_reason="single_gpu_memory_budget_exceeded",
    )
    for field in missing_fields:
        payload.pop(field, None)
    if "communication_bytes" in missing_fields:
        payload["communication_plan"].pop("estimated_transfer_bytes", None)
        payload["communication_plan"].pop("intra_node_communication_bytes", None)
        payload["communication_plan"].pop("inter_node_communication_bytes", None)

    transport = fq.evaluate_distributed_transport_evidence(payload)
    contract = fq.evaluate_distributed_evidence_contract(payload)

    assert transport.status == "topology_dependent_planning"
    assert expected_blocker in transport.blockers
    assert contract.status == "blocked"
    assert contract.checks["multi_node_transport_evidence_complete"] is False
    with pytest.raises(fq.DistributedScalabilityError) as excinfo:
        fq.require_distributed_scalability(payload)
    assert expected_blocker in excinfo.value.audit.errors


def test_multi_node_release_claim_fails_closed_without_transport_route_evidence():
    payload = _statevector_claimable_payload(
        claim_evidence_type="production_training_benchmark",
        world_size=4,
        local_world_size=2,
        node_count=2,
        rank_placement=(
            {"rank": 0, "node_id": 0, "local_rank": 0},
            {"rank": 1, "node_id": 0, "local_rank": 1},
            {"rank": 2, "node_id": 1, "local_rank": 0},
            {"rank": 3, "node_id": 1, "local_rank": 1},
        ),
        communication_bytes=256,
        intra_node_communication_bytes=128,
        inter_node_communication_bytes=128,
        communication_plan={
            "communication_execution": "jax_pmap_amplitude_exchange_backward_replay",
            "transport_patterns": ("pair_exchange", "all_to_all"),
            "topology_dependency": "collective_route_depends_on_jax_runtime",
            "estimated_transfer_bytes": 256,
            "intra_node_communication_bytes": 128,
            "inter_node_communication_bytes": 128,
        },
        single_gpu_expected_oom=True,
        capacity_baseline_device="single_gpu_24gb",
        capacity_failure_reason="single_gpu_memory_budget_exceeded",
    )

    transport = fq.evaluate_distributed_transport_evidence(payload)
    contract = fq.evaluate_distributed_evidence_contract(payload)

    assert transport.status == "topology_dependent_planning"
    assert transport.topology_dependent is True
    assert "multi_node_production_transport_evidence_missing" in transport.blockers
    assert contract.status == "blocked"
    assert contract.checks["multi_node_transport_evidence_complete"] is False
    with pytest.raises(fq.DistributedScalabilityError) as excinfo:
        fq.require_distributed_scalability(payload)
    assert (
        "release gate requires multi_node_production_transport evidence for node_count > 1"
        in excinfo.value.audit.errors
    )


def test_benchmark_require_scalability_uses_release_gate_validator():
    payload = {
        "claim_evidence_type": "production_training_benchmark",
        "distribution_semantics": "sharded_across_ranks",
        "scalability_claim_allowed": True,
        "world_size": 2,
        "local_world_size": 2,
        "node_count": 1,
        "rank_shards": (
            {
                "rank": 0,
                "amplitude_start": 0,
                "amplitude_end": 4,
                "local_state_bytes": 64,
            },
            {
                "rank": 1,
                "amplitude_start": 4,
                "amplitude_end": 8,
                "local_state_bytes": 64,
            },
        ),
        "communication_bytes": 128,
        "single_gpu_expected_oom": False,
        "capacity_baseline_device": "single_gpu_24gb",
        "capacity_failure_reason": "fits_on_single_gpu",
        "gradient_distribution_semantics": "sharded_across_ranks",
    }
    payload = _with_sharded_optimizer_evidence(payload)
    audit = fq.validate_distributed_claim_evidence(payload)
    audit_errors = audit.errors
    assert audit.release_gate_allowed is False
    assert "release gate requires single_gpu_expected_oom=True" in audit_errors


def test_benchmark_require_scalability_accepts_release_grade_payload():
    payload = {
        "claim_evidence_type": "production_training_benchmark",
        "distribution_semantics": "sharded_across_ranks",
        "scalability_claim_allowed": True,
        "world_size": 2,
        "local_world_size": 2,
        "node_count": 1,
        "rank_shards": (
            {
                "rank": 0,
                "amplitude_start": 0,
                "amplitude_end": 4,
                "local_state_bytes": 64,
            },
            {
                "rank": 1,
                "amplitude_start": 4,
                "amplitude_end": 8,
                "local_state_bytes": 64,
            },
        ),
        "communication_bytes": 128,
        "single_gpu_expected_oom": True,
        "capacity_baseline_device": "single_gpu_24gb",
        "capacity_failure_reason": "single_gpu_memory_budget_exceeded",
        "gradient_distribution_semantics": "sharded_across_ranks",
    }
    payload = _with_sharded_optimizer_evidence(payload)
    audit = fq.require_distributed_scalability(payload)
    assert audit.release_gate_allowed is True


def test_benchmark_results_layout_separates_non_release_evidence():
    root = Path("benchmarks/results")
    expected_dirs = {"local", "comparison", "smoke", "scalability"}

    assert expected_dirs.issubset(
        {path.name for path in root.iterdir() if path.is_dir()}
    )
    assert not [
        path
        for path in root.glob("*.json")
        if not path.name.endswith("audit_summary.json")
    ]

    payload_count = 0
    for directory in (root / "local", root / "comparison", root / "smoke"):
        payload_paths = [
            path
            for path in directory.glob("*.json")
            if not path.name.endswith("audit_summary.json")
        ]
        payload_count += len(payload_paths)
        for path in payload_paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            audit = fq.audit_distributed_scalability(payload)
            assert audit.valid, path
            assert payload["scalability_claim_allowed"] is False
            assert payload["release_gate_allowed"] is False
            assert payload["non_release_evidence"] is True
            assert payload["benchmark_evidence_class"] in {
                "local_non_release",
                "comparison_non_release",
                "non_release_smoke",
            }
            assert payload.get("scalability_blockers"), path
            assert not audit.release_gate_allowed
    assert payload_count, "expected at least one checked-in non-release payload"


def test_legacy_smoke_payloads_are_not_release_claims():
    smoke_paths = sorted(
        Path("benchmarks/results/smoke").glob("statevector_*_smoke.json")
    )

    def collect_keys(value):
        if isinstance(value, dict):
            keys = set(value)
            for child in value.values():
                keys.update(collect_keys(child))
            return keys
        if isinstance(value, list):
            keys = set()
            for child in value:
                keys.update(collect_keys(child))
            return keys
        return set()

    assert smoke_paths
    for path in smoke_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        audit = fq.audit_distributed_scalability(payload)
        conclusion = payload["conclusion"]

        assert payload["claim_evidence_type"] == "development_smoke"
        assert payload["benchmark_evidence_class"] == "non_release_smoke"
        assert payload["scalability_claim_allowed"] is False
        assert payload["release_gate_allowed"] is False
        assert payload["single_gpu_expected_oom"] is False
        assert (
            "development_smoke_not_release_scalability_evidence"
            in payload["scalability_blockers"]
        )
        assert "development_smoke_not_promotable" in payload["release_gate_blockers"]
        assert "recommended_claim" not in collect_keys(payload)
        assert "non_release_observation" in conclusion
        assert "Development smoke only" in conclusion["non_release_observation"]
        assert (
            "not release scalability evidence" in conclusion["non_release_observation"]
        )
        assert audit.valid
        assert not audit.scalability_claim_allowed
        with pytest.raises(fq.DistributedScalabilityError):
            fq.require_distributed_scalability(payload)


def test_benchmark_results_root_scan_is_hygienic_after_issue010():
    paths = _json_files(Path("benchmarks/results"))
    summary = audit_paths(paths, require_scalability=False)

    assert summary["file_count"] == len(paths)
    assert summary["invalid_count"] == 0
    assert summary["claimable_count"] == len(summary["claimable_paths"])


def test_auxiliary_report_is_valid_but_never_claimable(tmp_path: Path):
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "artifact_class": "auxiliary_report",
                "schema": "flagquantum.test.summary.v1",
                "status": "passed",
            }
        ),
        encoding="utf-8",
    )

    summary = audit_paths([path], require_scalability=False)

    assert summary["invalid_count"] == 0
    assert summary["claimable_count"] == 0
    assert summary["records"][0]["status"] == "auxiliary"


def test_non_scalability_result_payloads_fail_strict_release_gate_after_issue010():
    non_release_dirs = (
        Path("benchmarks/results/local"),
        Path("benchmarks/results/comparison"),
        Path("benchmarks/results/smoke"),
    )
    payload_paths = sorted(
        path
        for directory in non_release_dirs
        for path in directory.glob("*.json")
        if not path.name.endswith("audit_summary.json")
    )

    assert payload_paths
    for path in payload_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = audit_paths([path], require_scalability=True)

        assert payload["non_release_evidence"] is True
        assert payload["scalability_claim_allowed"] is False
        assert payload["release_gate_allowed"] is False
        assert summary["file_count"] == 1
        assert summary["claimable_count"] == 0
        assert summary["invalid_count"] == 1
        with pytest.raises(fq.DistributedScalabilityError):
            fq.require_distributed_scalability(payload)


@pytest.mark.parametrize(
    ("readme_path", "required_phrases"),
    (
        (
            Path("benchmarks/results/README.md"),
            (
                "local/",
                "comparison/",
                "smoke/",
                "scalability/",
                "Only place for promoted payloads accepted by `--require-scalability`",
            ),
        ),
        (
            Path("benchmarks/results/local/README.md"),
            ("single-device fast-path", "not distributed scalability claims"),
        ),
        (
            Path("benchmarks/results/comparison/README.md"),
            (
                "not release-grade scalability evidence",
                "scalability_claim_allowed=false",
            ),
        ),
        (
            Path("benchmarks/results/smoke/README.md"),
            (
                "development",
                "not promoted scalability evidence",
                'claim_evidence_type="development_smoke"',
            ),
        ),
        (
            Path("benchmarks/results/scalability/README.md"),
            (
                "only benchmark-results location for release-grade",
                "--require-scalability",
            ),
        ),
    ),
)
def test_benchmark_results_readmes_document_issue010_evidence_boundaries(
    readme_path, required_phrases
):
    text = readme_path.read_text(encoding="utf-8")

    for phrase in required_phrases:
        assert phrase in text


def test_repository_capacity_model_fixture_is_not_promoted_release_evidence():
    path = Path(
        "benchmarks/fixtures/capacity_models/statevector_capacity_model_8rank.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    summary = audit_paths([path], require_scalability=True)

    assert payload["artifact_class"] == "plan"
    assert payload["benchmark_evidence_class"] == "capacity_model_fixture"
    assert payload["release_gate_allowed"] is False
    assert summary["file_count"] == 1
    assert summary["claimable_count"] == 0
    assert summary["invalid_count"] == 1


@pytest.mark.parametrize(
    ("state_mode", "expected_family"),
    (
        ("jax_sharded_statevector", "statevector"),
        ("jax_sharded_mps", "mps"),
        ("jax_sharded_tensor_network", "tensor_network"),
    ),
)
def test_distributed_evidence_contract_normalizes_backend_families(
    state_mode, expected_family
):
    payload = {
        "claim_evidence_type": "plan_preflight",
        "state_mode": state_mode,
        "distribution_semantics": "requires_runtime_summary",
        "intended_distribution_semantics": "sharded_across_ranks",
        "sharding_plan_available": True,
        "scalability_claim_allowed": False,
        "world_size": 2,
        "local_world_size": 2,
        "node_count": 1,
        "rank_ownership": ({"rank": 0}, {"rank": 1}),
        "local_memory_bytes_by_rank": (64, 64),
        "communication_tiers": {
            "model": "planned_collective",
            "estimated_transfer_bytes": 128,
        },
        "scalability_blockers": ("executor_pending",),
    }

    contract = fq.evaluate_distributed_evidence_contract(payload)

    assert contract.backend_family == expected_family
    assert contract.claim_evidence_type == "plan_preflight"
    assert contract.status == "preflight_only"
    assert contract.fail_closed
    assert contract.blockers == ("executor_pending",)
    assert contract.checks["sharding_semantics_available"] is True
    assert contract.checks["rank_ownership_reported"] is True


def test_distributed_evidence_contract_fails_closed_for_rank_local_claim():
    contract = fq.evaluate_distributed_evidence_contract(
        {
            "claim_evidence_type": "production_training_benchmark",
            "state_mode": "jax_sharded_mps",
            "distribution_semantics": "rank_local_replicated_kernel",
            "scalability_claim_allowed": True,
            "world_size": 2,
            "local_world_size": 2,
            "node_count": 1,
            "rank_ownership": ({"rank": 0}, {"rank": 1}),
            "local_memory_bytes_by_rank": (64, 64),
            "communication_bytes": 128,
            "single_gpu_expected_oom": True,
            "capacity_baseline_device": "single_gpu_24gb",
            "capacity_failure_reason": "single_gpu_memory_budget_exceeded",
            "gradient_distribution_semantics": "sharded_across_ranks",
            "optimizer_update_semantics": "sharded_across_ranks",
            "training_step_count": 1,
        }
    )

    assert contract.status == "blocked"
    assert contract.fail_closed
    assert not contract.claimable_production_training
    assert contract.checks["not_replicated_or_rank_local"] is False
    assert (
        "distributed evidence contract rejects replicated or rank-local semantics"
        in contract.errors
    )


@pytest.mark.parametrize(
    ("state_mode", "expected_family"),
    (
        ("jax_sharded_statevector", "statevector"),
        ("jax_sharded_mps", "mps"),
        ("jax_sharded_tensor_network", "tensor_network"),
    ),
)
@pytest.mark.parametrize(
    "bad_semantics", ("replicated_per_rank", "rank_local_replicated_kernel")
)
def test_distributed_evidence_contract_rejects_replicated_release_claims_across_backend_families(
    state_mode,
    expected_family,
    bad_semantics,
):
    payload = {
        "claim_evidence_type": "production_training_benchmark",
        "state_mode": state_mode,
        "distribution_semantics": bad_semantics,
        "scalability_claim_allowed": True,
        "world_size": 2,
        "local_world_size": 2,
        "node_count": 1,
        "rank_ownership": ({"rank": 0}, {"rank": 1}),
        "local_memory_bytes_by_rank": (64, 64),
        "communication_bytes": 128,
        "single_gpu_expected_oom": True,
        "capacity_baseline_device": "single_gpu_24gb",
        "capacity_failure_reason": "single_gpu_memory_budget_exceeded",
        "gradient_distribution_semantics": "sharded_across_ranks",
        "optimizer_update_semantics": "sharded_across_ranks",
        "training_step_count": 1,
        "parameter_ownership_semantics": "sharded_across_ranks",
        "parameter_ownership": (
            {"rank": 0, "parameters": (0,)},
            {"rank": 1, "parameters": (1,)},
        ),
        "gradient_ownership_semantics": "sharded_across_ranks",
        "gradient_ownership": (
            {"rank": 0, "parameters": (0,)},
            {"rank": 1, "parameters": (1,)},
        ),
        "optimizer_update_ownership_semantics": "sharded_across_ranks",
        "optimizer_update_ownership": (
            {"rank": 0, "parameters": (0,)},
            {"rank": 1, "parameters": (1,)},
        ),
    }

    contract = fq.evaluate_distributed_evidence_contract(payload)

    assert contract.backend_family == expected_family
    assert contract.status == "blocked"
    assert contract.fail_closed
    assert not contract.claimable_production_training
    assert contract.checks["not_replicated_or_rank_local"] is False
    assert contract.checks["rank_ownership_reported"] is True
    assert contract.checks["memory_plan_reported"] is True
    assert contract.checks["communication_plan_reported"] is True
    assert (
        "distributed evidence contract rejects replicated or rank-local semantics"
        in contract.errors
    )
    assert (
        "distributed evidence contract requires release evidence to report "
        "distribution_semantics='sharded_across_ranks'"
    ) in contract.errors


@pytest.mark.parametrize(
    ("state_mode", "expected_family"),
    (
        ("jax_sharded_statevector", "statevector"),
        ("jax_sharded_mps", "mps"),
        ("jax_sharded_tensor_network", "tensor_network"),
    ),
)
@pytest.mark.parametrize(
    ("missing_keys", "check_name", "expected_error"),
    (
        (
            ("rank_ownership", "local_memory_bytes_by_rank"),
            "rank_ownership_reported",
            "distributed evidence contract requires rank ownership",
        ),
        (
            ("local_memory_bytes_by_rank",),
            "memory_plan_reported",
            "distributed evidence contract requires per-rank memory evidence",
        ),
        (
            ("communication_bytes",),
            "communication_plan_reported",
            "distributed evidence contract requires communication evidence",
        ),
    ),
)
def test_distributed_evidence_contract_fails_closed_for_missing_evidence_across_backend_families(
    state_mode,
    expected_family,
    missing_keys,
    check_name,
    expected_error,
):
    payload = {
        "claim_evidence_type": "production_training_benchmark",
        "state_mode": state_mode,
        "distribution_semantics": "sharded_across_ranks",
        "scalability_claim_allowed": True,
        "world_size": 2,
        "local_world_size": 2,
        "node_count": 1,
        "rank_ownership": ({"rank": 0}, {"rank": 1}),
        "local_memory_bytes_by_rank": (64, 64),
        "communication_bytes": 128,
        "single_gpu_expected_oom": True,
        "capacity_baseline_device": "single_gpu_24gb",
        "capacity_failure_reason": "single_gpu_memory_budget_exceeded",
        "gradient_distribution_semantics": "sharded_across_ranks",
        "optimizer_update_semantics": "sharded_across_ranks",
        "training_step_count": 1,
        "parameter_ownership_semantics": "sharded_across_ranks",
        "parameter_ownership": (
            {"rank": 0, "parameters": (0,)},
            {"rank": 1, "parameters": (1,)},
        ),
        "gradient_ownership_semantics": "sharded_across_ranks",
        "gradient_ownership": (
            {"rank": 0, "parameters": (0,)},
            {"rank": 1, "parameters": (1,)},
        ),
        "optimizer_update_ownership_semantics": "sharded_across_ranks",
        "optimizer_update_ownership": (
            {"rank": 0, "parameters": (0,)},
            {"rank": 1, "parameters": (1,)},
        ),
    }
    for missing_key in missing_keys:
        payload.pop(missing_key)

    contract = fq.evaluate_distributed_evidence_contract(payload)

    assert contract.backend_family == expected_family
    assert contract.status == "blocked"
    assert contract.fail_closed
    assert not contract.claimable_production_training
    assert contract.checks[check_name] is False
    assert expected_error in contract.errors


def test_repository_scalability_payload_preserves_issue007_capacity_evidence():
    path = Path(
        "benchmarks/fixtures/capacity_models/statevector_capacity_model_8rank.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    world_size = payload["world_size"]
    rank_shards = payload["rank_shards"]
    rank_placement = payload["rank_placement"]
    local_memory = payload["local_memory_bytes_by_rank"]
    parameter_ownership = payload["parameter_ownership"]
    gradient_ownership = payload["gradient_ownership"]
    optimizer_update_ownership = payload["optimizer_update_ownership"]

    assert payload["claim_evidence_type"] == "plan_preflight"
    assert payload["release_payload"] is False
    assert payload["distribution_semantics"] == "sharded_across_ranks"
    assert payload["forward_distribution_semantics"] == "sharded_across_ranks"
    assert payload["backward_distribution_semantics"] == "sharded_across_ranks"
    assert payload["gradient_distribution_semantics"] == "sharded_across_ranks"
    assert payload["single_gpu_expected_oom"] is True
    assert (
        payload["capacity_baseline_device_memory_bytes"] < payload["total_state_bytes"]
    )
    assert (
        payload["memory_plan"]["single_device_reference_bytes"]
        == payload["total_state_bytes"]
    )
    assert payload["memory_plan"]["capacity_baseline_device_memory_bytes"] == (
        payload["capacity_baseline_device_memory_bytes"]
    )
    assert world_size == 8
    assert payload["local_world_size"] == 8
    assert payload["node_count"] == 1
    assert len(rank_shards) == world_size
    assert len(rank_placement) == world_size
    assert len(local_memory) == world_size
    assert len(parameter_ownership) == world_size
    assert len(gradient_ownership) == world_size
    assert len(optimizer_update_ownership) == world_size
    assert rank_shards[0]["amplitude_start"] == 0
    assert rank_shards[-1]["amplitude_end"] == payload["state_amplitudes"]
    assert (
        sum(item["owned_amplitudes"] for item in rank_shards)
        == payload["state_amplitudes"]
    )
    assert all(
        item["local_state_bytes"] == payload["per_rank_state_bytes"]
        for item in rank_shards
    )
    assert all(item["node_id"] == 0 for item in rank_placement)
    assert (
        payload["communication_plan"]["topology_scope"]
        == "single_node_executed_collective"
    )
    assert payload["intra_node_communication_bytes"] == payload["communication_bytes"]
    assert payload["inter_node_communication_bytes"] == 0
    assert "all_to_all" in payload["communication_plan"]["patterns"]
    assert payload["parameter_ownership_semantics"] == "sharded_across_ranks"
    assert payload["gradient_ownership_semantics"] == "sharded_across_ranks"
    assert payload["optimizer_update_semantics"] == "sharded_across_ranks"
    assert payload["optimizer_update_ownership_semantics"] == "sharded_across_ranks"
    assert payload["training_step_count"] > 0
    assert payload["blockers"] == []
    assert payload["scalability_blockers"] == [
        "cpu_labelled_capacity_model_not_reproduced_on_claimed_hardware",
        "runtime_provenance_missing",
    ]
    assert payload["gradient_blockers"] == []
    assert payload["production_blockers"] == []
    assert payload["readiness_blockers"] == []


def test_benchmark_require_scalability_rejects_implicit_release_payload():
    payload = {
        "claim_evidence_type": "release_payload",
        "distribution_semantics": "sharded_across_ranks",
        "scalability_claim_allowed": True,
        "world_size": 2,
        "local_world_size": 2,
        "node_count": 1,
        "rank_shards": (
            {
                "rank": 0,
                "amplitude_start": 0,
                "amplitude_end": 4,
                "local_state_bytes": 64,
            },
            {
                "rank": 1,
                "amplitude_start": 4,
                "amplitude_end": 8,
                "local_state_bytes": 64,
            },
        ),
        "communication_bytes": 128,
        "single_gpu_expected_oom": True,
        "capacity_baseline_device": "single_gpu_24gb",
        "capacity_failure_reason": "single_gpu_memory_budget_exceeded",
        "gradient_distribution_semantics": "sharded_across_ranks",
    }
    payload = _with_sharded_optimizer_evidence(payload)
    audit = fq.validate_distributed_claim_evidence(payload)
    audit_errors = audit.errors
    assert "release_payload evidence must be explicit" in audit_errors


def test_require_scalability_rejects_missing_optimizer_update_semantics():
    payload = {
        "claim_evidence_type": "production_training_benchmark",
        "distribution_semantics": "sharded_across_ranks",
        "scalability_claim_allowed": True,
        "world_size": 2,
        "local_world_size": 2,
        "node_count": 1,
        "rank_shards": (
            {
                "rank": 0,
                "amplitude_start": 0,
                "amplitude_end": 4,
                "local_state_bytes": 64,
            },
            {
                "rank": 1,
                "amplitude_start": 4,
                "amplitude_end": 8,
                "local_state_bytes": 64,
            },
        ),
        "communication_bytes": 128,
        "single_gpu_expected_oom": True,
        "capacity_baseline_device": "single_gpu_24gb",
        "capacity_failure_reason": "single_gpu_memory_budget_exceeded",
        "gradient_distribution_semantics": "sharded_across_ranks",
    }

    with pytest.raises(fq.DistributedScalabilityError) as excinfo:
        fq.require_distributed_scalability(payload)

    assert (
        "release gate requires optimizer_update_semantics='sharded_across_ranks'"
        in excinfo.value.audit.errors
    )
    assert "release gate requires training_step_count > 0" in excinfo.value.audit.errors


def test_require_scalability_rejects_missing_optimizer_ownership_evidence():
    payload = {
        "claim_evidence_type": "production_training_benchmark",
        "distribution_semantics": "sharded_across_ranks",
        "scalability_claim_allowed": True,
        "world_size": 2,
        "local_world_size": 2,
        "node_count": 1,
        "rank_shards": (
            {
                "rank": 0,
                "amplitude_start": 0,
                "amplitude_end": 4,
                "local_state_bytes": 64,
            },
            {
                "rank": 1,
                "amplitude_start": 4,
                "amplitude_end": 8,
                "local_state_bytes": 64,
            },
        ),
        "communication_bytes": 128,
        "single_gpu_expected_oom": True,
        "capacity_baseline_device": "single_gpu_24gb",
        "capacity_failure_reason": "single_gpu_memory_budget_exceeded",
        "optimizer_update_semantics": "sharded_across_ranks",
        "training_step_count": 3,
        "gradient_distribution_semantics": "sharded_across_ranks",
    }

    with pytest.raises(fq.DistributedScalabilityError) as excinfo:
        fq.require_distributed_scalability(payload)

    assert (
        "release gate requires parameter_ownership_semantics='sharded_across_ranks' with explicit parameter ownership"
        in excinfo.value.audit.errors
    )
    assert (
        "release gate requires gradient_ownership_semantics='sharded_across_ranks' with explicit gradient ownership"
        in excinfo.value.audit.errors
    )
    assert (
        "release gate requires optimizer_update_ownership_semantics='sharded_across_ranks' with explicit optimizer-update ownership"
        in excinfo.value.audit.errors
    )


@pytest.mark.parametrize(
    ("semantics_key", "expected_error"),
    (
        (
            "parameter_ownership_semantics",
            "release gate requires parameter_ownership_semantics='sharded_across_ranks' with explicit parameter ownership",
        ),
        (
            "gradient_ownership_semantics",
            "release gate requires gradient_ownership_semantics='sharded_across_ranks' with explicit gradient ownership",
        ),
        (
            "optimizer_update_ownership_semantics",
            "release gate requires optimizer_update_ownership_semantics='sharded_across_ranks' with explicit optimizer-update ownership",
        ),
    ),
)
@pytest.mark.parametrize(
    "bad_semantics", ("unknown", "replicated_per_rank", "rank_local_replicated_kernel")
)
def test_require_scalability_rejects_non_sharded_optimizer_ownership_semantics(
    semantics_key,
    expected_error,
    bad_semantics,
):
    payload = {
        "claim_evidence_type": "production_training_benchmark",
        "distribution_semantics": "sharded_across_ranks",
        "scalability_claim_allowed": True,
        "world_size": 2,
        "local_world_size": 2,
        "node_count": 1,
        "rank_shards": (
            {
                "rank": 0,
                "amplitude_start": 0,
                "amplitude_end": 4,
                "local_state_bytes": 64,
            },
            {
                "rank": 1,
                "amplitude_start": 4,
                "amplitude_end": 8,
                "local_state_bytes": 64,
            },
        ),
        "communication_bytes": 128,
        "single_gpu_expected_oom": True,
        "capacity_baseline_device": "single_gpu_24gb",
        "capacity_failure_reason": "single_gpu_memory_budget_exceeded",
        "gradient_distribution_semantics": "sharded_across_ranks",
    }
    payload = _with_sharded_optimizer_evidence(payload)
    payload[semantics_key] = bad_semantics

    with pytest.raises(fq.DistributedScalabilityError) as excinfo:
        fq.require_distributed_scalability(payload)

    assert expected_error in excinfo.value.audit.errors


def test_require_scalability_fails_closed_for_replicated_payload():
    payload = {
        "distribution_semantics": "rank_local_replicated_kernel",
        "scalability_claim_allowed": False,
        "world_size": 8,
    }

    with pytest.raises(fq.DistributedScalabilityError) as excinfo:
        fq.require_distributed_scalability(payload)

    assert excinfo.value.audit.distribution_semantics == "rank_local_replicated_kernel"
    assert not excinfo.value.audit.scalability_claim_allowed


def test_scalability_audit_rejects_single_device_fast_path_scalability_claim():
    payload = {
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": True,
        "world_size": 1,
    }

    audit = fq.audit_distributed_scalability(payload)

    assert not audit.valid
    assert (
        "single-device fast paths cannot claim distributed scalability" in audit.errors
    )


def test_scalability_audit_rejects_single_device_fast_path_with_multi_rank():
    payload = {
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "world_size": 2,
    }

    audit = fq.audit_distributed_scalability(payload)

    assert not audit.valid
    assert "single-device fast paths require world_size <= 1" in audit.errors


def test_scalability_audit_warns_for_hybrid_mps_without_claim():
    payload = {
        "distribution_semantics": "hybrid_sharded_forward_with_replicated_state",
        "scalability_claim_allowed": False,
        "world_size": 2,
        "scalability_blockers": (
            "full_local_mps_state_view",
            "replicated_mps_autograd",
        ),
    }

    audit = fq.audit_distributed_scalability(payload)

    assert audit.valid
    assert not audit.scalability_claim_allowed
    assert audit.errors == ()


def test_scalability_audit_accepts_planner_runtime_summary_required_without_claim():
    payload = {
        "distribution_semantics": "requires_runtime_summary",
        "scalability_claim_allowed": False,
        "world_size": 2,
        "scalability_blockers": ("runtime_summary_required_for_scalability_claim",),
    }

    audit = fq.audit_distributed_scalability(payload)

    assert audit.valid
    assert not audit.scalability_claim_allowed
    assert audit.errors == ()


def test_scalability_audit_requires_evidence_for_sharded_claim():
    payload = {
        "distribution_semantics": "sharded_across_ranks",
        "scalability_claim_allowed": True,
        "world_size": 2,
    }

    audit = fq.audit_distributed_scalability(payload)

    assert not audit.valid
    assert "sharded payload must report local_world_size" in audit.errors
    assert "sharded payload must report node_count" in audit.errors
    assert (
        "sharded payload must report per-rank ownership or rank placement"
        in audit.errors
    )


def test_scalability_audit_rejects_empty_sharded_evidence_containers():
    payload = {
        "distribution_semantics": "sharded_across_ranks",
        "scalability_claim_allowed": True,
        "world_size": 2,
        "local_world_size": 2,
        "node_count": 1,
        "rank_shards": (),
        "local_memory_bytes_by_rank": (),
        "communication_tiers": {},
    }

    audit = fq.audit_distributed_scalability(payload)

    assert not audit.valid
    assert not audit.scalability_claim_allowed
    assert (
        "sharded payload must report per-rank ownership or rank placement"
        in audit.errors
    )
    assert "sharded payload must report per-rank memory evidence" in audit.errors
    assert "sharded payload must report communication evidence" in audit.errors


def test_scalability_audit_accepts_rank_shard_memory_without_duplicate_memory_list():
    payload = {
        "distribution_semantics": "sharded_across_ranks",
        "scalability_claim_allowed": True,
        "world_size": 2,
        "local_world_size": 2,
        "node_count": 1,
        "rank_shards": (
            {
                "rank": 0,
                "amplitude_start": 0,
                "amplitude_end": 4,
                "local_state_bytes": 64,
            },
            {
                "rank": 1,
                "amplitude_start": 4,
                "amplitude_end": 8,
                "local_state_bytes": 64,
            },
        ),
        "estimated_transfer_bytes": 0,
        "scalability_blockers": (),
    }

    audit = fq.audit_distributed_scalability(payload)

    assert audit.valid
    assert audit.scalability_claim_allowed
    assert not audit.release_gate_allowed
    assert audit.errors == ()


def test_scalability_audit_rejects_rank_ownership_without_memory_evidence():
    payload = {
        "distribution_semantics": "sharded_across_ranks",
        "scalability_claim_allowed": True,
        "world_size": 2,
        "local_world_size": 2,
        "node_count": 1,
        "rank_shards": (
            {"rank": 0, "amplitude_start": 0, "amplitude_end": 4},
            {"rank": 1, "amplitude_start": 4, "amplitude_end": 8},
        ),
        "communication_tiers": {"model": "rank_endpoint_attributed"},
    }

    audit = fq.audit_distributed_scalability(payload)

    assert not audit.valid
    assert not audit.scalability_claim_allowed
    assert "sharded payload must report per-rank memory evidence" in audit.errors


def _mps_backward_preflight_payload():
    return {
        "state_mode": "jax_sharded_mps",
        "claim_evidence_type": "plan_preflight",
        "distribution_semantics": "sharded_across_ranks",
        "intended_distribution_semantics": "sharded_across_ranks",
        "mps_forward_distribution_semantics": "sharded_across_ranks",
        "mps_backward_distribution_semantics": "sharded_across_ranks",
        "world_size": 2,
        "local_world_size": 2,
        "node_count": 1,
        "rank_ownership": ({"rank": 0}, {"rank": 1}),
        "local_memory_bytes_by_rank": (128, 128),
        "communication_bytes": 64,
        "site_shard_ownership": (
            {"rank": 0, "wires": (0, 1)},
            {"rank": 1, "wires": (2, 3)},
        ),
        "bond_shard_ownership": (
            {"left_rank": 0, "right_rank": 1, "left_wire": 1, "right_wire": 2},
        ),
        "parameter_gradient_ownership": (
            {"rank": 0, "parameter_indices": (0,)},
            {"rank": 1, "parameter_indices": (1,)},
        ),
        "boundary_gradient_ownership": ({"left_rank": 0, "right_rank": 1},),
        "boundary_adjoint_exchange": {
            "status": "planned",
            "route": "adjacent_rank_p2p",
        },
        "boundary_gradient_routes": (
            {"left_rank": 0, "right_rank": 1, "route": "adjacent_rank_p2p"},
        ),
        "canonicalization_backward_strategy": {
            "status": "planned",
            "strategy": "cross_shard_canonicalization_pullback",
        },
        "truncation_gradient_metadata": {"status": "not_required"},
        "mps_backward_memory_plan": {
            "status": "complete_estimate",
            "rank_memory": (
                {
                    "rank": 0,
                    "forward_tensor_bytes": 128,
                    "backward_adjoint_bytes": 128,
                    "boundary_gradient_buffer_bytes": 32,
                    "canonicalization_temporary_bytes": 64,
                    "truncation_temporary_bytes": 0,
                    "estimated_peak_backward_bytes": 352,
                },
                {
                    "rank": 1,
                    "forward_tensor_bytes": 128,
                    "backward_adjoint_bytes": 128,
                    "boundary_gradient_buffer_bytes": 32,
                    "canonicalization_temporary_bytes": 64,
                    "truncation_temporary_bytes": 0,
                    "estimated_peak_backward_bytes": 352,
                },
            ),
            "forward_tensor_bytes_by_rank": (128, 128),
            "backward_adjoint_bytes_by_rank": (128, 128),
            "boundary_gradient_buffer_bytes_by_rank": (32, 32),
            "canonicalization_temporary_bytes_by_rank": (64, 64),
            "truncation_temporary_bytes_by_rank": (0, 0),
            "per_rank_peak_bytes": (352, 352),
            "local_memory_bytes_by_rank": (352, 352),
            "communication_buffer_bytes": 64,
        },
        "mps_backward_communication_plan": {
            "status": "planned_not_executed",
            "boundary_edge_count": 1,
            "boundary_edges": (
                {
                    "boundary_edge_id": "edge_0:1-2",
                    "left_rank": 0,
                    "right_rank": 1,
                    "communication_bytes": 64,
                    "communication_primitive": "planned_adjacent_rank_point_to_point",
                    "topology_tier": "intra_node",
                    "topology_route": "planned_intra_node_route",
                    "execution_status": "planned_not_executed",
                },
            ),
            "communication_bytes": 64,
            "boundary_adjoint_exchange": "adjacent_rank_p2p",
        },
        "optimizer_update_semantics": "not_measured",
        "optimizer_update_ownership": (),
        "fallback_semantics": "none",
        "scalability_claim_allowed": False,
        "blockers": (),
    }


def _with_production_mps_resource_evidence(payload):
    payload["mps_backward_memory_plan"]["status"] = "production_measured"
    communication = payload["mps_backward_communication_plan"]
    communication["status"] = "production_executed"
    for edge in communication["boundary_edges"]:
        edge["execution_status"] = "executed"
    return payload


@pytest.mark.release_gate
@pytest.mark.benchmark_contract
def test_mps_backward_gate_accepts_static_preflight_but_not_release_claim():
    gate = evaluate_mps_backward_readiness(_mps_backward_preflight_payload())
    summary = gate.summary()

    assert summary["contract_version"] == "distributed_evidence_contract_v1"
    assert summary["status"] == "backward_preflight_ready"
    assert summary["production_training_claimable"] is False
    assert summary["fail_closed"] is True
    assert "mps_optimizer_update_semantics_not_measured" in summary["blockers"]


@pytest.mark.release_gate
def test_mps_optimizer_ownership_can_clear_only_optimizer_blockers():
    payload = _with_sharded_optimizer_evidence(
        _mps_backward_preflight_payload(),
        training_step_count=1,
    )

    gate = evaluate_mps_backward_readiness(payload)

    assert gate.checks["optimizer_parameter_ownership_sharded"] is True
    assert gate.checks["optimizer_gradient_ownership_sharded"] is True
    assert gate.checks["optimizer_update_sharded"] is True
    assert gate.checks["optimizer_update_ownership_sharded"] is True
    assert gate.checks["optimizer_ownership_aligned"] is True
    assert gate.checks["optimizer_training_steps_executed"] is True
    assert not any("optimizer" in blocker for blocker in gate.blockers)
    assert gate.checks["production_backward_execution_reported"] is False
    assert gate.status == "backward_preflight_ready"
    assert gate.production_training_claimable is False


@pytest.mark.release_gate
def test_mps_optimizer_ownership_rejects_mismatched_owner_and_missing_writeback():
    payload = _with_sharded_optimizer_evidence(
        _mps_backward_preflight_payload(),
        training_step_count=1,
    )
    mismatched = dict(payload)
    mismatched_gradients = [dict(item) for item in payload["gradient_ownership"]]
    mismatched_gradients[0]["rank"] = 1
    mismatched["gradient_ownership"] = tuple(mismatched_gradients)

    mismatched_gate = evaluate_mps_backward_readiness(mismatched)

    assert mismatched_gate.checks["optimizer_ownership_aligned"] is False
    assert "mps_optimizer_ownership_not_aligned" in mismatched_gate.blockers
    assert mismatched_gate.production_training_claimable is False

    missing_route = dict(payload)
    update_ownership = [dict(item) for item in payload["optimizer_update_ownership"]]
    update_ownership[0].pop("writeback_route")
    missing_route["optimizer_update_ownership"] = tuple(update_ownership)

    missing_route_gate = evaluate_mps_backward_readiness(missing_route)

    assert missing_route_gate.checks["optimizer_update_ownership_reported"] is False
    assert "mps_optimizer_update_ownership_incomplete" in missing_route_gate.blockers
    assert missing_route_gate.production_training_claimable is False

    replicated = dict(payload)
    replicated_ownership = [
        dict(item, writeback_route="replicated_all_rank_writeback")
        for item in payload["optimizer_update_ownership"]
    ]
    replicated["optimizer_update_ownership"] = tuple(replicated_ownership)

    replicated_gate = evaluate_mps_backward_readiness(replicated)

    assert replicated_gate.checks["optimizer_update_ownership_reported"] is False
    assert replicated_gate.production_training_claimable is False

    zero_step = dict(payload)
    zero_step["training_step_count"] = 0

    zero_step_gate = evaluate_mps_backward_readiness(zero_step)

    assert zero_step_gate.checks["optimizer_training_steps_executed"] is False
    assert "mps_optimizer_training_step_not_executed" in zero_step_gate.blockers
    assert zero_step_gate.production_training_claimable is False


@pytest.mark.release_gate
def test_mps_production_execution_rejects_preflight_resource_estimates():
    payload = _mps_backward_preflight_payload()
    payload.update(
        {
            "claim_evidence_type": "production_runtime",
            "backward_execution": "jax_pmap_backward",
        }
    )

    gate = evaluate_mps_backward_readiness(payload)

    assert gate.checks["production_backward_execution_reported"] is True
    assert gate.checks["production_memory_evidence_reported"] is False
    assert gate.checks["production_communication_evidence_reported"] is False
    assert "mps_production_backward_memory_not_measured" in gate.blockers
    assert "mps_production_backward_communication_not_executed" in gate.blockers
    assert gate.status == "backward_preflight_ready"
    assert gate.production_training_claimable is False


@pytest.mark.release_gate
def test_mps_production_runtime_evidence_cannot_bypass_shared_release_contract():
    payload = _mps_backward_preflight_payload()
    payload.update(
        {
            "claim_evidence_type": "production_runtime",
            "backward_execution": "jax_pmap_backward",
            "optimizer_update_semantics": "sharded_across_ranks",
            "optimizer_update_ownership": (
                {"rank": 0, "parameter_indices": (0,)},
                {"rank": 1, "parameter_indices": (1,)},
            ),
        }
    )

    payload = _with_production_mps_resource_evidence(payload)
    gate = evaluate_mps_backward_readiness(payload)

    assert gate.status == "production_backward_evidence"
    assert gate.checks["shared_release_contract_claimable"] is False
    assert gate.production_training_claimable is False
    assert gate.fail_closed is True


@pytest.mark.release_gate
def test_mps_backward_gate_requires_shared_release_contract_to_claim_training():
    payload = _mps_backward_preflight_payload()
    payload.update(
        {
            "claim_evidence_type": "production_training_benchmark",
            "scalability_claim_allowed": True,
            "backward_execution": "jax_pmap_mps_boundary_adjoint_backward",
            "gradient_distribution_semantics": "sharded_across_ranks",
            "single_gpu_expected_oom": True,
            "capacity_baseline_device": "single_gpu_24gb",
            "capacity_failure_reason": "single_gpu_memory_budget_exceeded",
        }
    )
    payload = _with_sharded_optimizer_evidence(payload)
    payload = _with_production_mps_resource_evidence(payload)
    payload["boundary_adjoint_exchange"]["status"] = "production_executed"

    gate = evaluate_mps_backward_readiness(payload)
    summary = gate.summary()

    assert summary["status"] == "production_training_claimable"
    assert summary["production_training_claimable"] is True
    assert summary["fail_closed"] is False
    assert summary["blockers"] == ()
    assert summary["errors"] == ()
    assert summary["checks"]["shared_release_contract_claimable"] is True
    assert fq.require_distributed_scalability(payload).release_gate_allowed is True


def test_mps_backward_gate_remains_internal_metadata_not_top_level_api():
    assert hasattr(fq.audit, "evaluate_mps_backward_readiness")
    assert not hasattr(fq, "evaluate_mps_backward_readiness")
    assert not hasattr(fq, "Phase5MPSBackwardReadinessGate")


@pytest.mark.release_gate
@pytest.mark.benchmark_contract
@pytest.mark.parametrize(
    ("field", "expected_blocker"),
    (
        (
            "mps_backward_distribution_semantics",
            "mps_backward_distribution_not_sharded",
        ),
        ("site_shard_ownership", "mps_backward_site_shard_ownership_incomplete"),
        ("bond_shard_ownership", "mps_backward_bond_shard_ownership_incomplete"),
        (
            "parameter_gradient_ownership",
            "mps_parameter_gradient_ownership_incomplete",
        ),
        (
            "boundary_gradient_ownership",
            "mps_boundary_gradient_ownership_incomplete",
        ),
        ("boundary_adjoint_exchange", "mps_boundary_gradient_exchange_pending"),
        ("boundary_gradient_routes", "mps_boundary_gradient_routes_incomplete"),
        (
            "canonicalization_backward_strategy",
            "mps_canonicalization_backward_strategy_pending",
        ),
        (
            "truncation_gradient_metadata",
            "mps_truncation_gradient_metadata_incomplete",
        ),
        ("mps_backward_memory_plan", "mps_backward_memory_plan_incomplete"),
        (
            "mps_backward_communication_plan",
            "mps_backward_communication_plan_incomplete",
        ),
    ),
)
def test_mps_backward_gate_fails_closed_for_missing_backward_evidence(
    field, expected_blocker
):
    payload = _mps_backward_preflight_payload()
    payload.pop(field)

    gate = evaluate_mps_backward_readiness(payload)

    assert gate.production_training_claimable is False
    assert gate.fail_closed is True
    assert expected_blocker in gate.blockers


@pytest.mark.release_gate
@pytest.mark.parametrize(
    "field",
    (
        "forward_tensor_bytes_by_rank",
        "backward_adjoint_bytes_by_rank",
        "boundary_gradient_buffer_bytes_by_rank",
        "canonicalization_temporary_bytes_by_rank",
        "truncation_temporary_bytes_by_rank",
        "per_rank_peak_bytes",
        "rank_memory",
    ),
)
def test_mps_backward_gate_rejects_incomplete_rank_memory(field):
    payload = _mps_backward_preflight_payload()
    payload["mps_backward_memory_plan"].pop(field)

    gate = evaluate_mps_backward_readiness(payload)

    assert gate.checks["backward_memory_plan_reported"] is False
    assert "mps_backward_memory_plan_incomplete" in gate.blockers


@pytest.mark.release_gate
@pytest.mark.parametrize(
    "field",
    (
        "communication_bytes",
        "communication_primitive",
        "topology_tier",
        "topology_route",
        "execution_status",
    ),
)
def test_mps_backward_gate_rejects_incomplete_boundary_edge(field):
    payload = _mps_backward_preflight_payload()
    payload["mps_backward_communication_plan"]["boundary_edges"][0].pop(field)

    gate = evaluate_mps_backward_readiness(payload)

    assert gate.checks["backward_communication_plan_reported"] is False
    assert "mps_backward_communication_plan_incomplete" in gate.blockers


@pytest.mark.release_gate
@pytest.mark.benchmark_contract
@pytest.mark.parametrize(
    ("fallback_semantics", "expected_blocker"),
    (
        ("replicated_mps_autograd", "mps_replicated_autograd_not_claimable"),
        ("full_local_mps_state_view", "mps_full_local_state_view_not_claimable"),
        ("full_local_mps_replay", "mps_full_local_mps_replay_not_claimable"),
        ("statevector_fallback", "mps_statevector_fallback_not_claimable"),
    ),
)
def test_mps_backward_gate_rejects_fallback_semantics(
    fallback_semantics, expected_blocker
):
    payload = _mps_backward_preflight_payload()
    payload["fallback_semantics"] = fallback_semantics

    gate = evaluate_mps_backward_readiness(payload)

    assert gate.status == "blocked"
    assert gate.production_training_claimable is False
    assert expected_blocker in gate.blockers


@pytest.mark.release_gate
@pytest.mark.benchmark_contract
def test_mps_cpu_preflight_evidence_cannot_be_promoted_to_release_claim():
    payload = _with_sharded_optimizer_evidence(
        _mps_backward_preflight_payload(),
        training_step_count=1,
    )
    payload.update(
        {
            "claim_evidence_type": "production_training_benchmark",
            "scalability_claim_allowed": True,
            "backward_execution": "local_simulated_backward",
            "gradient_distribution_semantics": "sharded_across_ranks",
            "single_gpu_expected_oom": True,
            "capacity_baseline_device": "single_gpu_24gb",
            "capacity_failure_reason": "single_gpu_memory_budget_exceeded",
            "fallback_semantics": "local_simulation",
        }
    )
    payload["boundary_adjoint_exchange"]["status"] = "local_cpu_executed"
    payload["mps_backward_communication_plan"]["status"] = "local_cpu_accounted"
    for edge in payload["mps_backward_communication_plan"]["boundary_edges"]:
        edge["execution_status"] = "local_cpu_executed"

    gate = evaluate_mps_backward_readiness(payload)
    with pytest.raises(fq.DistributedScalabilityError) as excinfo:
        fq.require_distributed_scalability(payload)

    assert gate.production_training_claimable is False
    assert gate.fail_closed is True
    assert "mps_local_simulation_not_claimable" in gate.blockers
    assert "MPS release gate requires executed production backward evidence" in str(
        excinfo.value
    )
    assert (
        "MPS release gate requires production-measured per-rank backward memory evidence"
        in str(excinfo.value)
    )
    assert "MPS release gate rejects local replay" in str(excinfo.value)


@pytest.mark.release_gate
@pytest.mark.benchmark_contract
def test_mps_backward_gate_rejects_local_simulation_and_nonempty_blockers():
    local_payload = _mps_backward_preflight_payload()
    local_payload["claim_evidence_type"] = "development_smoke"
    local_payload["fallback_semantics"] = "local_simulation"
    local_gate = evaluate_mps_backward_readiness(local_payload)

    blocked_payload = _mps_backward_preflight_payload()
    blocked_payload["blockers"] = ("production_boundary_runtime_pending",)
    blocked_gate = evaluate_mps_backward_readiness(blocked_payload)

    assert local_gate.status == "blocked"
    assert "mps_local_simulation_not_claimable" in local_gate.blockers
    assert blocked_gate.status == "blocked"
    assert "production_boundary_runtime_pending" in blocked_gate.blockers
    assert blocked_gate.production_training_claimable is False
