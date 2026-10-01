"""Contract tests for the two-node CUDA statevector probe."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _ROOT / "tools" / "probe_cuda_multinode_statevector.py"
_A800_ARTIFACT = (
    _ROOT / "artifacts" / "cuda_multinode_statevector_a800_jp171_jp172_20260930.json"
)
_SPEC = importlib.util.spec_from_file_location("cuda_multinode_probe", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_the_launched_shape_is_the_only_one_the_probe_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    assert _MODULE._declared_shape() == (4, 2)

    for world_size, local_world_size in (
        # One node wearing the pair's label.
        (2, 2),
        # Three nodes, which this lane cannot reach.
        (6, 2),
        # Ranks split unevenly, so a node holds fewer than it was told.
        (6, 4),
        # A world size the amplitude planner cannot shard by address bits.
        (3, 1),
    ):
        monkeypatch.setenv("WORLD_SIZE", str(world_size))
        monkeypatch.setenv("LOCAL_WORLD_SIZE", str(local_world_size))
        with pytest.raises(RuntimeError):
            _MODULE._declared_shape()


def test_network_observation_requires_the_declared_socket_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "nccl.log"
    log.write_text("NCCL INFO NET/Socket : Using [0]ens22f0:192.0.2.3\n")
    monkeypatch.setenv("NCCL_SOCKET_IFNAME", "ens22f0")
    monkeypatch.setenv("NCCL_IB_DISABLE", "1")

    observation = _MODULE._network_observation(log)

    assert observation["route"] == "socket"
    assert observation["socket_transport_observed"] is True
    assert observation["configured_interface_observed"] is True
    assert observation["evidence_level"] == "observed_debug_log"


def test_network_observation_rejects_a_missing_interface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "nccl.log"
    log.write_text("NCCL INFO NET/Socket : Using [0]eth0:10.0.0.1\n")
    monkeypatch.setenv("NCCL_SOCKET_IFNAME", "ens22f0")
    monkeypatch.setenv("NCCL_IB_DISABLE", "1")

    with pytest.raises(RuntimeError, match="does not confirm"):
        _MODULE._network_observation(log)


def test_checked_in_a800_multinode_evidence_is_narrow_and_self_consistent() -> None:
    payload = json.loads(_A800_ARTIFACT.read_text(encoding="utf-8"))
    evidence = payload["evidence"]
    encoded = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )

    assert payload["evidence_sha256"] == hashlib.sha256(encoded).hexdigest()
    assert evidence["status"] == "passed"
    assert evidence["scope"] == {
        "distribution_semantics": "sharded_across_ranks",
        "dtype": "complex128",
        "execution": "forward_backward_optimizer_and_checkpoint_resume",
        "local_world_size": 1,
        "n_wires": 5,
        "node_count": 2,
        "world_size": 2,
    }
    assert evidence["scalability_claim_allowed"] is False
    assert evidence["release_gate_allowed"] is False

    observations = evidence["observations"]
    metrics = observations["numerical_metrics"]
    # Complex128 against a single-device reference, so these are round-off
    # rather than a modelling allowance, and they hold for the gradient and the
    # training trajectory as well as for the statevector.
    for name in (
        "statevector_max_abs_error",
        "statevector_norm_error",
        "statevector_infidelity",
        "expectation_value_error",
        "gradient_max_abs_error",
        "first_leg_initial_expectation_error",
        "resume_prefix_max_abs_error",
        "resume_trajectory_max_abs_error",
    ):
        assert metrics[name] <= 1e-10, (name, metrics[name])
    # An optimizer that did not move the expectation would make every resume
    # comparison agree for the wrong reason.
    assert abs(metrics["training_loss_decrease"]) > 0

    network = observations["network"]
    assert network["route"] == "socket"
    assert network["configured_interface"] == "ens22f0"
    assert network["socket_transport_observed"] is True
    assert network["configured_interface_observed"] is True

    ranks = observations["rank_records"]
    assert {item["rank"] for item in ranks} == {0, 1}
    assert len({item["hostname_sha256"] for item in ranks}) == 2
    assert len({item["device_uuid"] for item in ranks}) == 2
    assert all(item["rank_ownership"]["local_amplitudes"] == 16 for item in ranks)
    assert all(item["inter_node_communication_count"] > 0 for item in ranks)
    assert all(item["inter_node_communication_bytes"] > 0 for item in ranks)
    # One node per rank is the placement the whole artifact is about. If the
    # plan resolved it as intra-node, the exchange below would prove nothing.
    assert all(item["intra_node_communication_count"] == 0 for item in ranks)
    assert all(item["node_count"] == 2 for item in ranks)
    assert all(item["local_world_size"] == 1 for item in ranks)
    # The backward and the training legs carry their own placement, because a
    # forward result that resolved two nodes does not place the other two.
    assert all(item["backward"]["node_count"] == 2 for item in ranks)
    assert all(item["backward"]["local_world_size"] == 1 for item in ranks)
    assert all(item["backward"]["backend"] == "nccl" for item in ranks)
    assert all(item["training"]["node_count"] == 2 for item in ranks)
    assert all(item["training"]["local_world_size"] == 1 for item in ranks)

    training = observations["training"]
    assert training["resumed_start_step"] == training["checkpoint_steps"] == 2
    assert training["uninterrupted_start_step"] == 0
    # The resumed leg computes the steps the uninterrupted run computed after
    # the checkpoint, and nothing else.
    assert training["resumed_leg_losses"] == (
        training["uninterrupted_leg_losses"][training["checkpoint_steps"] :]
    )
    assert training["first_leg_losses"] == (
        training["uninterrupted_leg_losses"][: training["checkpoint_steps"]]
    )
    # A checkpoint per rank, on a filesystem both nodes mounted.
    assert len(training["checkpoint_files"]) >= 2

    assert "production_performance_not_measured" in evidence["claim_blockers"]
    # The two blockers this probe exists to remove must be gone: a forward-only
    # artifact that still carried them would not support a training claim.
    assert "distributed_gradient_not_tested" not in evidence["claim_blockers"]
    assert "checkpoint_restart_not_tested" not in evidence["claim_blockers"]


def _artifact_kwargs(**overrides: object) -> dict:
    """The shape `_artifact` needs, with every observation retracted by default."""

    kwargs: dict = {
        "rank_records": [],
        "metrics": {},
        "network": {
            "evidence_level": "observed_debug_log",
            "route": "infiniband",
            "configured_interface": "ens22f0",
            "configured_interface_observed": True,
            "infiniband_transport_observed": True,
            "socket_transport_observed": False,
            "roce_transport_observed": True,
            "gpu_direct_observed": True,
            "configured_infiniband_disabled": False,
        },
        "training": {},
        "performance": {
            "measured": True,
            "synchronized": True,
            "measurement": _MODULE.MEASUREMENT,
            "warmup_iterations": _MODULE.MEASUREMENT_WARMUP_ITERATIONS,
            "measured_iterations": _MODULE.MEASUREMENT_ITERATIONS,
            "seconds": [0.001, 0.0011, 0.0009, 0.001, 0.0012],
        },
        "host_staging": {
            "profiled": True,
            "profiled_workload": _MODULE.MEASUREMENT,
            "host_transfer_observed": False,
            "host_transfer_events": [],
        },
        "world_size": 2,
        "local_world_size": 1,
    }
    kwargs.update(overrides)
    return kwargs


def test_every_retracted_blocker_needs_its_own_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absence is not evidence: each retraction is derived from a positive fact."""

    monkeypatch.setattr(_MODULE.torch.cuda.nccl, "version", lambda: (2, 29, 7))

    def blockers(**overrides: object) -> set[str]:
        payload = _MODULE._artifact(**_artifact_kwargs(**overrides))
        return set(payload["evidence"]["claim_blockers"])

    complete = blockers()
    # The pair and the circuit are declared boundaries, so nothing this run does
    # can retract them.
    assert {
        "two_node_pair_only_no_wider_topology",
        "toy_circuit_parameters_only",
        "validation_only_tiny_full_state_gather",
    } <= complete
    assert (
        complete
        - {
            "two_node_pair_only_no_wider_topology",
            "toy_circuit_parameters_only",
            "validation_only_tiny_full_state_gather",
        }
        == set()
    )

    # A socket run, and a run with no debug log to read, both leave the fabric
    # untested: the route cannot be asserted from the configuration alone.
    assert "rdma_not_tested" in blockers(
        network={
            "evidence_level": "observed_debug_log",
            "route": "socket",
            "socket_transport_observed": True,
            "infiniband_transport_observed": False,
        }
    )
    assert "rdma_not_tested" in blockers(network={})

    # An absent measurement is not a measurement of zero.
    assert "production_performance_not_measured" in blockers(performance=None)

    # A profiler that could not run has shown nothing, and a transfer it did see
    # is a finding that keeps the blocker rather than one that hides it.
    for staging in (
        {"profiled": False, "profiler_error": "RuntimeError: no profiler"},
        {"profiled": True},
    ):
        assert "hidden_host_staging_not_audited" in blockers(host_staging=staging)

    # An audit that ran and found a transfer reports the finding under its own
    # name. Reporting "nobody looked" there would be false, and dropping the
    # blocker altogether would retract a limitation the audit just confirmed.
    found = blockers(
        host_staging={
            "profiled": True,
            "profiled_workload": _MODULE.MEASUREMENT,
            "host_transfer_observed": True,
            "host_transfer_events": [
                {"name": "memcpy_DtoH", "direction": "device_to_host"}
            ],
        }
    )
    assert "host_staging_in_measured_region" in found
    assert "hidden_host_staging_not_audited" not in found
