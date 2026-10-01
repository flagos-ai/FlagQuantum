"""Contract tests for the two-node CUDA MPS probe."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _ROOT / "tools" / "probe_cuda_multinode_mps.py"
_SPEC = importlib.util.spec_from_file_location("cuda_multinode_mps_probe", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_A800_ARTIFACT = (
    _ROOT / "artifacts" / "cuda_multinode_mps_a800_jp171_jp172_20260930.json"
)


def _artifact() -> dict:
    return json.loads(_A800_ARTIFACT.read_text(encoding="utf-8"))


def test_the_launched_shape_is_the_only_one_the_probe_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "6")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "3")
    assert _MODULE._declared_shape() == (6, 3)

    for world_size, local_world_size in (
        # One node wearing the pair's label.
        (4, 4),
        # Three nodes, which this lane cannot reach.
        (6, 2),
        # Ranks split unevenly, so a node holds fewer than it was told.
        (6, 4),
    ):
        monkeypatch.setenv("WORLD_SIZE", str(world_size))
        monkeypatch.setenv("LOCAL_WORLD_SIZE", str(local_world_size))
        with pytest.raises(RuntimeError):
            _MODULE._declared_shape()


def test_each_declared_leg_has_to_report_the_launched_shape() -> None:
    good = {
        "world_size": 2,
        "local_world_size": 1,
        "node_count": 2,
        "distribution_semantics": "sharded_across_ranks",
    }

    _MODULE._require_two_node_placement(good, leg="test", expected_world_size=2)
    # The check is against the launched width rather than a constant, so a
    # two-rank leg and a four-rank leg are both accepted when both are what the
    # caller asked for.
    _MODULE._require_two_node_placement(
        {**good, "world_size": 4, "local_world_size": 2},
        leg="test",
        expected_world_size=4,
    )

    for wrong in (
        # Two ranks on one node is a different claim that this probe does not
        # make, however the ranks are counted.
        {**good, "local_world_size": 2, "node_count": 1},
        # A single rank would validate the arithmetic and nothing else.
        {**good, "world_size": 1, "node_count": 1},
        # A leg that resolved a width the launch did not have means the declared
        # placement never reached it.
        {**good, "world_size": 4, "local_world_size": 2},
        # A bespoke semantics string would fail the shared evidence contract.
        {**good, "distribution_semantics": "site_sharded"},
    ):
        with pytest.raises(RuntimeError):
            _MODULE._require_two_node_placement(
                wrong, leg="test", expected_world_size=2
            )


def test_a_sharded_parameter_has_to_cross_the_host_boundary() -> None:
    """The reduction is only inter-node if the owners are on different hosts."""

    def entry(owners: tuple[int, ...], count: int = 2) -> dict:
        return {
            "parameter_index": 0,
            "owner_ranks": list(owners),
            "occurrence_count": count,
            "reduction": "all_reduce_sum",
        }

    # One rank per node: every pair of distinct ranks straddles the boundary.
    assert _MODULE._split_parameter_ownership(
        [entry((0, 1)), {"parameter_index": 1, "owner_ranks": [0]}],
        local_world_size=1,
    ) == entry((0, 1))
    # Two ranks per node, which is what the recorded shape does not cover: the
    # pair the reverse chose is ranks 1 and 3, not the adjacent 1 and 2.
    assert _MODULE._split_parameter_ownership(
        [entry((1, 3)), {"parameter_index": 1, "owner_ranks": [1]}],
        local_world_size=2,
    ) == entry((1, 3))

    for refused, width in (
        # Both owners inside one host, so the reduction never leaves it.
        (
            [entry((0, 1)), {"parameter_index": 1, "owner_ranks": [0]}],
            2,
        ),
        # Every parameter owner-sharded would mean no wire is bound twice.
        (
            [
                entry((0, 1)),
                {"parameter_index": 1, "owner_ranks": [0, 1], "occurrence_count": 2},
            ],
            1,
        ),
        # No sharded parameter at all, so the collectives shed a partial.
        ([{"parameter_index": 0, "owner_ranks": [0], "occurrence_count": 2}], 1),
    ):
        with pytest.raises(RuntimeError):
            _MODULE._split_parameter_ownership(refused, local_world_size=width)


def test_the_measured_objective_is_not_degenerate() -> None:
    """A constant expectation would let a broken reduction agree with itself."""

    expectation, gradients = _MODULE._reference_expectation()

    assert 0.0 < abs(expectation) < 1.0
    assert all(value != 0.0 for value in gradients)
    # One measured wire on each side of the ownership boundary, which is what
    # makes the expectation a contraction neither rank could do alone.
    first_wire_rank_one_owns = _MODULE.BOUNDARY_PAIR[1]
    assert _MODULE.OBSERVABLE_WIRES[0] < first_wire_rank_one_owns
    assert _MODULE.OBSERVABLE_WIRES[1] >= first_wire_rank_one_owns


def test_checked_in_a800_multinode_mps_evidence_is_narrow_and_self_consistent() -> None:
    payload = _artifact()
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
        "max_bond": 64,
        "n_wires": 6,
        "node_count": 2,
        "observable_wires": [2, 5],
        # Half the sites each, contiguous, so the boundary gate between wires 2
        # and 3 is the one the exchange exists for.
        "site_ownership": [[0, 1, 2], [3, 4, 5]],
        "world_size": 2,
    }
    # Nothing here is a scale claim, and the artifact says so in the summary as
    # well as in the blockers.
    assert evidence["scalability_claim_allowed"] is False
    assert evidence["release_gate_allowed"] is False

    observations = evidence["observations"]
    metrics = observations["numerical_metrics"]
    # Complex128 against the exact statevector reference, so these are round-off
    # rather than a modelling allowance, and they hold for the gradient, the
    # expectation and the training trajectory as well as for the state.
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
    assert all(item["node_count"] == 2 for item in ranks)
    assert all(item["local_world_size"] == 1 for item in ranks)
    assert all(
        item["distribution_semantics"] == "sharded_across_ranks" for item in ranks
    )
    # Site ownership is the sharding: three owned sites each, disjoint, and the
    # gate between wire 2 and wire 3 is the one that has to cross the boundary.
    assert sorted(wire for item in ranks for wire in item["owned_sites"]) == list(
        range(_MODULE.N_WIRES)
    )
    assert all(item["boundary_gate_count"] >= 1 for item in ranks)
    assert all(item["local_gate_count"] >= 1 for item in ranks)

    # The forward exchange happened, and it happened between the nodes: one
    # rank per node means every boundary message is an inter-node message. The
    # workload totals are the ones in the numerical metrics; the figures inside
    # a rank record are that rank's own contribution, which is a different
    # thing, so the two are checked against each other rather than both being
    # read as the workload's. The convention matters: a rank can observe a gate
    # spanning two other ranks' sites and take no part in it, so a workload
    # total that was one rank's count would understate the exchange.
    communicate = [item["communication"] for item in ranks]
    assert all(item["forward_boundary_messages"] >= 1 for item in communicate)
    assert all(item["forward_boundary_bytes"] > 0 for item in communicate)
    assert metrics["forward_boundary_messages"] == sum(
        item["forward_boundary_messages"] for item in communicate
    )
    assert metrics["forward_boundary_bytes"] == sum(
        item["forward_boundary_bytes"] for item in communicate
    )
    communication = communicate[0]
    # The reverse prefetches the layer-boundary halo rather than reconstructing
    # it, and the assembler attributes those bytes to the inter-node tier.
    assert communication["backward_layer_halo_inter_node_bytes"] > 0
    assert communication["backward_layer_halo_intra_node_bytes"] == 0
    # The parameter bound on both rank-owned halves is what the reduction is
    # for; without a collective the gradients would be per-rank partials.
    assert communication["backward_gradient_collective_count"] >= 1
    assert communication["backward_gradient_collective_bytes"] > 0

    # The backward and the training legs carry their own placement, because a
    # forward result that resolved two nodes does not place the other two.
    assert all(item["backward"]["node_count"] == 2 for item in ranks)
    assert all(item["backward"]["local_world_size"] == 1 for item in ranks)
    assert all(
        item["backward"]["distribution_semantics"] == "sharded_across_ranks"
        for item in ranks
    )
    assert all(item["backward"]["scalability_claim_allowed"] is False for item in ranks)
    # An exact reverse on an exact MPS is not a replicated autograd, and the
    # artifact is only evidence of sharding if it says so.
    assert all(
        item["backward"]["mps_backward_execution"] == "completed" for item in ranks
    )
    assert all(item["backward"]["gradient_accuracy"] == "exact" for item in ranks)
    assert all(item["backward"]["replicated_autograd"] is False for item in ranks)
    assert all(item["backward"]["full_mps_reconstruction"] is False for item in ranks)
    assert all(item["backward"]["statevector_fallback"] is False for item in ranks)
    # A reverse that ran its checkpoints eagerly is still a reverse; only the
    # halo prefetch is compiled-only, which is why the tier is asserted on the
    # communication block rather than here.
    assert all(item["backward"]["blockers"] == [] for item in ranks)
    # One parameter is owned by two ranks and the rest by exactly one, which is
    # what makes the reduction an owner-sharded one. The two owners are on
    # different hosts, so the reduction it needs leaves a node; which ranks they
    # are follows from where the sharding put the wires the parameter is bound
    # on, so the pair is derived here rather than written down.
    ownership = sorted(
        (item["parameter_index"], tuple(item["owner_ranks"]))
        for item in next(iter(ranks))["backward"]["parameter_ownership"]
    )
    local_world_size = ranks[0]["local_world_size"]
    spanning = [owners for _, owners in ownership if len(owners) == 2]
    assert len(spanning) == 1
    assert len({owner // local_world_size for owner in spanning[0]}) == 2
    assert all(len(owners) == 1 for _, owners in ownership if len(owners) != 2)

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
    # A checkpoint per rank, on a filesystem both nodes mounted, and the resumed
    # leg continuing from the second rank's shard as well as the first's.
    assert len(training["checkpoint_files"]) >= 2
    assert training["optimizer_state_ownership_semantics"] == "sharded_across_ranks"

    assert "production_performance_not_measured" in evidence["claim_blockers"]
    # The forward result is the one shard of this run that legitimately cannot
    # support a training claim, and it says so as a scope rather than as work
    # that is still missing. A reader must not find the forward pass and the
    # completed reverse contradicting each other inside one rank record.
    assert all(
        item["blockers"]
        == [
            "forward_only_result_excludes_backward_and_optimizer",
            "accelerator_capacity_acceptance_pending",
        ]
        for item in ranks
    )
    # The blockers this probe exists to remove must be gone: an artifact that
    # still carried them would not support the training claim it makes.
    for resolved in (
        "distributed_gradient_not_tested",
        "checkpoint_restart_not_tested",
        "mps_sharded_backward_pending",
        "mps_sharded_optimizer_pending",
        "full_mps_reconstruction_required",
    ):
        assert resolved not in json.dumps(evidence)


def _artifact_kwargs(**overrides: object) -> dict:
    """The shape `_artifact` needs, with every observation retracted by default."""

    kwargs: dict = {
        "rank_records": [],
        "metrics": {},
        "training": {},
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
    # The pair, the circuit and the single inter-node site boundary are declared
    # boundaries. The site plan derives ownership from the world size alone, so
    # this lane cannot move its cut and the blocker has to stay.
    assert complete == {
        "two_node_pair_only_no_wider_topology",
        "toy_circuit_parameters_only",
        "inter_node_cut_width_not_swept",
        "validation_only_tiny_full_mps_gather",
    }

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
        {
            "profiled": True,
            "profiled_workload": _MODULE.MEASUREMENT,
            "host_transfer_observed": True,
            "host_transfer_events": [
                {"name": "memcpy_DtoH", "direction": "device_to_host"}
            ],
        },
    ):
        assert "hidden_host_staging_not_audited" in blockers(host_staging=staging)
