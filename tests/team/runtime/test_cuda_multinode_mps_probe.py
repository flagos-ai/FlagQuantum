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


def test_the_two_rank_pair_is_the_only_placement_the_probe_accepts() -> None:
    good = {
        "world_size": 2,
        "local_world_size": 1,
        "node_count": 2,
        "distribution_semantics": "sharded_across_ranks",
    }

    _MODULE._require_two_node_placement(good, leg="test")

    for wrong in (
        # One node per rank is the claim; two ranks on one node is a different
        # claim that this probe does not make.
        {**good, "local_world_size": 2, "node_count": 1},
        # A single rank would validate the arithmetic and nothing else.
        {**good, "world_size": 1, "node_count": 1},
        # A bespoke semantics string would fail the shared evidence contract.
        {**good, "distribution_semantics": "site_sharded"},
    ):
        with pytest.raises(RuntimeError):
            _MODULE._require_two_node_placement(wrong, leg="test")


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
    # rank per node means every boundary message is an inter-node message.
    communication = next(iter(ranks))["communication"]
    assert communication["forward_boundary_messages"] >= 1
    assert communication["forward_boundary_bytes"] > 0
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
    # One parameter is owned by both ranks and the rest by exactly one, which is
    # what makes the reduction an owner-sharded one.
    ownership = sorted(
        (item["parameter_index"], tuple(item["owner_ranks"]))
        for item in next(iter(ranks))["backward"]["parameter_ownership"]
    )
    assert [owners for _, owners in ownership].count((0, 1)) == 1
    assert all(len(owners) == 1 for _, owners in ownership if owners != (0, 1))

    training = observations["training"]
    assert training["resumed_start_step"] == training["checkpoint_steps"] == 2
    assert training["uninterrupted_start_step"] == 0
    # The resumed leg computes the steps the uninterrupted run computed after
    # the checkpoint, and nothing else.
    assert (
        training["resumed_leg_losses"]
        == (training["uninterrupted_leg_losses"][training["checkpoint_steps"] :])
    )
    assert (
        training["first_leg_losses"]
        == (training["uninterrupted_leg_losses"][: training["checkpoint_steps"]])
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
