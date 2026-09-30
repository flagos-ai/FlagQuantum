"""Contract tests for the two-node CUDA tensor-network probe."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _ROOT / "tools" / "probe_cuda_multinode_tn.py"
_SPEC = importlib.util.spec_from_file_location("cuda_multinode_tn_probe", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_A800_ARTIFACT = (
    _ROOT / "artifacts" / "cuda_multinode_tn_a800_jp171_jp172_20260930.json"
)


def _artifact() -> dict:
    return json.loads(_A800_ARTIFACT.read_text(encoding="utf-8"))


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
        # Eight ranks would give the four declared slices out unevenly.
        (8, 4),
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
        "claim_evidence_type": "production_runtime",
    }

    _MODULE._require_two_node_placement(good, leg="test", expected_world_size=2)
    # The same summary is a valid four-rank leg, because the probe accepts any
    # width and the check is against the shape that was launched rather than
    # against a constant baked into the checker.
    _MODULE._require_two_node_placement(
        {
            **good,
            "world_size": 4,
            "local_world_size": 2,
            "node_count": 2,
        },
        leg="test",
        expected_world_size=4,
    )

    for wrong in (
        # Two ranks on one node is a different claim that this probe does not
        # make, however the ranks are counted.
        {**good, "local_world_size": 2, "node_count": 1},
        # A single rank would validate the arithmetic and nothing else.
        {**good, "world_size": 1, "node_count": 1},
        # A leg that resolved fewer or more ranks than were launched means the
        # declared placement never reached it.
        {**good, "world_size": 4, "local_world_size": 2},
        # A bespoke semantics string would fail the shared evidence contract.
        {**good, "distribution_semantics": "local_simulated_slice_parallel"},
        # A CPU collective reduces correctly but is not accelerator transport,
        # so an artifact that recorded one would not be accelerator evidence.
        {**good, "claim_evidence_type": "development_smoke"},
    ):
        with pytest.raises(RuntimeError):
            _MODULE._require_two_node_placement(
                wrong, leg="test", expected_world_size=2
            )


def test_the_slice_plan_is_the_declared_partition_and_not_a_scale_claim() -> None:
    every_rank_owns = _MODULE.EXPECTED_SLICE_COUNT // 2
    summary = {
        "slice_tasks": _MODULE.EXPECTED_SLICE_COUNT,
        "slice_labels": list(_MODULE.SLICED_LABELS),
        "tasks_by_rank": {0: every_rank_owns, 1: every_rank_owns},
        "scalability_claim_allowed": False,
    }

    _MODULE._require_slice_partition(summary, leg="test", expected_world_size=2)
    # The same four slices are a valid four-rank plan at one slice each, so the
    # expected share is derived from the launched width rather than fixed.
    _MODULE._require_slice_partition(
        {**summary, "tasks_by_rank": dict.fromkeys(range(4), 1)},
        leg="test",
        expected_world_size=4,
    )

    for wrong in (
        # One slice per rank would mean each rank is a whole contraction, which
        # is the replicated-execution shape the reduction exists to replace.
        {**summary, "slice_tasks": 2},
        # A different cut is a different partition; the artifact declares this
        # one, so a run that took another path is not the recorded evidence.
        {**summary, "slice_labels": [1, 2]},
        # Uneven ownership is a legitimate plan, but not this probe's scope.
        {**summary, "tasks_by_rank": {0: 3, 1: 1}},
        # A rank missing from the plan owns no slice and contributes nothing.
        {**summary, "tasks_by_rank": {0: 2, 1: 2, 2: 0, 3: 0}},
        # The probe must never let a two-node pair look like a scale result.
        {**summary, "scalability_claim_allowed": True},
    ):
        with pytest.raises(RuntimeError):
            _MODULE._require_slice_partition(wrong, leg="test", expected_world_size=2)


def test_the_sliced_labels_are_a_cut_carried_by_two_nodes() -> None:
    multiplicities = _MODULE._require_sliced_labels_are_a_cut()

    # Each sliced label has to appear on two nodes, or slicing it splits nothing.
    assert multiplicities == dict.fromkeys(_MODULE.SLICED_LABELS, 2)


def test_the_measured_objective_is_not_degenerate() -> None:
    """A constant expectation would let a broken reduction agree with itself."""

    expectation, gradients = _MODULE._reference_expectation()

    assert 0.0 < abs(expectation) < 1.0
    # Both trainable parameters have to move the objective, or a gradient
    # reduction could drop one rank and still match the reference.
    assert all(value != 0.0 for value in gradients)
    # The observable spans the two wires the entangling gate joins, so the
    # contraction crosses the slice boundary rather than sitting inside a slice.
    assert _MODULE.OBSERVABLE_WIRES[0] == 0
    assert _MODULE.OBSERVABLE_WIRES[1] == _MODULE.N_WIRES - 1


def test_the_checked_bitstrings_are_distinct_and_non_zero() -> None:
    reference = _MODULE._reference_statevector()

    amplitudes = [
        reference[int(bitstring, 2)] for bitstring in _MODULE.CHECKED_BITSTRINGS
    ]
    assert all(value != 0 for value in amplitudes)
    # Four distinct values, so a reduction that returned one rank's slice for
    # every bitstring could not agree with the reference by coincidence.
    assert len({complex(value) for value in amplitudes}) == len(amplitudes)


def test_checked_in_a800_multinode_tn_evidence_is_narrow_and_self_consistent() -> None:
    payload = _artifact()
    evidence = payload["evidence"]
    encoded = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )

    assert payload["evidence_sha256"] == hashlib.sha256(encoded).hexdigest()
    assert evidence["status"] == "passed"
    assert evidence["scope"] == {
        "checked_bitstrings": ["00000", "10000", "00001", "10001"],
        "distribution_semantics": "sharded_across_ranks",
        "dtype": "complex128",
        "execution": "amplitudes_gradient_optimizer_and_checkpoint_resume",
        "local_world_size": 1,
        "n_wires": 5,
        "node_count": 2,
        "observable_wires": [0, 4],
        # The cut is declared rather than chosen at run time, because the
        # automatic slicer optimizes peak memory and can pick a label whose
        # partials are zero on one rank -- a partition that reports sharding
        # while one rank does the arithmetic.
        "slice_count": 4,
        "sliced_labels": [6, 7],
        "sliced_label_multiplicities": {"6": 2, "7": 2},
        "slices_per_rank": 2,
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
    # amplitudes and the training trajectory as well as for the expectation.
    for name in (
        "amplitude_batch_max_abs_error",
        "single_amplitude_max_abs_error",
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
    # A sliced contraction is not a reconstructed one: the whole point of the
    # probe is the partition, so the full-state flag has to be false.
    assert observations["production_full_state_materialization"] is False
    # The returned expectation is all-reduced, while the global gradient is the
    # sum of per-rank partials the probe gathered itself. An artifact that
    # confused the two would be claiming a gradient reduction the runtime did
    # not perform on this path.
    assert observations["gradient_reduction"] == "probe_summed_rank_partials"

    ranks = observations["rank_records"]
    assert len(ranks) == 2
    placement = [item["rank_placement"] for item in ranks]
    assert sorted(item["rank"] for item in placement) == [0, 1]
    assert sorted(item["node_rank"] for item in placement) == [0, 1]
    assert all(item["world_size"] == 2 for item in placement)
    assert all(item["local_world_size"] == 1 for item in placement)
    assert all(item["node_count"] == 2 for item in placement)
    # Two hosts and two distinct devices: a pair that resolved one host or one
    # device would be a single-node result wearing a two-node label. The
    # placement names a host by digest, so the evidence stays publishable, and a
    # record repeats the placement's digest rather than computing a second one
    # that could disagree with it.
    assert len({item["hostname_sha256"] for item in ranks}) == 2
    assert [item["rank_placement"]["hostname_sha256"] for item in ranks] == [
        item["hostname_sha256"] for item in ranks
    ]
    assert len({item["device_uuid"] for item in ranks}) == 2
    assert all(item["device_name"] == "NVIDIA A800-SXM4-80GB" for item in ranks)
    assert all(
        item["distribution_semantics"] == "sharded_across_ranks" for item in ranks
    )
    # A CPU collective would reduce correctly and prove nothing about the
    # accelerator transport this artifact is evidence for.
    assert all(item["claim_evidence_type"] == "production_runtime" for item in ranks)
    assert all(item["full_state_materialized"] is False for item in ranks)

    # The partition is the sharding: four slices over the two declared labels,
    # two per rank, and no rank owning the whole contraction.
    for item in ranks:
        assert item["slice_labels"] == [6, 7]
        assert item["slice_tasks"] == 4
        assert item["tasks_by_rank"] == {"0": 2, "1": 2}
        assert item["amplitudes"]["slice_labels"] == [6, 7]
        assert item["amplitudes"]["tasks_by_rank"] == {"0": 2, "1": 2}
        assert item["single_amplitude"]["slice_tasks"] == 4
        # The single-amplitude leg takes the other output path and must reach
        # the same partition.
        assert item["single_amplitude"]["full_state_materialized"] is False

    # Each rank's slices have to have produced something. A rank whose partials
    # were all zero would leave the reduced numbers correct while contributing
    # nothing, which is the replicated-execution shape this probe rules out.
    for item in ranks:
        assert all(value != 0.0 for value in item["rank_gradient_contribution"])
    # Both ranks agree on the all-reduced expectation and on the summed
    # gradient, and neither rank's own contribution is that gradient.
    assert len({item["distributed_expectation"] for item in ranks}) == 1
    assert len({tuple(item["reduced_gradient"]) for item in ranks}) == 1
    assert all(
        tuple(item["reduced_gradient"]) != tuple(item["rank_gradient_contribution"])
        for item in ranks
    )
    # The reduced gradient is the sum of the rank partials, and the reference is
    # the single-device adjoint; they agree to round-off, which
    # `gradient_max_abs_error` bounds, so a byte comparison would be asserting
    # the addition order as well as the arithmetic.
    assert all(
        all(
            abs(reduced - reference) <= 1e-10
            for reduced, reference in zip(
                item["reduced_gradient"], item["reference_gradient"], strict=True
            )
        )
        for item in ranks
    )
    assert len({tuple(item["rank_gradient_contribution"]) for item in ranks}) == 2

    # The expectation the distributed leg measured has to be the objective the
    # training legs optimized, and both have to be the exact reference.
    reference_expectation, _ = _MODULE._reference_expectation()
    assert abs(
        next(iter(ranks))["distributed_expectation"] - reference_expectation
    ) <= (1e-10)
    assert abs(
        observations["training"]["first_leg_losses"][0] - reference_expectation
    ) <= (1e-10)

    # One rank per node means every collective byte is an inter-node byte, so a
    # non-zero intra-node tier would contradict the placement.
    for item in ranks:
        communication = item["communication"]
        assert communication["amplitudes_collective_bytes"] > 0
        assert communication["amplitudes_inter_node_collective_bytes"] > 0
        assert communication["amplitudes_intra_node_collective_bytes"] == 0
        assert communication["expectation_inter_node_collective_bytes"] > 0
        assert all(
            value > 0
            for value in communication[
                "expectation_rank_partial_bytes_by_rank"
            ].values()
        )
        assert communication["training_communication_bytes"] > 0
        assert (
            communication["training_communication_tiers"]["intra_node_collective_bytes"]
            == 0
        )
        # Rank-owned parameter state is the optimizer sharding: both ranks own
        # part of it, so neither is a bystander.
        owned = communication["training_owned_parameter_bytes_by_rank"]
        assert len(owned) == 2
        assert all(value > 0 for value in owned)

    # The optimizer owns one parameter per rank, so the update is a rank-owned
    # one rather than the same update replayed everywhere.
    for item in ranks:
        update_ownership = item["training_step"]["optimizer"][
            "optimizer_update_ownership"
        ]
        assert sorted(
            index for entry in update_ownership for index in entry["parameter_indices"]
        ) == [0, 1]
        assert sorted(entry["rank"] for entry in update_ownership) == [0, 1]
        assert item["training_step"]["optimizer"]["optimizer_state_distribution"] == (
            "rank_owned"
        )
        # Both ranks contract their own slices and reverse their own slices.
        assert item["training_step"]["local_task_count"] == 2
        assert item["training_step"]["local_forward_operation_count"] > 0
        assert item["training_step"]["local_reverse_operation_count"] > 0

    training = observations["training"]
    assert training["resumed_start_step"] == training["checkpoint_steps"] == 2
    assert training["uninterrupted_start_step"] == 0
    assert training["resumed_completed_steps"] == training["resumed_steps"] == 4
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
    assert sorted(training["parameter_owner_ranks"]) == [0, 1]
    assert training["tasks_by_rank"] == {"0": 2, "1": 2}
    # One checkpoint write per step of each leg, and the resumed leg rewrites
    # exactly the steps it ran rather than the whole trajectory.
    assert training["interrupted_leg_checkpoint_writes"] == 2
    assert training["uninterrupted_leg_checkpoint_writes"] == 4
    assert training["resumed_leg_checkpoint_writes"] == 2
    assert training["artifact_rank_checkpoint_names"] == ["rank-0-tn-training.pt"]
    assert training["checkpoint_storage_semantics"] == "rank_local_shared_filesystem"

    # The committed generation is read back from the shared directory, so it
    # covers both ranks: one integrity-checked file each, stamped with the last
    # step of the run and belonging to the plan the run reported.
    generation = training["checkpoint_generation"]
    assert generation["committed_step"] == 4
    assert (
        generation["task_plan_identity"]
        == "b91894bb4b7e4165431f11f55ea9870fe9a296ac7cc693ec838fddda160f2dd7"
    )
    assert sorted(generation["shards"]) == ["0", "1"]
    assert generation["rank_local_files_hold_replicated_parameters"] is True
    assert {shard["file"] for shard in generation["shards"].values()} == {
        "rank-0-tn-training.pt",
        "rank-1-tn-training.pt",
    }
    for shard in generation["shards"].values():
        assert shard["step"] == 4
        assert shard["schema_version"] == "flagquantum.tn_training_checkpoint.v1"
        assert shard["world_size"] == 2
        assert shard["local_world_size"] == 1
        assert shard["node_count"] == 2

    assert "production_performance_not_measured" in evidence["claim_blockers"]
    # The blockers this probe exists to remove must be gone: an artifact that
    # still carried them would not support the training claim it makes.
    for resolved in (
        "distributed_gradient_not_tested",
        "checkpoint_restart_not_tested",
        "tensor_network_distributed_execution_pending",
        "tensor_network_sharded_optimizer_pending",
    ):
        assert resolved not in json.dumps(evidence)
