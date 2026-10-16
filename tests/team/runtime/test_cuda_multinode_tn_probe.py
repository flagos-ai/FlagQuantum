"""Contract tests for the two-node CUDA tensor-network probe."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from tools.evidence_provenance import ProvenanceUnavailableError

pytestmark = pytest.mark.unit
_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _ROOT / "tools" / "probe_cuda_multinode_tn.py"
_SPEC = importlib.util.spec_from_file_location("cuda_multinode_tn_probe", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

#: The parameterization the released contract declares, read through the probe's
#: own observation of it rather than restated here. Loading it also runs the
#: probe's agreement check, so a probe constant that drifted from the committed
#: manifest fails at import rather than only in the artifacts it writes.
_ARTIFACT_FROZEN_CIRCUIT = _MODULE._frozen_circuit_observation()

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


def test_checked_in_a800_multinode_tn_evidence_is_narrow_and_self_consistent(
    manifest_digest_at: Callable[[str, str], str],
) -> None:
    payload = _artifact()
    evidence = payload["evidence"]
    encoded = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )

    assert payload["evidence_sha256"] == hashlib.sha256(encoded).hexdigest()
    assert evidence["status"] == "passed"
    assert evidence["scope"] == {
        "checked_bitstrings": list(_MODULE.CHECKED_BITSTRINGS),
        "depth": _MODULE.DEPTH,
        "distribution_semantics": "sharded_across_ranks",
        "dtype": "complex128",
        "execution": "amplitudes_gradient_optimizer_and_checkpoint_resume",
        # The product the export leg materializes. It is named in the scope so a
        # reader can see which object the leg exported rather than having to
        # infer it from the observation block.
        "exported_product": "full_tensor_network_state",
        "local_world_size": 1,
        "n_wires": _MODULE.N_WIRES,
        "node_count": 2,
        "observable_wires": list(_MODULE.OBSERVABLE_WIRES),
        "parameter_count": _MODULE.PARAMETER_COUNT,
        "release_configuration": _MODULE.RELEASE_CONFIGURATION,
        # The cut is declared rather than chosen at run time, because the
        # automatic slicer optimizes peak memory and can pick a label whose
        # partials are zero on one rank -- a partition that reports sharding
        # while one rank does the arithmetic.
        "slice_count": _MODULE.EXPECTED_SLICE_COUNT,
        "sliced_labels": list(_MODULE.SLICED_LABELS),
        "sliced_label_multiplicities": {
            str(label): count
            for label, count in _MODULE._sliced_label_multiplicities().items()
        },
        "slices_per_rank": 2,
        "world_size": 2,
    }
    # Nothing here is a scale claim, and the artifact says so in the summary as
    # well as in the blockers.
    assert evidence["scalability_claim_allowed"] is False
    assert evidence["release_gate_allowed"] is False

    # The measured workload is the release contract's own narrowest matched-speed
    # rung rather than a hand-written circuit, and the artifact says so by naming
    # the frozen file and the digest of the bytes it read. The declared and bound
    # leaf counts have to agree, because the point of the observation is that the
    # parameterization is a measured fact rather than a declaration beside one.
    frozen = evidence["observations"]["frozen_circuit"]
    assert frozen["configuration"] == _MODULE.RELEASE_CONFIGURATION
    assert frozen["manifest"] == "benchmarks/manifests/tensor_network_release_v1.json"
    assert frozen["n_wires"] == _MODULE.N_WIRES == 14
    assert frozen["depth"] == _MODULE.DEPTH == 6
    assert frozen["declared_parameter_count"] == _MODULE.PARAMETER_COUNT == 84
    assert frozen["bound_parameter_count"] == frozen["declared_parameter_count"]
    # Every field but the digest, which is scoped to the revision the probe
    # ran at and is resolved below, still has to be what the probe observes
    # from the contract checked out here.
    assert {
        key: value for key, value in frozen.items() if key != "manifest_sha256"
    } == {
        key: value
        for key, value in _ARTIFACT_FROZEN_CIRCUIT.items()
        if key != "manifest_sha256"
    }
    # The recorded digest is the contract as of the revision the probe ran at,
    # which is the file it actually read, rather than whatever happens to be
    # checked out beside it now. The current contract is checked separately:
    # loading this module runs the probe's own observation of it, which
    # cross-checks the frozen ladder against the ladder file it names. Answering
    # it needs that commit, so a lane whose clone never fetched the revision
    # skips this one assertion instead of reading the clone as a mismatch; the
    # full-history quality job is where it reaches a verdict.
    try:
        recorded_digest = manifest_digest_at(
            evidence["environment"]["source_revision"], frozen["manifest"]
        )
    except ProvenanceUnavailableError as error:
        pytest.skip(f"the recorded contract digest needs full history: {error}")
    assert frozen["manifest_sha256"] == recorded_digest

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
    # A sliced contraction is not a reconstructed one: the amplitude legs and
    # the single-amplitude projection report their own summaries, and neither
    # materializes the whole state. The full-state flag is true in this artifact
    # because the export leg is a separate leg that does, which is exactly the
    # distinction the gather blocker turns on.
    for rank_record in observations["rank_records"]:
        assert rank_record["amplitudes"]["full_state_materialized"] is False
        assert rank_record["single_amplitude"]["full_state_materialized"] is False
    assert observations["production_full_state_materialization"] is True
    assert observations["export"]["product"] == _MODULE.EXPORTED_PRODUCT
    # The returned expectation is all-reduced, while the global gradient is the
    # sum of per-rank partials the probe gathered itself. An artifact that
    # confused the two would be claiming a gradient reduction the runtime did
    # not perform on this path.
    assert observations["gradient_reduction"] == "probe_summed_rank_partials"

    # The route is what the debug log recorded, on the interface the planner was
    # told to use, and it is the fabric rather than the socket fallback. A
    # socket route here would mean the pair that produced this artifact never
    # exercised the transport the run is evidence about.
    network = observations["network"]
    assert network["evidence_level"] == "observed_debug_log"
    assert network["route"] == "infiniband"
    assert network["configured_transport"] == "infiniband"
    assert network["configured_interface"] == "ens22f0"
    assert network["configured_interface_observed"] is True
    assert network["infiniband_transport_observed"] is True
    assert network["roce_transport_observed"] is True
    assert network["gpu_direct_observed"] is True
    assert network["socket_transport_observed"] is False
    assert network["backend"] == "nccl"

    # The cut sweep moved, and the exchange moved with it, which is the whole
    # difference between a swept cut and a cut that happens to be declared. Each
    # width is listed with the bytes its ranks exchanged and the ranks that
    # participated, so a width nobody exercised cannot hide in the summary.
    cut = observations["cut_widths"]
    assert cut["widths"] == [1, 2]
    assert cut["unexercised_widths"] == []
    assert set(cut["inter_node_bytes_by_width"]) == {"1", "2"}
    assert all(bytes_ > 0 for bytes_ in cut["inter_node_bytes_by_width"].values())
    assert sorted(cut["inter_node_bytes_by_slice_count"]) == ["2", "4"]
    assert all(bytes_ > 0 for bytes_ in cut["inter_node_bytes_by_slice_count"].values())
    assert all(ranks == [0, 1] for ranks in cut["ranks_by_width"].values())

    # The measured leg, and the two properties that make it a measurement: the
    # clock is synchronized across ranks, and the samples were taken after the
    # warmup rather than as the warmup. The summary is recomputed from the
    # samples so a hand-edited median cannot disagree with them.
    performance = observations["performance"]
    assert performance["measurement"] == "sharded_tensor_network_amplitudes"
    assert performance["measured"] is True
    assert performance["synchronized"] is True
    assert performance["warmup_iterations"] >= 1
    assert performance["measured_iterations"] >= 3
    samples = performance["seconds"]
    assert len(samples) == performance["measured_iterations"]
    assert performance["minimum_seconds"] == min(samples)
    assert performance["maximum_seconds"] == max(samples)
    assert performance["minimum_seconds"] <= performance["median_seconds"]
    assert performance["median_seconds"] <= performance["maximum_seconds"]

    # The staging audit ran, and it found the profiled region clean. That is a
    # measurement rather than an absence of measurement: the flag that says
    # nobody looked and the blocker that says transfers were found are both
    # keyed off this observation, so a run that stopped profiling would report
    # the first and a run that regressed would report the second.
    staging = observations["host_staging"]
    assert staging["profiled"] is True
    assert staging["profiled_workload"] == performance["measurement"]
    assert staging["host_transfer_observed"] is False
    assert staging["host_transfer_events"] == []
    assert staging["profiler_event_count"] > 0

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
        assert item["slice_labels"] == list(_MODULE.SLICED_LABELS)
        assert item["slice_tasks"] == 4
        assert item["tasks_by_rank"] == {"0": 2, "1": 2}
        assert item["amplitudes"]["slice_labels"] == list(_MODULE.SLICED_LABELS)
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
    assert training["resumed_leg_losses"] == (
        training["uninterrupted_leg_losses"][training["checkpoint_steps"] :]
    )
    assert training["first_leg_losses"] == (
        training["uninterrupted_leg_losses"][: training["checkpoint_steps"]]
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
    # The plan identity is a digest of the contraction plan, so it moves with the
    # circuit: this is the digest of the released rung's plan rather than of the
    # five-wire circuit the earlier recording carried. It is compared against the
    # plan the interrupted leg committed, not recomputed here, because recomputing
    # it would only check this file against itself.
    assert (
        generation["task_plan_identity"]
        == "52cd77094c1332f2e9f9df29b129305ca4a72d5c708ac762dc4f9ccac8142d75"
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

    # The boundary is exactly these blockers. The fabric, the measured workload,
    # the audited staging path, the swept cut and the full-state gather are each
    # retired by the observation that justifies them, so a reader cannot find one
    # here and cannot find one silently missing either.
    assert sorted(evidence["claim_blockers"]) == [
        "two_node_pair_only_no_wider_topology",
    ]
    # The blockers this probe exists to remove must be gone: an artifact that
    # still carried them would not support the training claim it makes.
    for resolved in (
        "distributed_gradient_not_tested",
        "checkpoint_restart_not_tested",
        "tensor_network_distributed_execution_pending",
        "tensor_network_sharded_optimizer_pending",
    ):
        assert resolved not in json.dumps(evidence)


def _artifact_kwargs(**overrides: object) -> dict:
    """The shape `_artifact` needs, with every observation retracted by default."""

    kwargs: dict = {
        "rank_records": [],
        "metrics": {},
        "training": {},
        "sliced_label_multiplicities": dict.fromkeys(_MODULE.SLICED_LABELS, 2),
        "world_size": 2,
        "local_world_size": 1,
        "frozen_circuit": dict(_ARTIFACT_FROZEN_CIRCUIT),
        "network": {
            "evidence_level": "observed_debug_log",
            "route": "infiniband",
            "configured_interface": "ens22f0",
            "configured_interface_observed": True,
            "infiniband_transport_observed": True,
            "socket_transport_observed": False,
            "roce_transport_observed": True,
            "gpu_direct_observed": True,
            "configured_transport": "infiniband",
        },
        # Five synchronized samples after two warmups: the shape
        # `measured_performance` returns, so the retraction is derived from a
        # measurement rather than from the presence of a key.
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
        "cut_widths": {
            "widths": list(_MODULE.CUT_WIDTHS),
            "inter_node_bytes_by_width": {"1": 32, "2": 64},
            "inter_node_bytes_by_slice_count": {"2": 32, "4": 64},
            "ranks_by_width": {"1": [0, 1], "2": [0, 1]},
            "unexercised_widths": [],
        },
        # No export leg by default, so the helper describes a run whose only
        # whole-state read is the validation comparison and leaves the
        # full-state-gather blocker standing. A test that wants it retracted
        # passes an export of its own.
        "export": None,
    }
    kwargs.update(overrides)
    return kwargs


def test_the_cut_width_sweep_derives_both_topology_blockers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cut that moved and stayed cut is the only thing that clears them."""

    monkeypatch.setattr(_MODULE.torch.cuda.nccl, "version", lambda: (2, 29, 7))
    evidence = _MODULE._artifact(**_artifact_kwargs())["evidence"]
    blockers = set(evidence["claim_blockers"])

    # The pair is fixed by the lane, so it cannot be retracted by anything this
    # run does and it has to stay. The staging blocker is retracted by the same
    # default shape, because the profiled region this helper describes is one
    # that leaked nothing across the host boundary; the full-state-gather blocker
    # stays because this helper describes a run whose only whole-state read is
    # the validation comparison. The circuit is not declared: the default shape
    # runs the release contract's own rung, so that blocker is retracted too.
    assert sorted(blockers) == [
        "two_node_pair_only_no_wider_topology",
        "validation_only_tiny_full_state_gather",
    ]
    assert "inter_node_cut_width_not_swept" not in blockers
    assert "slice_count_fixed_at_world_size" not in blockers
    assert "rdma_not_tested" not in blockers
    assert "production_performance_not_measured" not in blockers
    assert "hidden_host_staging_not_audited" not in blockers
    assert "host_staging_in_measured_region" not in blockers

    # One width, or a width whose ranks exchanged nothing, is not a sweep: the
    # whole point is that the cut moved and the exchange moved with it.
    for degenerate in (
        {"1": 32},
        {"1": 0, "2": 0},
        {"1": 32, "2": 0},
    ):
        swept = _MODULE._artifact(
            **_artifact_kwargs(
                cut_widths={
                    "widths": list(_MODULE.CUT_WIDTHS),
                    "inter_node_bytes_by_width": degenerate,
                    "inter_node_bytes_by_slice_count": {"2": 32, "4": 64},
                    "unexercised_widths": [],
                }
            )
        )["evidence"]
        assert "inter_node_cut_width_not_swept" in swept["claim_blockers"]

    # The slice count carries the same condition: two counts, both crossed.
    for degenerate_counts in ({"2": 32}, {"2": 32, "4": 0}):
        swept = _MODULE._artifact(
            **_artifact_kwargs(
                cut_widths={
                    "widths": list(_MODULE.CUT_WIDTHS),
                    "inter_node_bytes_by_width": {"1": 32, "2": 64},
                    "inter_node_bytes_by_slice_count": degenerate_counts,
                    "unexercised_widths": [],
                }
            )
        )["evidence"]
        assert "slice_count_fixed_at_world_size" in swept["claim_blockers"]


def test_the_route_measurement_and_staging_are_each_derived(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each retraction needs its own positive observation, not a declaration."""

    monkeypatch.setattr(_MODULE.torch.cuda.nccl, "version", lambda: (2, 29, 7))

    def blockers(**overrides: object) -> set[str]:
        return set(
            _MODULE._artifact(**_artifact_kwargs(**overrides))["evidence"][
                "claim_blockers"
            ]
        )

    # A socket run, or a run with no debug log to read, leaves the fabric
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

    # The toy-circuit blocker turns on the parameterization, so it is retracted
    # only by a parameterization the release contract itself accepts. An absent
    # observation is not a big circuit, and neither is one that names no frozen
    # configuration, names a file without a digest, declares fewer leaves than
    # the project's own smallest released workload, or declares a count the
    # built circuit did not bind.
    for incomplete in (
        None,
        {},
        {**_ARTIFACT_FROZEN_CIRCUIT, "configuration": ""},
        {**_ARTIFACT_FROZEN_CIRCUIT, "manifest_sha256": "unavailable"},
        {**_ARTIFACT_FROZEN_CIRCUIT, "declared_parameter_count": 8},
        {**_ARTIFACT_FROZEN_CIRCUIT, "bound_parameter_count": 83},
        {**_ARTIFACT_FROZEN_CIRCUIT, "declared_parameter_count": 84.5},
    ):
        assert "toy_circuit_parameters_only" in blockers(frozen_circuit=incomplete)
    assert "toy_circuit_parameters_only" not in blockers()

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


def test_the_export_leg_is_the_only_materialization_that_retracts_the_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A state the workload produces, not a shard moved aside to check an answer."""

    monkeypatch.setattr(_MODULE.torch.cuda.nccl, "version", lambda: (2, 29, 7))

    def blockers(**overrides: object) -> set[str]:
        return set(
            _MODULE._artifact(**_artifact_kwargs(**overrides))["evidence"][
                "claim_blockers"
            ]
        )

    # The amplitude legs reduce a handful of projected outputs and never
    # materialize the state, so a run with no export leg leaves the blocker.
    assert "validation_only_tiny_full_state_gather" in blockers()
    assert "validation_only_tiny_full_state_gather" in blockers(export=None)

    # Three separate things have to hold, and each is a real way the leg could
    # fail to be the product the blocker is about: the state has to have been
    # materialized, it has to be the tensor-network state rather than another
    # result shape, and it has to be the leg's product rather than a shard.
    materialized = {
        "result": {
            "full_state_materialized": True,
            "state_mode": "distributed_tensor_network",
        }
    }
    assert "validation_only_tiny_full_state_gather" in blockers(
        export={**materialized, "product": "shard_state"}
    )
    assert "validation_only_tiny_full_state_gather" in blockers(
        export={
            "product": _MODULE.EXPORTED_PRODUCT,
            "result": {
                "full_state_materialized": False,
                "state_mode": "distributed_tensor_network",
            },
        }
    )
    assert "validation_only_tiny_full_state_gather" in blockers(
        export={
            "product": _MODULE.EXPORTED_PRODUCT,
            "result": {
                "full_state_materialized": True,
                "state_mode": "distributed_tensor_network_amplitudes",
            },
        }
    )

    # The complete leg retracts it, and retracts nothing else: the pair is a
    # declared boundary that no leg of this run can move, while the circuit is
    # the release contract's own rung and is therefore not a boundary at all.
    exported = blockers(export={**materialized, "product": _MODULE.EXPORTED_PRODUCT})
    assert "validation_only_tiny_full_state_gather" not in exported
    assert exported == {"two_node_pair_only_no_wider_topology"}


def test_the_export_leg_refuses_a_state_it_did_not_materialize() -> None:
    """The guard reads the object's own distribution, not the probe's intent."""

    placed = {
        "node_count": 2,
        "world_size": 2,
        "local_world_size": 1,
        "claim_evidence_type": "production_runtime",
        "state_distribution_semantics": _MODULE.EXPORTED_STATE_DISTRIBUTION,
        "full_state_materialized": True,
        "scalability_claim_allowed": False,
    }
    _MODULE._require_export_placement(placed, expected_world_size=2)

    # Every field the guard reads is a way the leg could have run somewhere or
    # produced something other than the whole state on the declared pair.
    for broken in (
        {**placed, "node_count": 1},
        {**placed, "world_size": 4},
        {**placed, "claim_evidence_type": "development_smoke"},
        {**placed, "state_distribution_semantics": "sharded_across_ranks"},
        {**placed, "full_state_materialized": False},
        {**placed, "scalability_claim_allowed": True},
    ):
        with pytest.raises(RuntimeError):
            _MODULE._require_export_placement(broken, expected_world_size=2)

    # And the derivation is read from the recorded result rather than from a
    # flag beside it, so a leg whose record disagrees with its own summary is
    # not a materialization.
    assert _MODULE._materializes_the_full_state(None) is False
    assert _MODULE._materializes_the_full_state({"product": "shard_state"}) is False


def test_the_sweep_widths_are_prefixes_of_the_declared_cut() -> None:
    """A narrower cut is the declared one with one label fewer, and nothing else."""

    assert tuple(range(1, len(_MODULE.SLICED_LABELS) + 1)) == _MODULE.CUT_WIDTHS
    assert [_MODULE._slices_for_width(width) for width in _MODULE.CUT_WIDTHS] == [
        2,
        _MODULE.EXPECTED_SLICE_COUNT,
    ]
    # The widest width is the scope's own cut, so the sweep includes the shape
    # the artifact declares rather than only shapes beside it.
    assert _MODULE._slices_for_width(max(_MODULE.CUT_WIDTHS)) == (
        _MODULE.EXPECTED_SLICE_COUNT
    )


@pytest.mark.parametrize("route", ["infiniband", "socket"])
def test_the_route_is_applied_to_the_blockers_after_the_log_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: str
) -> None:
    """A run that used the fabric must not record it as untested.

    The route is the one observation that arrives after the artifact is
    assembled, because it is read from a log the process group is still writing
    while it exists. `_apply_network_observation` is the seam that runs the
    derivation a second time for it; this fails if the second pass is dropped,
    and it fails if the second pass forgets to re-digest the evidence it
    changed.
    """

    monkeypatch.setattr(_MODULE.torch.cuda.nccl, "version", lambda: (2, 29, 7))
    monkeypatch.setenv("NCCL_SOCKET_IFNAME", "ens22f0")
    log = tmp_path / "nccl.log"
    if route == "infiniband":
        log.write_text(
            "NET/IB : Using [0]mlx5_101:1/RoCE [RO]; OOB ens22f0:10.1.15.172<0>\n"
            "via NET/IB/0/GDRDMA\n"
        )
        monkeypatch.setenv("NCCL_IB_DISABLE", "0")
    else:
        log.write_text("NET/Socket : Using [0]ens22f0:10.1.15.172<0>\n")
        monkeypatch.setenv("NCCL_IB_DISABLE", "1")

    payload = _MODULE._artifact(**_artifact_kwargs(network={}))
    assert "rdma_not_tested" in payload["evidence"]["claim_blockers"]

    _MODULE._apply_network_observation(payload, network_log=log, local_world_size=1)
    evidence = payload["evidence"]

    assert evidence["observations"]["network"]["route"] == route
    assert ("rdma_not_tested" in evidence["claim_blockers"]) == (route == "socket")
    # The digest covers the evidence, and the evidence changed.
    encoded = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    assert payload["evidence_sha256"] == hashlib.sha256(encoded).hexdigest()


@pytest.mark.parametrize("payload", [None])
def test_an_absent_artifact_has_no_route_to_record(payload: None) -> None:
    """Only rank zero writes the artifact, and it does so through the same seam."""

    assert (
        _MODULE._apply_network_observation(
            payload, network_log=None, local_world_size=1
        )
        is None
    )
