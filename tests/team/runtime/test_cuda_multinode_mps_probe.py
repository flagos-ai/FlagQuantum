"""Contract tests for the two-node CUDA MPS probe."""

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
_SCRIPT = _ROOT / "tools" / "probe_cuda_multinode_mps.py"
_SPEC = importlib.util.spec_from_file_location("cuda_multinode_mps_probe", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

#: The parameterization the released contract declares, read through the probe's
#: own observation of it rather than restated here. Loading it also runs the
#: probe's agreement checks against both the manifest and the contract's own
#: workload builder, so a probe constant that drifted from either fails at
#: import rather than only in the artifacts it writes.
_ARTIFACT_FROZEN_CIRCUIT = _MODULE._frozen_circuit_observation()

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


def test_checked_in_a800_multinode_mps_evidence_is_narrow_and_self_consistent(
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
        "distribution_semantics": "sharded_across_ranks",
        "dtype": "complex128",
        "execution": (
            "forward_backward_optimizer_checkpoint_resume_and_full_state_export"
        ),
        # The export's product is named in the scope rather than left to the
        # observation, so a reader comparing this run with the statevector one
        # sees which object was materialized without reading the leg.
        "exported_product": "full_mps_state",
        "local_world_size": 1,
        "max_bond": 64,
        "n_wires": 6,
        "node_count": 2,
        "observable_wires": [2, 5],
        "parameter_count": _MODULE.PARAMETER_COUNT,
        "release_configuration": _MODULE.RELEASE_CONFIGURATION,
        # Half the sites each, contiguous, so the boundary gate between wires 2
        # and 3 is the one the exchange exists for.
        "site_ownership": [[0, 1, 2], [3, 4, 5]],
        "world_size": 2,
    }
    # Nothing here is a scale claim, and the artifact says so in the summary as
    # well as in the blockers.
    assert evidence["scalability_claim_allowed"] is False
    assert evidence["release_gate_allowed"] is False

    # The measured workload carries the release contract's own parameterization:
    # one independent leaf per gate, thirty-one of them, which is what the
    # contract freezes for the shape it certifies. The artifact says so by naming
    # the frozen file and the digest of the bytes it read, and by recording both
    # the count the contract declares and the count the built circuit bound. The
    # two have to agree, because the point of the observation is that the
    # parameterization is a measured fact rather than a declaration beside one.
    frozen = evidence["observations"]["frozen_circuit"]
    assert frozen["configuration"] == _MODULE.RELEASE_CONFIGURATION
    assert frozen["manifest"] == "benchmarks/manifests/mps_release_v1.json"
    assert frozen["n_sites"] == _MODULE.RELEASE_N_SITES == 8192
    assert frozen["trained_max_bond"] == _MODULE.RELEASE_MAX_BOND == 64
    assert frozen["n_wires"] == _MODULE.N_WIRES == 6
    assert frozen["declared_parameter_count"] == _MODULE.PARAMETER_COUNT == 31
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

    # The boundary is exactly these blockers. The fabric, the measured workload,
    # the audited staging path, the cut width and the production full-state
    # materialization are each retired by the observation that justifies them;
    # the pair alone is declared. The staging blocker stays because the audit
    # found transfers rather than because nobody looked, and it is kept apart
    # from the blocker that says so.
    assert sorted(evidence["claim_blockers"]) == [
        "host_staging_in_measured_region",
        "two_node_pair_only_no_wider_topology",
    ]

    # The cut width is retired by a sweep rather than by a declaration, so the
    # sweep's own numbers are checked here: the boundary moved to more than one
    # rank, and every width that moved carried bytes across the hosts.
    cut_widths = observations["cut_widths"]
    assert len(cut_widths["widths"]) >= 2
    for width in cut_widths["widths"]:
        assert cut_widths["inter_node_bytes_by_width"][str(width)] > 0
    assert len(cut_widths["inter_node_bytes_by_width"]) >= 2
    # Every plan this shape can hold was run, each against the exact reference.
    assert len(cut_widths["plans"]) == _MODULE.N_WIRES - 1
    assert {tuple(plan["cut_positions"]) for plan in cut_widths["plans"]} == {
        (cut,) for cut in range(_MODULE.N_WIRES - 1)
    }
    for plan in cut_widths["plans"]:
        assert plan["worst_error"] <= 1e-10
        # A width is the exact Schmidt rank of the state at that cut, so it
        # cannot exceed what the reference says the state holds there.
        assert plan["width"] <= _MODULE.MAX_BOND
    assert cut_widths["exact_schmidt_ranks"] == [
        2,
        4,
        4,
        4,
        2,
    ]
    # At two ranks a plan carries one boundary, so every height of the profile is
    # placed by some leg and the blocked list carries nothing. The profile is
    # recorded whole beside the widths that were placed, so a height the sweep
    # did not reach would be visible here as a rank the widths do not cover.
    profile_widths = set(cut_widths["exact_schmidt_ranks"])
    assert set(cut_widths["widths"]) == profile_widths

    # The measured leg, and the two properties that make it a measurement: the
    # clock is synchronized across ranks, and the samples were taken after the
    # warmup rather than as the warmup. The medians are recomputed from the
    # samples so a hand-edited summary cannot disagree with the raw numbers it
    # summarizes.
    performance = observations["performance"]
    assert performance["measurement"] == _MODULE.MEASUREMENT
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

    # The staging audit ran, and it found transfers inside the region it
    # profiled. That is a finding about the workload, so it keeps a blocker --
    # but not the one that says nobody looked.
    staging = observations["host_staging"]
    assert staging["profiled"] is True
    assert staging["profiled_workload"] == _MODULE.MEASUREMENT
    assert staging["host_transfer_observed"] is True
    assert staging["host_transfer_events"]
    assert all(event["count"] > 0 for event in staging["host_transfer_events"])
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


def test_the_frozen_rung_is_checked_against_the_ladder_file_it_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rung is a copy, so it is verified against the file it was copied from.

    A check that cannot fail is decoration, so each way the copy could drift
    from the ladder is exercised here: a fingerprint that covers every rung
    shape, an acceptance name, and the shape of the rung this probe reads.
    """

    ladder = _MODULE._contract_module(
        "internal/evidence/general_mps_speed_ladder_v1.py",
        "flagquantum_probe_mps_speed_ladder_for_test",
    )
    real = _MODULE._contract_module

    def with_ladder(replacement: object) -> None:
        def loader(relative_path: str, module_name: str):
            if relative_path.endswith("general_mps_speed_ladder_v1.py"):
                return replacement
            return real(relative_path, module_name)

        monkeypatch.setattr(_MODULE, "_contract_module", loader)

    # Unmodified, it agrees -- otherwise everything below would pass trivially.
    assert _MODULE._frozen_circuit_observation() == _ARTIFACT_FROZEN_CIRCUIT

    class Drifted:
        ACCEPTANCE = ladder.ACCEPTANCE

        @staticmethod
        def ladder_fingerprint() -> str:
            return "0" * 64

        @staticmethod
        def rung(name: str) -> dict[str, object]:
            return ladder.rung(name)

    with_ladder(Drifted())
    with pytest.raises(RuntimeError, match="frozen ladder disagrees"):
        _MODULE._frozen_circuit_observation()

    class Renamed:
        ACCEPTANCE = "matched_speed_16384sites_chi64"

        @staticmethod
        def ladder_fingerprint() -> str:
            return ladder.ladder_fingerprint()

        @staticmethod
        def rung(name: str) -> dict[str, object]:
            return ladder.rung(name)

    with_ladder(Renamed())
    with pytest.raises(RuntimeError, match="publishes its speedup at"):
        _MODULE._frozen_circuit_observation()

    class Moved:
        ACCEPTANCE = ladder.ACCEPTANCE

        @staticmethod
        def ladder_fingerprint() -> str:
            return ladder.ladder_fingerprint()

        @staticmethod
        def rung(name: str) -> dict[str, object]:
            return {**ladder.rung(name), "n_sites": 4096}

    with_ladder(Moved())
    with pytest.raises(RuntimeError, match="declares n_sites=4096"):
        _MODULE._frozen_circuit_observation()


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
            "configured_transport": "infiniband",
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
        "export": {
            "measurement": _MODULE.EXPORT_MEASUREMENT,
            "product": _MODULE.EXPORTED_PRODUCT,
            "result": {
                "full_state_materialization": True,
                "site_order": _MODULE.SITE_ORDER_CANONICAL_LOGICAL,
                "distribution_semantics": "replicated_per_rank",
            },
            "gather": {
                "measured": True,
                "synchronized": True,
                "measurement": _MODULE.EXPORT_MEASUREMENT,
                "warmup_iterations": _MODULE.EXPORT_WARMUP_ITERATIONS,
                "measured_iterations": _MODULE.EXPORT_ITERATIONS,
                "seconds": [0.002, 0.0021, 0.0019, 0.002, 0.0022],
            },
        },
        "cut_widths": {
            "widths": [2, 4],
            "exact_schmidt_ranks": [2, 4, 4, 4, 2],
            "inter_node_bytes_by_width": {"2": 896, "4": 3520},
            "plans": [
                {
                    "plan": [[0, 1], [2, 3, 4, 5]],
                    "cut_positions": [1],
                    "width": 4,
                    "inter_node_bytes": 1344,
                    "inter_node_messages": 6,
                    "worst_error": 3.9e-16,
                },
                {
                    "plan": [[0, 1, 2, 3, 4], [5]],
                    "cut_positions": [4],
                    "width": 2,
                    "inter_node_bytes": 320,
                    "inter_node_messages": 6,
                    "worst_error": 3.9e-16,
                },
            ],
        },
        "world_size": 2,
        "local_world_size": 1,
        "frozen_circuit": dict(_ARTIFACT_FROZEN_CIRCUIT),
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
    # The pair of hosts is a declared boundary, so nothing this run does can
    # retract it. The circuit used to be declared the same way; it is now a
    # measured fact, because the run records the configuration it bound and the
    # leaf count the built circuit reported. The cut width is not declared
    # either: the sweep moved the site boundary and read back what each placement
    # carried, so the blocker that says no cut ever moved is gone. The full-state
    # export is not either: the leg produces the whole state rather than checking
    # an answer, so the blocker that says the only full MPS was a validation
    # gather is gone too.
    assert "two_node_pair_only_no_wider_topology" in complete
    assert complete - {"two_node_pair_only_no_wider_topology"} == set()

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
        {**_ARTIFACT_FROZEN_CIRCUIT, "bound_parameter_count": 30},
        {**_ARTIFACT_FROZEN_CIRCUIT, "declared_parameter_count": 31.5},
    ):
        assert "toy_circuit_parameters_only" in blockers(frozen_circuit=incomplete)
    assert "toy_circuit_parameters_only" not in complete

    # The full-MPS gather blocker turns on whether a materialization the workload
    # needs was recorded. A validation gather is not one, so an absent export
    # leaves the blocker; so does an export that did not materialize the whole
    # state, one published in some other site order, and one whose product was a
    # shard rather than the state. Each of those is a real way the leg could fail
    # to be the product the blocker is about.
    full_state = {
        "result": {
            "full_state_materialization": True,
            "site_order": _MODULE.SITE_ORDER_CANONICAL_LOGICAL,
        }
    }
    assert "validation_only_tiny_full_mps_gather" in blockers(export=None)
    assert "validation_only_tiny_full_mps_gather" in blockers(
        export={**full_state, "product": "shard_state"}
    )
    assert "validation_only_tiny_full_mps_gather" in blockers(
        export={
            "product": _MODULE.EXPORTED_PRODUCT,
            "result": {
                "full_state_materialization": False,
                "site_order": _MODULE.SITE_ORDER_CANONICAL_LOGICAL,
            },
        }
    )
    assert "validation_only_tiny_full_mps_gather" in blockers(
        export={
            "product": _MODULE.EXPORTED_PRODUCT,
            "result": {
                "full_state_materialization": True,
                "site_order": "internal_site_order_requires_permutation",
            },
        }
    )
    assert "validation_only_tiny_full_mps_gather" not in complete

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


def test_a_wider_shape_is_not_credited_with_a_cut_it_never_carried() -> None:
    """The recorded sweep is honest about the shapes it cannot answer for.

    A plan at two ranks carries one boundary, so every height of the profile is
    placed and the blocker that says no cut ever moved is the one the sweep
    retires. A wider shape carries several boundaries per plan and names each leg
    by the heaviest of them, so a lighter height is never placed -- and what the
    run then reports is the one width it did carry, which the blocker's own rule
    refuses to count as a sweep. Naming the leg after a lighter boundary instead
    would retract the blocker on a bond no leg crossed.
    """

    profile = (2, 4, 4, 4, 2)

    def placed(world_size: int) -> dict[str, int]:
        return {
            str(
                _MODULE._plan_width(plan, profile)
            ): 128  # one width per plan at this shape
            for plan in _MODULE._ownership_sweep_plans(world_size)
        }

    # The recorded pair places both heights and carries bytes at each of them, so
    # the two crossed widths are what retires the blocker there.
    assert placed(2) == {"2": 128, "4": 128}
    assert _MODULE.cut_width_claim_blockers(placed(2)) == ()

    # Every wider shape reaches only the heaviest height, because each of its
    # plans holds more than one boundary at a time. One crossed width is not a
    # sweep, so the blocker the pair retired stands again at those shapes.
    for world_size in range(3, _MODULE.N_WIRES + 1):
        assert placed(world_size) == {"4": 128}
        assert _MODULE.cut_width_claim_blockers(placed(world_size)) == (
            "inter_node_cut_width_not_swept",
        )

    # The profile is recorded whole beside the widths that were placed, so a
    # reader can see which heights the sweep reached and which it did not.
    assert sorted(set(profile)) == [2, 4]


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
