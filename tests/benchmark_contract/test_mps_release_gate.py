"""The MPS release gate is the only thing standing between a run and a claim.

The gate decides whether a set of signed artifacts certifies a distributed
matrix-product-state release, so what it accepts, what it refuses, and what it
names when it refuses are all contract behaviour: a gate that accepted an
approximate MPS claim without sharded ownership, or that described a missing
capability as a missing run, would let the campaign chase the wrong defect. These
tests hold the checked-in manifest to what it declares, hold the gate to every
blocker it can emit, and check that the two capabilities the contract depends on
but does not own -- the evidence envelope and the promotion tool -- behave the way
the contract assumes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from benchmarks.internal.evidence.mps_release_gate import (
    MANIFEST,
    RESULTS,
    baseline_results,
    envelope_carries_world,
    evaluate_mps_release,
    load_manifest,
)
from benchmarks.internal.evidence.mps_release_gate import main as gate_main
from flagquantum.runtime.observability.evidence import (
    ArtifactClass,
    EvidenceScope,
    RuntimeProvenance,
    create_evidence_artifact,
)
from tools.promote_release_candidates import main as promote_main

pytestmark = [pytest.mark.benchmark_contract, pytest.mark.release_gate]

KEYS = b"mps-release-contract-test-key"
# A world of two ranks, one device per host, is the widest topology whose payload
# the evidence envelope can actually seal, so the end-to-end tests below force
# the topology to it. This is not a release topology: the frozen one is sixteen
# ranks over two hosts, and the gate's refusal to pretend otherwise is asserted
# separately.
SEALABLE_TOPOLOGY = {
    "devices_per_node": 1,
    "local_world_size": 1,
    "minimum_multi_node_count": 2,
    "multi_node_required": True,
    "release_world_sizes": [2],
}
# The deployment topology separates provenance lists from the payload while a
# signed artifact is being assembled.
SCOPE_BY_DEVICE_COUNT = {
    1: EvidenceScope.ONE_GPU_LOCAL,
    2: EvidenceScope.TWO_GPU_SEMANTIC,
    4: EvidenceScope.SCHEDULED_SCALE,
    8: EvidenceScope.SCHEDULED_SCALE,
}


def _manifest(topology: dict[str, Any] | None = None, **patches: Any) -> dict[str, Any]:
    """Return the frozen manifest with test-only patches.

    ``premise_established`` and the speed ladder are patched only where a test is
    exercising the accept path, because the checked-in manifest deliberately
    leaves the premise unestablished and the ladder unfrozen. Every other field
    comes from the frozen document itself.
    """

    manifest = copy.deepcopy(load_manifest())
    if topology is not None:
        manifest["topologies"] = dict(topology)
    for name, value in patches.items():
        manifest[name] = value
    return manifest


def _sealable_topology_manifest(**patches: Any) -> dict[str, Any]:
    """Return a manifest whose release world can be sealed, and accepted.

    The premise is established and the speed ladder is frozen here so that the
    accept path is exercised: with the frozen document's real state the gate must
    report those two as open, which is what the checked-in-manifest test asserts.
    """

    manifest = _manifest(SEALABLE_TOPOLOGY, **patches)
    manifest["capacity_workload"]["premise_established"] = True
    manifest["speed_workload"]["configuration_ladder"] = [
        {
            "name": "matched_speed_2x32512_chi768",
            "world_size": 2,
            "n_sites": 65024,
            "sites_per_rank": 32512,
            "bond_dimension": 768,
        }
    ]
    return manifest


def _baseline_evidence(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return the single-device capacity failure the contract reads as provenance."""

    capacity = manifest["capacity_workload"]
    return {
        "acceptance_case": "single_gpu_capacity_failure",
        "state_mode": "mps",
        "distribution_semantics": "single_device_fast_path",
        "mps_forward_distribution_semantics": "single_device_fast_path",
        "world_size": 1,
        "local_world_size": 1,
        "node_count": 1,
        "release_gate_allowed": False,
        "single_device_oom_observed": True,
        "single_gpu_expected_oom": True,
        "capacity_baseline_device": "NVIDIA A800-SXM4-80GB",
        "capacity_failure_reason": "CUDA out of memory",
        "measured_peak_memory_bytes": 75148300800,
        "n_sites": capacity["n_sites"],
        "initial_max_bond": capacity["trained_max_bond"],
        "max_bond_dimension": capacity["trained_max_bond"],
        "batch_size": 1,
        "dtype": "complex64",
        "logical_mps_bytes": capacity["logical_mps_bytes"],
        "site_ownership": {
            "policy": "single_device_owns_every_site",
            "sharded": False,
            "site_count": capacity["n_sites"],
        },
        "workload_sha256": capacity["workload_sha256"],
        "blockers": [],
    }


def _sharded_evidence(
    manifest: dict[str, Any],
    *,
    acceptance_case: str,
    world_size: int | None = None,
    local_world_size: int | None = None,
    node_count: int | None = None,
) -> dict[str, Any]:
    """Return a release payload that satisfies every contract requirement.

    The default is the widest topology the manifest declares, so a test that
    changes nothing gets a payload the gate accepts. Each requirement the gate
    checks has a field here, because a fixture that was missing one would make
    the accept-path test pass for the wrong reason.
    """

    capacity = manifest["capacity_workload"]
    topology = manifest["topologies"]
    world = (
        int(topology["release_world_sizes"][-1]) if world_size is None else world_size
    )
    local = (
        int(topology["local_world_size"])
        if local_world_size is None
        else local_world_size
    )
    nodes = int(world // local) if node_count is None else node_count
    return {
        "acceptance_case": acceptance_case,
        "state_mode": "mps",
        "distribution_semantics": "sharded_across_ranks",
        "mps_forward_distribution_semantics": "sharded_across_ranks",
        "mps_backward_distribution_semantics": "sharded_across_ranks",
        "backward_execution": "executed_sharded_reverse_pass",
        "backward_execution_measured": True,
        "world_size": world,
        "local_world_size": local,
        "node_count": nodes,
        "topology_scope": "multi_node_production_transport",
        "release_gate_allowed": True,
        "training_step_count": 3,
        "n_sites": capacity["n_sites"],
        "max_bond_dimension": capacity["trained_max_bond"],
        "batch_size": capacity["batch_size"],
        "dtype": capacity["dtype"],
        "collective_backend": "nccl",
        "optimizer": capacity["optimizer"],
        "site_shard_ownership": {
            f"rank:{rank}": [rank * 2, rank * 2 + 1] for rank in range(world)
        },
        "bond_shard_ownership": {
            f"bond:{rank}": f"rank:{rank}" for rank in range(world)
        },
        "boundary_gradient_ownership": {
            f"boundary:{rank}": [f"rank:{rank}", f"rank:{rank + 1}"]
            for rank in range(world - 1)
        },
        "parameter_gradient_ownership": {
            f"parameter:{rank}": f"rank:{rank}" for rank in range(world)
        },
        "parameter_ownership": {
            f"rank:{rank}": [f"parameter:{rank}"] for rank in range(world)
        },
        "gradient_ownership": {
            f"rank:{rank}": [f"parameter:{rank}"] for rank in range(world)
        },
        "optimizer_update_ownership": {
            f"rank:{rank}": [f"parameter:{rank}"] for rank in range(world)
        },
        "parameter_ownership_semantics": "sharded_across_ranks",
        "gradient_ownership_semantics": "sharded_across_ranks",
        "optimizer_update_ownership_semantics": "sharded_across_ranks",
        "boundary_adjoint_exchange": {
            "status": "executed",
            "execution_status": "executed",
            "boundary_count": world - 1,
        },
        "boundary_gradient_routes": {
            f"boundary:{rank}": {
                "owner_ranks": [rank, rank + 1],
                "gradient_route": "bounded_dtype_owner_reduce",
            }
            for rank in range(world - 1)
        },
        "mps_backward_memory_plan": {
            "status": "measured",
            "measured_backward_peak_memory_bytes": 68728000512,
        },
        "mps_backward_communication_plan": {
            "status": "executed",
            "boundary_edges": [
                {
                    "owner_ranks": [rank, rank + 1],
                    "execution_status": "executed",
                    "payload_bytes": 37748736,
                }
                for rank in range(world - 1)
            ],
        },
        "boundary_communication_bytes": {
            f"boundary:{rank}": 37748736 for rank in range(world - 1)
        },
        "communication_fraction": 0.12,
        "rank_outputs": [0.0] * world,
        "rank_gradients": [0.0] * world,
        "rank_timings": [1.0] * world,
        "rank_peak_memory_bytes": [68728000512] * world,
        # The envelope refuses to carry a production payload with no
        # runtime-measured field, so the fixture reports the peak the way the
        # producer's payload does rather than only as a per-rank list.
        "measured_peak_memory_bytes": 68728000512,
        "measured_peak_memory_bytes_by_rank": [68728000512] * world,
        "gpu_activity": 0.9,
        "fallback_semantics": manifest["runtime"]["fallback_semantics"],
        "fallback_events": [],
        "single_gpu_expected_oom": True,
        "capacity_baseline_device": "NVIDIA A800-SXM4-80GB",
        "capacity_failure_reason": "CUDA out of memory",
        "speedup": 1.2,
        "speedup_confidence_interval": [1.1, 1.3],
        "scaling_efficiency": 0.62,
        "workload_sha256": capacity["workload_sha256"],
        "blockers": [],
    }


def _payloads(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the artifact payloads a complete MPS release set is made of."""

    return [
        _baseline_evidence(manifest),
        _sharded_evidence(manifest, acceptance_case="multi_gpu_capacity_completion"),
        _sharded_evidence(manifest, acceptance_case="matched_speed"),
    ]


def _scope_for(world_size: int) -> EvidenceScope:
    """Return the narrowest evidence scope that carries a world of this width.

    A world no scope carries returns the widest one, so the envelope itself
    raises its own refusal rather than the fixture inventing a scope name.
    """

    for count, scope in SCOPE_BY_DEVICE_COUNT.items():
        if count == world_size:
            return scope
    return EvidenceScope.SCHEDULED_SCALE


def _provenance(payload: dict[str, Any]) -> RuntimeProvenance:
    """Return the provenance record a sealed envelope would carry.

    The device count follows the payload's world size, because the envelope's
    scope table is what decides whether a payload can be sealed at all.
    """

    world = int(payload["world_size"])
    return RuntimeProvenance(
        commit="a" * 40,
        workload_sha256=str(payload["workload_sha256"]),
        command=("torchrun", "--nproc-per-node", str(world)),
        devices=tuple(f"GPU-{rank}" for rank in range(world)),
        topology="two hosts, one rank per device",
        rank_mapping=tuple(f"rank={rank}" for rank in range(world)),
        collective_backend="nccl",
        warmup=2,
        iterations=10,
        seeds=(440044,),
        raw_log_sha256="b" * 64,
        fallback_events=(),
    )


def _envelope(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the artifact shape the gate reads, without a signature.

    The gate reads evidence out of an envelope and verifies the signature only
    when a key is supplied, which is what lets a contract test exercise the
    contract without a cluster. The frozen sixteen-rank world cannot be sealed by
    any scope, so a test that needs a *signed* envelope forces a sealable topology
    and the tests that read the frozen contract use this shape.
    """

    return {
        "schema": "flagquantum_runtime_evidence_v1",
        "artifact_class": "measured_production_run",
        "evidence_scope": _scope_for(int(payload["world_size"])).value,
        "provenance": asdict(_provenance(payload)),
        "evidence": payload,
        "integrity": {},
    }


def _artifacts(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_envelope(payload) for payload in payloads]


def _read(manifest: dict[str, Any], payloads: list[dict[str, Any]] | None = None):
    """Read the frozen contract over a payload set, without envelope signing."""

    return evaluate_mps_release(
        _artifacts(_payloads(manifest) if payloads is None else payloads), manifest
    )


def _signed(payload: dict[str, Any], *, key: bytes = KEYS) -> dict[str, Any]:
    """Return a signed envelope for a payload the evidence scope can carry."""

    return create_evidence_artifact(
        artifact_class=ArtifactClass.MEASURED_PRODUCTION_RUN,
        evidence_scope=_scope_for(int(payload["world_size"])),
        provenance=_provenance(payload),
        evidence=payload,
        signing_key=key,
    ).summary()


def _write(directory: Path, payloads: list[dict[str, Any]]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for index, payload in enumerate(payloads):
        (directory / f"candidate-{index}.json").write_text(
            json.dumps(_signed(payload)), encoding="utf-8"
        )
    return directory


def _candidate_set(root: Path, manifest: dict[str, Any]) -> tuple[Path, Path, Path]:
    """Lay out a candidate set the way a campaign does.

    Sealed payloads go into a candidate directory and the single-device baseline
    into its own declared provenance directory, because one device has no ranks to
    shard across and can never be release evidence. The manifest is written beside
    them because the gate reads the frozen document from a path. A topology whose
    world no evidence scope carries cannot be sealed, so its candidate set is
    staged as the unsigned shape the gate reads instead.
    """

    payloads = _payloads(manifest)
    baseline = [item for item in payloads if item["world_size"] == 1]
    sharded = [item for item in payloads if item["world_size"] != 1]
    sealable = envelope_carries_world(
        int(manifest["topologies"]["release_world_sizes"][-1])
    )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return (
        manifest_path,
        _write(root / "candidates", sharded)
        if sealable
        else _stage(root / "candidates", sharded),
        _write(root / "baseline", baseline)
        if sealable
        else _stage(root / "baseline", baseline),
    )


def _stage(directory: Path, payloads: list[dict[str, Any]]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for index, payload in enumerate(payloads):
        (directory / f"candidate-{index}.json").write_text(
            json.dumps(_envelope(payload)), encoding="utf-8"
        )
    return directory


def _last_report(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


# Every blocker the payload contract can raise, with the one mutation that raises
# it. The table is exhaustive rather than illustrative: the gate's value is the
# precision of its refusals, so a requirement with no reachable blocker name would
# be a requirement nobody could act on.
PAYLOAD_BLOCKERS: tuple[tuple[str, str], ...] = (
    ("missing_mps_state_mode_evidence", "state_mode", "statevector"),
    (
        "missing_sharded_mps_forward_evidence",
        "mps_forward_distribution_semantics",
        "replicated_per_rank",
    ),
    (
        "missing_sharded_mps_backward_evidence",
        "mps_backward_distribution_semantics",
        "single_device_fast_path",
    ),
    (
        "missing_executed_production_backward_evidence",
        "backward_execution",
        "planned_boundary_adjoint",
    ),
    ("missing_site_shard_ownership", "site_shard_ownership", {}),
    ("missing_bond_shard_ownership", "bond_shard_ownership", {}),
    ("missing_boundary_gradient_ownership", "boundary_gradient_ownership", {}),
    ("missing_parameter_gradient_ownership", "parameter_gradient_ownership", {}),
    (
        "missing_executed_boundary_adjoint_exchange",
        "boundary_adjoint_exchange",
        {"status": "planned", "execution_status": "planned"},
    ),
    (
        "missing_production_backward_memory_evidence",
        "mps_backward_memory_plan",
        {"status": "planned", "measured_backward_peak_memory_bytes": 4096},
    ),
    (
        "missing_production_boundary_communication_evidence",
        "mps_backward_communication_plan",
        {"status": "pending", "boundary_edges": [{"execution_status": "executed"}]},
    ),
    (
        "missing_sharded_training_ownership",
        "distribution_semantics",
        "replicated_per_rank",
    ),
    ("missing_positive_training_step_count", "training_step_count", 0),
    ("missing_single_gpu_expected_oom_declaration", "single_gpu_expected_oom", False),
    ("missing_no_fallback_semantics", "fallback_semantics", "statevector_fallback"),
    (
        "production_payload_declares_fallback_events",
        "fallback_events",
        ["statevector_fallback"],
    ),
    ("production_payload_declares_blockers", "blockers", ["rank_0_idle"]),
    ("missing_capacity_baseline_device", "capacity_baseline_device", ""),
    ("missing_capacity_failure_reason", "capacity_failure_reason", ""),
    ("missing_multinode_transport_scope", "topology_scope", "single_node_local"),
    ("missing_capacity_workload_shape_evidence", "n_sites", None),
    ("capacity_workload_shape_mismatch", "n_sites", 4096),
    ("production_artifact_missing_required_fields", "collective_backend", ""),
)


def test_the_manifest_freezes_the_workload_the_topology_and_the_thresholds() -> None:
    manifest = load_manifest()
    capacity = manifest["capacity_workload"]

    assert manifest["capability"] == "distributed_matrix_product_state"
    assert manifest["frozen_before_release_run"] is True
    # One logical workload, sharded: a wide-site MPS whose bonds also exceed one
    # device's memory, measured at batch size one and one optimizer step.
    assert capacity["n_sites"] == 131072
    assert capacity["trained_max_bond"] == 768
    assert capacity["batch_size"] == 1
    assert capacity["steps"] == 1
    assert capacity["logical_mps_bytes"] > capacity["single_device_total_memory_bytes"]
    assert capacity["state_mode"] == "mps"
    assert manifest["runtime"]["state_mode"] == "mps"
    assert manifest["runtime"]["backend"] == "pytorch_native"
    assert manifest["topologies"]["minimum_multi_node_count"] == 2
    assert manifest["topologies"]["multi_node_required"] is True
    assert manifest["speed_workload"]["minimum_speedup"] > 1
    assert manifest["speed_workload"]["confidence_interval_must_exclude_speedup"] == 1.0
    assert manifest["required_artifact_fields"]


def test_the_frozen_workload_definition_is_digest_bound() -> None:
    """The manifest names the files that define the workload, and their bytes."""

    capacity = load_manifest()["capacity_workload"]
    launcher = Path(capacity["workload_definition_path"])

    assert (
        hashlib.sha256(launcher.read_bytes()).hexdigest()
        == (capacity["workload_definition_sha256"])
    )
    assert capacity["workload_sha256"] == capacity["workload_definition_sha256"]
    # The body is bound as well as the launcher: the launcher pins the site count
    # while the body builds the circuit and the rank boundaries, so digesting only
    # the launcher would let the measured workload change under a stable filename.
    roles = {
        source["role"].split(":")[0]
        for source in capacity["workload_definition_sources"]
    }
    assert roles == {"launcher", "workload body"}
    for source in capacity["workload_definition_sources"]:
        path = Path(source["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]


def test_a_workload_that_drifted_from_its_frozen_digest_is_rejected(
    tmp_path: Path,
) -> None:
    manifest = load_manifest()
    manifest["capacity_workload"]["workload_definition_sources"][1]["sha256"] = "0" * 64
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match the digest"):
        load_manifest(path)


def test_the_single_device_baseline_is_provenance_evidence_not_a_release_world() -> (
    None
):
    manifest = load_manifest()
    baseline = manifest["capacity_workload"]["single_gpu_baseline"]

    assert baseline["release_gate_payload"] is False
    assert baseline["evidence_scope"] == "one_gpu_local"
    assert baseline["evidence_role"] == "provenance_evidence"
    assert 1 not in manifest["topologies"]["release_world_sizes"]


def test_the_baseline_directory_stays_outside_the_promoted_release_directory() -> None:
    path = baseline_results(load_manifest())

    assert path != RESULTS
    assert RESULTS not in path.parents
    assert path.is_relative_to("benchmarks/results")


def test_a_baseline_directory_under_the_promoted_set_is_rejected() -> None:
    manifest = load_manifest()
    manifest["capacity_workload"]["single_gpu_baseline"]["results_path"] = str(
        RESULTS / "mps_single_gpu_capacity"
    )

    with pytest.raises(ValueError, match="outside"):
        baseline_results(manifest)


# The blockers that describe the contract or the manifest rather than one
# payload's fields. They are listed here as well as asserted below because the
# blocker-vocabulary test checks that every name the gate can print is accounted
# for by a test in this file.
MANIFEST_STATE_BLOCKERS: tuple[str, ...] = (
    "release_world_size_not_carriable_by_evidence_envelope",
    "missing_release_world_sizes",
    "missing_multinode_correctness_artifact",
    "speed_configuration_ladder_not_frozen",
    "missing_statistically_significant_speedup_artifact",
    "capacity_premise_not_established",
    "missing_single_gpu_measured_oom_artifact",
    "single_gpu_baseline_missing_required_fields",
    "missing_multi_gpu_capacity_completion_artifact",
    "invalid_or_unsigned_production_artifact",
    "single_device_world_cannot_be_a_release_payload",
)


def test_the_checked_in_manifest_discloses_every_open_requirement() -> None:
    """The checked-in state fails the gate for exactly three named reasons.

    A full payload set is supplied, so every requirement a run can satisfy is
    satisfied. What remains are the three things the frozen document itself
    discloses: the capacity premise is unestablished, no MPS speed configuration
    has been chosen or timed, and the sixteen-rank release world is wider than any
    evidence scope can carry. Asserting the exact tuple means a fourth blocker
    appearing here would have to be explained rather than absorbed.
    """

    manifest = load_manifest()
    passed, blockers = _read(manifest)

    assert passed is False
    assert blockers == (
        "release_world_size_not_carriable_by_evidence_envelope",
        "speed_configuration_ladder_not_frozen",
        "capacity_premise_not_established",
    )
    assert manifest["capacity_workload"]["premise_established"] is False
    assert manifest["speed_workload"]["configuration_ladder"] is None


def test_a_complete_release_set_satisfies_the_contract() -> None:
    """The accept path is reachable, so the refusal list is not vacuous.

    The premise is established and the speed ladder is frozen here because those
    two are the manifest's disclosures rather than a payload's shortcomings; every
    other requirement is met by the same fixture the refusal tests mutate.
    """

    manifest = _sealable_topology_manifest()
    passed, blockers = _read(manifest)

    assert blockers == ()
    assert passed is True


def test_the_frozen_release_world_cannot_be_sealed_by_the_evidence_envelope() -> None:
    """The contract's widest dependency is stated rather than discovered late.

    The frozen world is sixteen ranks, and no evidence scope carries more than
    eight devices, so no sealed artifact can describe it. The gate names that
    capability gap instead of reporting a missing run, and the manifest discloses
    the same bound so that a campaign can read it before spending the cluster.
    """

    frozen = load_manifest()

    assert frozen["topologies"]["release_world_sizes"] == [16]
    assert envelope_carries_world(1) is True
    assert envelope_carries_world(2) is True
    assert envelope_carries_world(8) is True
    assert envelope_carries_world(16) is False
    assert frozen["release_payload_envelope"]["carriable"] is False
    assert frozen["release_payload_envelope"]["device_counts_by_scope"] != {}
    with pytest.raises(ValueError, match="requires device count"):
        _signed(_sharded_evidence(frozen, acceptance_case="matched_speed"))
    passed, blockers = _read(frozen)
    assert passed is False
    assert "release_world_size_not_carriable_by_evidence_envelope" in blockers


def test_flat_unsigned_production_payload_is_rejected() -> None:
    artifact = {
        "artifact_class": "measured_production_run",
        "release_gate_allowed": True,
        "world_size": 8,
    }

    passed, blockers = evaluate_mps_release([artifact], load_manifest())

    assert passed is False
    assert "invalid_or_unsigned_production_artifact" in blockers


def test_an_envelope_signed_with_another_key_is_rejected() -> None:
    manifest = _sealable_topology_manifest()
    artifacts = [_signed(payload) for payload in _payloads(manifest)][1:]

    passed, blockers = evaluate_mps_release(
        artifacts, manifest, signing_key=b"a-different-campaign-key"
    )

    assert passed is False
    assert "invalid_or_unsigned_production_artifact" in blockers


def test_single_device_world_cannot_pose_as_a_release_payload() -> None:
    manifest = _sealable_topology_manifest()
    payloads = _payloads(manifest)
    payloads[0]["release_gate_allowed"] = True

    passed, blockers = _read(manifest, payloads)

    assert passed is False
    assert "single_device_world_cannot_be_a_release_payload" in blockers


def test_a_missing_shape_field_is_reported_apart_from_a_wrong_one() -> None:
    """An incomplete record and a different workload are different defects."""

    manifest = _sealable_topology_manifest()
    absent = _payloads(manifest)
    del absent[1]["n_sites"]
    _, absent_blockers = _read(manifest, absent)
    altered = _payloads(manifest)
    altered[1]["n_sites"] = manifest["capacity_workload"]["n_sites"] // 2
    _, altered_blockers = _read(manifest, altered)

    assert "missing_capacity_workload_shape_evidence" in absent_blockers
    assert "capacity_workload_shape_mismatch" not in absent_blockers
    assert "capacity_workload_shape_mismatch" in altered_blockers
    assert "missing_capacity_workload_shape_evidence" not in altered_blockers


@pytest.mark.parametrize(
    ("blocker", "field", "value"),
    PAYLOAD_BLOCKERS,
    ids=[item[0] for item in PAYLOAD_BLOCKERS],
)
def test_every_payload_contract_blocker_is_reachable(
    blocker: str, field: str, value: Any
) -> None:
    manifest = _sealable_topology_manifest()
    payloads = copy.deepcopy(_payloads(manifest))
    if value is None:
        del payloads[1][field]
    else:
        payloads[1][field] = value

    passed, blockers = _read(manifest, payloads)

    assert passed is False
    assert blocker in blockers


def test_every_manifest_state_blocker_is_reachable() -> None:
    """The blockers that describe the contract rather than one payload's fields."""

    established = _sealable_topology_manifest()
    frozen = load_manifest()

    def blockers_for(
        manifest: dict[str, Any], payloads: list[dict[str, Any]]
    ) -> tuple[str, ...]:
        return _read(manifest, payloads)[1]

    # A world wider than the envelope can carry is reported by the gate and not
    # left for the sealer to discover.
    assert "release_world_size_not_carriable_by_evidence_envelope" in blockers_for(
        frozen, _payloads(frozen)
    )
    # No sealed payload at all: the release world size is the manifest's.
    assert "missing_release_world_sizes" in blockers_for(
        established, [_baseline_evidence(established)]
    )
    # A ladder chosen after the numbers were known is not a frozen protocol.
    unfrozen = _sealable_topology_manifest()
    unfrozen["speed_workload"]["configuration_ladder"] = None
    assert "speed_configuration_ladder_not_frozen" in blockers_for(
        unfrozen, _payloads(unfrozen)
    )
    # A named speed case cannot bypass the frozen statistical thresholds.
    weak = _sealable_topology_manifest()
    payloads = _payloads(weak)
    payloads[2]["speedup_confidence_interval"] = [0.99, 1.4]
    assert "missing_statistically_significant_speedup_artifact" in blockers_for(
        weak, payloads
    )
    # Every artifact sharded but none spanning hosts: transport scope is missing.
    single_node = _sealable_topology_manifest()
    payloads = [
        _baseline_evidence(single_node),
        _sharded_evidence(
            single_node,
            acceptance_case="multi_gpu_capacity_completion",
            node_count=1,
        ),
        _sharded_evidence(single_node, acceptance_case="matched_speed", node_count=1),
    ]
    assert "missing_multinode_correctness_artifact" in blockers_for(
        single_node, payloads
    )
    assert "missing_multinode_transport_scope" in blockers_for(single_node, payloads)
    # Signing is impossible for a payload the envelope cannot carry, so the
    # premise-dependent blockers are reached with an unsigned contract read.
    premise = _sealable_topology_manifest()
    premise["capacity_workload"]["premise_established"] = False
    assert "capacity_premise_not_established" in blockers_for(
        premise, _payloads(premise)
    )
    assert "missing_single_gpu_measured_oom_artifact" in blockers_for(
        established,
        [
            payload
            for payload in _payloads(established)
            if payload["acceptance_case"] != "single_gpu_capacity_failure"
        ],
    )
    incomplete = _sealable_topology_manifest()
    payloads = copy.deepcopy(_payloads(incomplete))
    del payloads[0]["logical_mps_bytes"]
    assert "single_gpu_baseline_missing_required_fields" in blockers_for(
        incomplete, payloads
    )
    assert "missing_multi_gpu_capacity_completion_artifact" in blockers_for(
        established,
        [
            payload
            for payload in _payloads(established)
            if payload["acceptance_case"] != "multi_gpu_capacity_completion"
        ],
    )


def test_the_blocker_vocabulary_is_fully_accounted_for() -> None:
    """Every name the gate can print is asserted somewhere in this file.

    The gate's value is the precision of its refusals, so a requirement that could
    only fail under an unnamed blocker would be a requirement nobody could act on.
    The check is in both directions: a name the gate can emit but no test asserts,
    and a name a test asserts but the gate cannot emit, are both failures.
    """

    source = Path("benchmarks/internal/evidence/mps_release_gate.py").read_text(
        encoding="utf-8"
    )
    emitted = set(re.findall(r'blockers\.append\(\s*"([a-z_]+)"', source))
    emitted |= {
        pair[1] for pair in re.findall(r'\("([a-z_]+)", "(missing_[a-z_]+)"\)', source)
    }
    accounted = (
        {item[0] for item in PAYLOAD_BLOCKERS}
        | set(MANIFEST_STATE_BLOCKERS)
        | {
            "missing_sharded_training_ownership",
            "production_artifact_missing_required_fields",
        }
    )

    assert emitted - accounted == set()
    assert accounted - emitted == set()


# ``benchmarks/results/scalability/README.md`` is the published release contract:
# its Phase 5 section lists the fields an MPS payload must add. One entry per
# bullet here, in the order the README states them, naming the blockers that
# enforce the bullet.
README_REQUIREMENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("state_mode", ("missing_mps_state_mode_evidence",)),
    ("distribution_semantics", ("missing_sharded_mps_forward_evidence",)),
    (
        "sharded_forward_and_executed_backward_evidence",
        (
            "missing_sharded_mps_forward_evidence",
            "missing_sharded_mps_backward_evidence",
            "missing_executed_production_backward_evidence",
        ),
    ),
    (
        "site_bond_boundary_and_parameter_ownership",
        (
            "missing_site_shard_ownership",
            "missing_bond_shard_ownership",
            "missing_boundary_gradient_ownership",
            "missing_parameter_gradient_ownership",
        ),
    ),
    ("boundary_adjoint_exchange", ("missing_executed_boundary_adjoint_exchange",)),
    ("backward_memory", ("missing_production_backward_memory_evidence",)),
    ("boundary_communication", ("missing_production_boundary_communication_evidence",)),
    ("sharded_training_ownership", ("missing_sharded_training_ownership",)),
    ("positive_training_step_count", ("missing_positive_training_step_count",)),
    (
        "single_gpu_expected_oom",
        (
            "missing_single_gpu_expected_oom_declaration",
            "missing_capacity_baseline_device",
            "missing_capacity_failure_reason",
        ),
    ),
    (
        "no_fallback_semantics_and_empty_blockers",
        (
            "missing_no_fallback_semantics",
            "production_payload_declares_fallback_events",
            "production_payload_declares_blockers",
        ),
    ),
)
# The README requires "all general release fields plus" that list, so the general
# fields this contract turns into a requirement are named here.
GENERAL_RELEASE_REQUIREMENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("multinode_transport_scope", ("missing_multinode_transport_scope",)),
    (
        "capacity_workload_shape",
        (
            "missing_capacity_workload_shape_evidence",
            "capacity_workload_shape_mismatch",
        ),
    ),
)


def _mapped_requirement_blockers() -> set[str]:
    """Return every blocker the published contract is mapped onto."""

    return {
        blocker
        for _, blockers in (*README_REQUIREMENTS, *GENERAL_RELEASE_REQUIREMENTS)
        for blocker in blockers
    }


def _payload_requirement_blockers() -> set[str]:
    """Return every blocker ``_mps_contract_blockers`` can emit."""

    source = Path("benchmarks/internal/evidence/mps_release_gate.py").read_text(
        encoding="utf-8"
    )
    start = source.index("def _mps_contract_blockers")
    end = source.index("def evaluate_mps_release")
    emitted = set(re.findall(r'blockers\.append\(\s*"([a-z_]+)"', source[start:end]))
    emitted |= {
        pair[1]
        for pair in re.findall(
            r'\("([a-z_]+)", "(missing_[a-z_]+)"\)', source[start:end]
        )
    }
    return emitted


def test_the_published_release_contract_is_what_the_gate_enforces() -> None:
    """The README's Phase 5 list and the requirement blockers are one contract.

    The README is what a reader is told a release payload must carry, so the gate
    has to enforce exactly that list: a requirement the README states but no
    blocker enforces would be a promise the gate cannot keep, and a payload
    blocker the README never states would refuse a payload for an unpublished
    reason. Each bullet is mapped to its blockers, and the mapping is checked
    against the gate rather than against a copy of it.
    """

    readme = Path("benchmarks/results/scalability/README.md").read_text(
        encoding="utf-8"
    )
    section = readme[readme.index("## Phase 5 MPS Preparation") :]
    bullets = [line for line in section.splitlines() if line.startswith("- ")]

    # One mapping entry per bullet, so a bullet added to the README without a
    # blocker behind it fails here rather than being enforced nowhere.
    assert len(README_REQUIREMENTS) == len(bullets)
    for phrase in (
        "`state_mode` identifying an MPS path",
        '`distribution_semantics="sharded_across_ranks"`',
        "sharded MPS forward and executed production backward evidence",
        "site, bond, boundary-gradient, and parameter-gradient ownership",
        "executed boundary-adjoint exchange evidence",
        "production-measured per-rank backward memory",
        "production-executed per-boundary communication",
        "sharded parameter, gradient, and optimizer-update ownership",
        "`training_step_count > 0`",
        "`single_gpu_expected_oom=True` with a real baseline device and failure",
        "explicit no-fallback semantics and empty blockers",
    ):
        assert phrase in section

    mapped = _mapped_requirement_blockers()
    emitted = _payload_requirement_blockers()

    assert emitted - mapped == set()
    assert mapped - emitted == set()


def test_malformed_numeric_measurements_fail_closed_without_crashing() -> None:
    manifest = _sealable_topology_manifest()
    payloads = _payloads(manifest)
    del payloads[1]["training_step_count"]
    payloads[2]["speedup"] = None
    payloads[2]["scaling_efficiency"] = "unknown"

    passed, blockers = _read(manifest, payloads)

    assert passed is False
    assert "missing_positive_training_step_count" in blockers
    assert "missing_sharded_training_ownership" in blockers
    assert "missing_statistically_significant_speedup_artifact" in blockers


def test_the_gate_evaluates_a_candidate_set_before_promotion(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sealing stages into a candidate directory, so the gate has to read one."""

    manifest_path, candidates, baseline = _candidate_set(
        tmp_path, _sealable_topology_manifest()
    )
    monkeypatch.setenv("FQ_EVIDENCE_SIGNING_KEY", KEYS.decode())

    gate_main(
        [
            "--manifest",
            str(manifest_path),
            "--candidate",
            str(candidates),
            "--baseline-directory",
            str(baseline),
        ]
    )

    report = _last_report(capsys)
    assert report["passed"] is True
    assert report["capability"] == "distributed_matrix_product_state"
    assert report["artifact_count"] == 3
    assert str(candidates) in report["evaluated"]
    assert str(baseline) in report["evaluated"]
    assert report["blockers"] == []


def test_a_candidate_set_is_not_promoted_when_the_gate_refuses_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, candidates, baseline = _candidate_set(tmp_path, load_manifest())
    monkeypatch.setenv("FQ_EVIDENCE_SIGNING_KEY", KEYS.decode())
    release = tmp_path / "release"

    with pytest.raises(SystemExit) as excinfo:
        promote_main(
            [
                "--gate",
                "mps",
                "--candidate",
                str(candidates),
                "--baseline-directory",
                str(baseline),
                "--release-directory",
                str(release),
            ]
        )

    assert "capacity_premise_not_established" in str(excinfo.value)
    assert not release.exists()
    assert manifest_path.is_file()


def test_an_unsigned_candidate_set_is_never_promoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, candidates, baseline = _candidate_set(
        tmp_path, _sealable_topology_manifest()
    )
    monkeypatch.delenv("FQ_EVIDENCE_SIGNING_KEY", raising=False)
    release = tmp_path / "release"

    with pytest.raises(SystemExit) as excinfo:
        promote_main(
            [
                "--gate",
                "mps",
                "--candidate",
                str(candidates),
                "--baseline-directory",
                str(baseline),
                "--release-directory",
                str(release),
            ]
        )

    assert "FQ_EVIDENCE_SIGNING_KEY is required" in str(excinfo.value)
    assert not release.exists()
    assert manifest_path.is_file()


def test_promotion_moves_only_the_release_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The single-device baseline is read as provenance and left where it is.

    Promote now uses the frozen manifest, so the topology it checks is the
    checked-in sixteen-rank one and the staged set cannot pass. The promotion path
    that moves files is therefore exercised through its refusal, and the baseline
    is asserted to stay in place -- which is the property that matters here.
    """

    _, candidates, baseline = _candidate_set(tmp_path, load_manifest())
    staged = sorted(path.name for path in candidates.glob("*.json"))
    monkeypatch.setenv("FQ_EVIDENCE_SIGNING_KEY", KEYS.decode())

    with pytest.raises(SystemExit):
        promote_main(
            [
                "--gate",
                "mps",
                "--candidate",
                str(candidates),
                "--baseline-directory",
                str(baseline),
                "--release-directory",
                str(tmp_path / "release"),
            ]
        )

    assert sorted(path.name for path in candidates.glob("*.json")) == staged
    assert len(list(baseline.glob("*.json"))) == 1
    assert MANIFEST.is_file()
