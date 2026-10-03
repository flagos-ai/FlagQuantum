"""The MPS release producer is machine-checked before it is trusted.

The producer is what turns a measured distributed MPS run into a payload the
release gate reads, so the parts of it that decide *what a payload may claim* are
tested here rather than left to the campaign that runs it: the frozen workload
whose arithmetic the manifest digest-binds, the shape guard that refuses a launch
at a bond dimension or step count nobody froze, the agreement rule that stops one
rank speaking for the others, the ownership and boundary maps that every rank has
to corroborate, the capacity premise the completion inherits, and the projection
that lifts a role document onto the contract the sealer reads. A producer that got
any of these wrong would seal a run as evidence for something it did not measure.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from benchmarks import mps_release_evidence as producer
from benchmarks.internal.evidence import mps_release_gate as gate
from benchmarks.internal.evidence.mps_release_gate import MANIFEST as GATE_MANIFEST
from flagquantum.testing import MPSCapacityCertificationError

pytestmark = [pytest.mark.benchmark_contract, pytest.mark.release_gate]

RELEASE_MANIFEST = Path("benchmarks/manifests/mps_release_v1.json")
FROZEN = json.loads(RELEASE_MANIFEST.read_text(encoding="utf-8"))
CAPACITY = FROZEN["capacity_workload"]
PREMISE_ARTIFACT = Path(CAPACITY["premise_provenance"]["artifact"])
FAILURE_REASON = "CUDA out of memory"


def _contract() -> dict:
    return producer._capacity_contract(RELEASE_MANIFEST)


def _sharded_contract_fields() -> set[str]:
    """Return the fields the sharded contract is assembled from.

    The contract is one literal dictionary, so its keys can be read without
    building a run: this is what lets the producer be checked against the release
    manifest before a campaign spends a cluster on it.
    """

    tree = ast.parse(Path(producer.__file__).read_text(encoding="utf-8"))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_sharded_contract"
    )
    returned = next(node for node in ast.walk(function) if isinstance(node, ast.Return))
    return {key.value for key in returned.value.keys if isinstance(key, ast.Constant)}


def _gate_payload_fields() -> set[str]:
    """Return every payload field the requirement checker reads.

    These are the fields whose absence turns into a blocker, so the producer has
    to state all of them for a measured run to satisfy the contract.
    """

    source = Path(gate.__file__).read_text(encoding="utf-8")
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in {"_mps_contract_blockers", "_is_sharded_training"}:
            continue
        for item in ast.walk(node):
            if (
                isinstance(item, ast.Call)
                and isinstance(item.func, ast.Attribute)
                and item.func.attr == "get"
                and isinstance(item.func.value, ast.Name)
                and item.func.value.id == "evidence"
                and item.args
                and isinstance(item.args[0], ast.Constant)
                and isinstance(item.args[0].value, str)
            ):
                names.add(item.args[0].value)
            if (
                isinstance(item, ast.Subscript)
                and isinstance(item.value, ast.Name)
                and item.value.id == "evidence"
                and isinstance(item.slice, ast.Constant)
                and isinstance(item.slice.value, str)
            ):
                names.add(item.slice.value)
    return names


def _failure_record(
    path: Path, *, status: str = "cuda_oom", device_name: str | None = None
) -> Path:
    """Write a single-device failure record the premise can point at."""

    record: dict[str, object] = {
        "status": status,
        "single_gpu_peak_memory_bytes": 75148300800,
        "single_gpu_device_total_memory_bytes": 85093777408,
        "rank_record": {
            "error": FAILURE_REASON,
            "device_total_memory_bytes": 85093777408,
        },
    }
    if device_name is not None:
        record["rank_record"]["device_name"] = device_name
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _source(path: Path, kind: str) -> dict[str, str]:
    return {
        "kind": kind,
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _edge(ranks: list[int], bond: int) -> dict[str, object]:
    return {
        "owner_ranks": ranks,
        "bond": bond,
        "operation_id": f"bond:{bond}",
        "payload_bytes": 1024,
        "forward_transport": "isend_irecv",
        "reverse_transport": "isend_irecv",
    }


def test_the_producer_and_the_gate_freeze_the_same_manifest() -> None:
    """Two loaders of one document cannot disagree about what was frozen."""

    assert producer.RELEASE_MANIFEST == GATE_MANIFEST
    assert producer.SCHEMA == "flagquantum.mps_release_measurements.v1"
    assert set(producer.ROLES) == {
        "capacity-failure",
        "capacity-completion",
        "matched-speed",
        "speed-summary",
        "release-payload",
    }


def test_the_producer_states_every_field_the_release_contract_requires() -> None:
    """A payload the gate would reject as incomplete is a producer defect.

    The four fields the sealer reads off the machine that ran the rank --
    ``commit``, ``hardware_inventory``, ``raw_log_sha256`` and ``rank_mapping`` --
    are carried by the envelope, and the checker itself says which they are, so
    the requirement is read from the gate rather than restated here.
    """

    produced = _sharded_contract_fields()
    envelope = {
        "devices": ("GPU-0000",),
        "raw_log_sha256": "a" * 64,
        "commit": "b" * 40,
        "rank_mapping": ("rank=0:device_uuid=GPU-0000",),
    }
    required_from_payload = set(
        gate._has_required_fields({}, envelope, FROZEN["required_artifact_fields"])
    )

    assert required_from_payload - produced == set()


def test_the_producer_states_every_field_the_gate_reads_for_a_requirement() -> None:
    """Every field a blocker is decided from is a field the producer measures.

    Three names are alternatives the gate accepts instead of the field the
    producer states: the two distribution semantics have an ``mps_``-prefixed form
    and the backward peak memory is also read from the memory plan.
    """

    alternatives = {
        "forward_distribution_semantics",
        "backward_distribution_semantics",
        "rank_backward_peak_memory_bytes",
    }

    assert _gate_payload_fields() - _sharded_contract_fields() - alternatives == set()


def test_the_checked_in_manifest_requires_the_fields_the_gate_reads() -> None:
    """The frozen field list covers every field a requirement is decided from."""

    required = set(FROZEN["required_artifact_fields"])
    expected = {
        "backward_execution",
        "collective_backend",
        "training_step_count",
        "single_gpu_expected_oom",
        "capacity_baseline_device",
        "capacity_failure_reason",
        "fallback_semantics",
    }

    assert expected <= required


def test_the_frozen_workload_is_the_one_the_manifest_digest_binds() -> None:
    """The producer measures the workload the manifest froze, not a near copy."""

    contract = _contract()
    assert producer._workload_digest(contract) == CAPACITY["workload_sha256"]
    # The body is bound as well as the launcher: the launcher pins the site count
    # while the body builds the circuit and the rank boundaries, so digesting only
    # the launcher would let the measured workload change under a stable filename.
    for source in (
        {
            "path": contract["workload_definition_path"],
            "sha256": contract["workload_definition_sha256"],
        },
        *contract["workload_definition_sources"],
    ):
        path = Path(source["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]

    workload = producer._frozen_workload(contract)
    assert contract["n_sites"] == workload.N_SITES
    assert contract["trained_max_bond"] == workload.MAX_BOND
    assert contract["target_world_size"] == workload.TARGET_WORLD
    assert (
        workload.logical_mps_bytes(workload.N_SITES, workload.MAX_BOND)
        == contract["logical_mps_bytes"]
    )


def test_the_contract_and_the_code_that_builds_the_workload_cannot_drift() -> None:
    contract = _contract()
    drifted = copy.deepcopy(contract)
    drifted["workload_definition_sources"][1]["sha256"] = "0" * 64

    with pytest.raises(SystemExit, match="not the workload the release manifest froze"):
        producer._frozen_workload(drifted)


def test_a_workload_whose_arithmetic_disagrees_with_the_manifest_is_refused() -> None:
    """A frozen shape and a manifest number that contradict each other stop the run."""

    contract = _contract()
    contract["logical_mps_bytes"] = int(contract["logical_mps_bytes"]) + 1

    with pytest.raises(SystemExit, match="logical MPS bytes"):
        producer._frozen_workload(contract)


def test_a_launch_at_a_shape_the_manifest_did_not_freeze_is_refused() -> None:
    """The bond dimension and step count are workload identity, not tuning."""

    contract = _contract()
    frozen_bond = int(contract["trained_max_bond"])
    frozen_steps = int(contract["steps"])

    producer._require_frozen_shape(
        argparse.Namespace(bond_dimension=frozen_bond, steps=frozen_steps), contract
    )
    # A launcher that configured nothing states nothing, which is not the same as
    # a launcher that configured a different workload.
    producer._require_frozen_shape(
        argparse.Namespace(bond_dimension=None, steps=None), contract
    )

    with pytest.raises(SystemExit, match="maximum bond dimension"):
        producer._require_frozen_shape(
            argparse.Namespace(bond_dimension=frozen_bond // 2, steps=None), contract
        )
    with pytest.raises(SystemExit, match="optimizer"):
        producer._require_frozen_shape(
            argparse.Namespace(bond_dimension=None, steps=frozen_steps + 1), contract
        )


def test_every_rank_has_to_agree_before_a_global_fact_is_stated() -> None:
    """Global evidence is stated by every rank, so a disagreement is a defect."""

    shared = {"rank:0": ["parameter:0"]}
    assert producer._agreed([shared, shared], context="the site map") == shared
    with pytest.raises(SystemExit, match="disagree"):
        producer._agreed([shared, {"rank:0": ["parameter:1"]}], context="the site map")
    with pytest.raises(SystemExit, match="no rank published"):
        producer._agreed([], context="the site map")


def test_the_parameter_ownership_map_is_grouped_by_the_rank_that_owns_it() -> None:
    entries = (
        {"owner_rank": 1, "parameter_index": 3},
        {"owner_rank": 0, "parameter_index": 0},
        {"owner_rank": 1, "parameter_index": 2},
    )

    assert producer._ownership_map(entries, world_size=2) == {
        "rank:0": ["parameter:0"],
        "rank:1": ["parameter:2", "parameter:3"],
    }
    with pytest.raises(SystemExit, match="outside a world"):
        producer._ownership_map(
            ({"owner_rank": 2, "parameter_index": 0},), world_size=2
        )
    with pytest.raises(SystemExit, match="no parameter ownership"):
        producer._ownership_map((), world_size=2)


def test_the_payload_states_one_route_for_every_parameter() -> None:
    entries = (
        {"owner_rank": 0, "gradient_route": "bounded_dtype_owner_reduce"},
        {"owner_rank": 1, "gradient_route": "bounded_dtype_owner_reduce"},
    )
    assert (
        producer._routes(entries, "gradient_route", context="gradient route")
        == "bounded_dtype_owner_reduce"
    )
    # A leg whose parameters travelled by different routes has no single route to
    # report, and one that reported none at all is the same refusal.
    with pytest.raises(SystemExit, match="one gradient route"):
        producer._routes(
            (entries[0], {"owner_rank": 1, "gradient_route": "local_replay"}),
            "gradient_route",
            context="gradient route",
        )
    with pytest.raises(SystemExit, match="one gradient route"):
        producer._routes(
            ({"owner_rank": 0},), "gradient_route", context="gradient route"
        )


def test_a_boundary_is_stated_once_and_only_where_both_ranks_reported_it() -> None:
    """A boundary only one rank saw is the signature of a local transport."""

    pair = _edge([0, 1], bond=8)
    edges = producer._boundary_edges(([pair], [pair]), world_size=2)
    assert edges == [
        {
            "owner_ranks": [0, 1],
            "bond": 8,
            "operation_id": "bond:8",
            "payload_bytes": 1024,
            "forward_transport": "isend_irecv",
            "reverse_transport": "isend_irecv",
            "execution_status": "executed",
        }
    ]
    # Reported by one rank only: the other rank did not take part in it, so the
    # transport did not cross the cut it claims to cross.
    with pytest.raises(SystemExit, match="only one of the ranks which own it"):
        producer._boundary_edges(([pair], []), world_size=2)
    with pytest.raises(SystemExit, match="no transport for every adjacent rank"):
        producer._boundary_edges(([pair], [pair], []), world_size=3)
    with pytest.raises(SystemExit, match="disagree about the boundary"):
        producer._boundary_edges(([pair], [_edge([0, 1], bond=9)]), world_size=2)
    with pytest.raises(SystemExit, match="not a pair of adjacent ranks"):
        producer._boundary_edges(([_edge([0, 1, 2], bond=8)],), world_size=3)
    with pytest.raises(SystemExit, match="outside a world"):
        producer._boundary_edges(([_edge([0, 1], bond=8)],), world_size=1)


def test_the_host_layout_decides_which_measured_boundaries_cross_hosts() -> None:
    """Intra-host traffic may not be reported as inter-host transport."""

    edges = [_edge([rank, rank + 1], bond=rank + 1) for rank in range(7)]

    assert producer._node_placement(edges, local_world_size=4, node_count=2) == [[3, 4]]
    # The declared layout puts the host cut at rank 3|4, and the measured leg
    # reported no boundary there.
    cut_missing = [edge for rank, edge in enumerate(edges) if rank != 3]
    with pytest.raises(SystemExit, match="do not cross hosts"):
        producer._node_placement(cut_missing, local_world_size=4, node_count=2)
    with pytest.raises(SystemExit, match="positive local world size"):
        producer._node_placement(edges, local_world_size=0, node_count=2)


def test_bond_ownership_names_the_pair_that_measured_each_cut() -> None:
    ownership = producer._bond_shard_ownership(
        [[0, 1, 2], [3, 4]], [_edge([0, 1], bond=12)]
    )

    assert ownership["bond:0-2"] == "rank:0"
    assert ownership["bond:3-4"] == "rank:1"
    # A cut bond is owned by both ranks that exchanged it, which is what the
    # measured boundary record names rather than something inferred from the split.
    assert ownership["bond:12"] == "rank:0+rank:1"


def test_the_capacity_baseline_states_single_device_semantics() -> None:
    """One device has nothing to shard, and the payload says so."""

    baseline = producer._baseline_measurements(
        _contract(),
        status="completed",
        failure_reason="",
        peak_memory_bytes=1024,
        device_total_memory_bytes=2048,
        device_name="NVIDIA A800-SXM4-80GB",
        digest=CAPACITY["workload_sha256"],
        world_size=1,
        local_world_size=1,
        node_count=1,
    )

    assert baseline["distribution_semantics"] == "single_device_fast_path"
    assert baseline["mps_forward_distribution_semantics"] == "single_device_fast_path"
    assert baseline["site_ownership"]["sharded"] is False
    # A workload that fitted on one device is not a capacity premise, and the
    # payload carries that as a blocker rather than as a quantity.
    assert baseline["single_device_oom_observed"] is False
    assert baseline["blockers"] == [
        "measured single-device capacity failure is baseline provenance only"
    ]


def test_a_device_name_is_never_invented_for_a_run_that_did_not_record_one() -> None:
    measured = {
        "device_name": "NVIDIA A800-SXM4-80GB",
        "device_total_memory_bytes": 2048,
    }
    unnamed = {"device_total_memory_bytes": 85093777408}

    assert producer._baseline_device_identity(measured) == "NVIDIA A800-SXM4-80GB"
    identity = producer._baseline_device_identity(unnamed)
    assert "85093777408" in identity
    assert "NVIDIA" not in identity
    assert "not recorded" in identity


def test_the_single_device_failure_is_read_through_its_recorded_digest(
    tmp_path: Path,
) -> None:
    failure = _failure_record(tmp_path / "failure.json")
    telemetry = tmp_path / "gpu_samples.csv"
    premise = {
        "source_artifacts": [
            _source(failure, "single_gpu_failure"),
            {"kind": "gpu_samples", "path": str(telemetry), "sha256": "0" * 64},
        ]
    }

    record, unverifiable = producer._read_single_gpu_failure(premise, tmp_path)

    assert record["status"] == "cuda_oom"
    # The failure record is what the baseline facts come from, so it has to be
    # readable; the sources that are gone are reported rather than implied away.
    assert unverifiable == [str(telemetry)]

    gone = {"source_artifacts": [{"kind": "single_gpu_failure", "path": "gone.json"}]}
    with pytest.raises(SystemExit, match="is gone"):
        producer._read_single_gpu_failure(gone, tmp_path)

    drifted = {
        "source_artifacts": [
            {
                "kind": "single_gpu_failure",
                "path": str(failure),
                "sha256": "0" * 64,
            }
        ]
    }
    with pytest.raises(SystemExit, match="its digest does not match"):
        producer._read_single_gpu_failure(drifted, tmp_path)

    with pytest.raises(SystemExit, match="records no single-device failure"):
        producer._read_single_gpu_failure({"source_artifacts": []}, tmp_path)


def test_an_uncertified_premise_is_refused_before_it_is_inherited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(payload: object) -> None:
        raise MPSCapacityCertificationError("schema is not a general capacity claim")

    monkeypatch.setattr(producer, "require_general_mps_capacity", refuse)

    with pytest.raises(SystemExit, match="not a certified capacity premise"):
        producer._premise(
            _contract(),
            PREMISE_ARTIFACT,
            workload=producer._frozen_workload(_contract()),
        )


def test_a_premise_measured_at_another_shape_cannot_carry_this_workload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A premise for a different workload cannot make this one a capacity claim."""

    monkeypatch.setattr(producer, "require_general_mps_capacity", lambda payload: None)
    contract = _contract()
    workload = producer._frozen_workload(contract)
    payload = json.loads(PREMISE_ARTIFACT.read_text(encoding="utf-8"))

    def refuse(mutation: dict, message: str) -> None:
        candidate = copy.deepcopy(payload)
        candidate.update(mutation)
        path = tmp_path / "premise.json"
        path.write_text(json.dumps(candidate), encoding="utf-8")
        with pytest.raises(SystemExit, match=message):
            producer._premise(contract, path, workload=workload)

    refuse({"n_sites": 4096}, "sites where the frozen")
    refuse({"initial_max_bond": 512}, "maximum bond dimension")
    refuse({"world_size": 8}, "ranks where the frozen topology")


def test_a_premise_whose_failure_was_not_an_exhaustion_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A run that completed on one device did not establish a capacity premise."""

    monkeypatch.setattr(producer, "require_general_mps_capacity", lambda payload: None)
    contract = _contract()
    failure = _failure_record(tmp_path / "completed.json", status="completed")
    premise = {
        "n_sites": contract["n_sites"],
        "initial_max_bond": contract["trained_max_bond"],
        "world_size": contract["target_world_size"],
        "topology_fingerprint": str(
            producer._frozen_workload(contract).topology_fingerprint()
        ),
        "source_artifacts": [_source(failure, "single_gpu_failure")],
    }
    path = tmp_path / "premise.json"
    path.write_text(json.dumps(premise), encoding="utf-8")

    with pytest.raises(SystemExit, match="single-device exhaustion"):
        producer._premise(contract, path, workload=producer._frozen_workload(contract))


def test_a_premise_measured_at_another_topology_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The rank boundaries and bond schedule are part of the workload identity."""

    monkeypatch.setattr(producer, "require_general_mps_capacity", lambda payload: None)
    contract = _contract()
    payload = json.loads(PREMISE_ARTIFACT.read_text(encoding="utf-8"))
    payload["topology_fingerprint"] = "0" * 64
    path = tmp_path / "premise.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SystemExit, match="measured topology"):
        producer._premise(contract, path, workload=producer._frozen_workload(contract))


def test_the_checked_in_completion_artifact_resolves_as_the_capacity_premise() -> None:
    """The manifest digest-binds both the premise and the source-integrity record."""

    contract = _contract()
    premise = producer._premise(
        contract,
        PREMISE_ARTIFACT,
        workload=producer._frozen_workload(contract),
    )

    assert premise["sha256"] == CAPACITY["premise_provenance"]["artifact_sha256"]
    assert premise["topology_fingerprint"] == str(
        producer._frozen_workload(contract).topology_fingerprint()
    )
    baseline = premise["baseline"]
    assert (
        baseline["single_gpu_peak_memory_bytes"]
        == CAPACITY["measured_single_device_peak_memory_bytes"]
    )
    assert (
        baseline["single_gpu_device_total_memory_bytes"]
        == CAPACITY["single_device_total_memory_bytes"]
    )
    assert baseline["capacity_failure_reason"].startswith(FAILURE_REASON)
    # The checked-in run recorded no device name, so the premise states the
    # measured total memory instead of inventing a model for the hardware.
    assert "not recorded" in baseline["capacity_baseline_device"]
    # The raw log and the device telemetry are gone and the workload body on disk
    # has drifted from the digest the premise recorded, so three of the five
    # sources cannot be checked here and the premise says so rather than implying
    # that it re-verified them.
    integrity = premise["source_integrity"]
    assert integrity["finalized"] is True
    assert sorted(integrity["unverifiable_paths"]) == sorted(
        [
            *CAPACITY["premise_provenance"]["absent_sources"],
            CAPACITY["workload_definition_sources"][1]["path"],
        ]
    )


def test_the_completion_role_needs_the_measured_premise() -> None:
    """The sharded leg is a capacity claim only because one device failed."""

    with pytest.raises(SystemExit, match="requires --premise"):
        producer.main(
            [
                "--role",
                "capacity-completion",
                "--release-manifest",
                str(RELEASE_MANIFEST),
            ]
        )


def test_the_baseline_role_records_one_device_and_nothing_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")

    with pytest.raises(SystemExit, match="world size of one"):
        producer.main(
            ["--role", "capacity-failure", "--release-manifest", str(RELEASE_MANIFEST)]
        )


def test_the_baseline_role_refuses_to_report_a_run_without_the_measured_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A baseline from a host with no accelerator measured no exhaustion.

    The accelerator check is forced to fail rather than being skipped on, so the
    refusal is stated on every host instead of only on a host that happens to
    have no device. An inverted `skipif` would have made this the one test in the
    tree that a device lane could never reach, and a guard that skips when the
    hardware is present reads exactly like a pass.
    """

    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(SystemExit, match="needs the measured device"):
        producer.main(
            ["--role", "capacity-failure", "--release-manifest", str(RELEASE_MANIFEST)]
        )


def test_the_completion_role_refuses_a_world_that_does_not_span_hosts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The frozen topology is a two-host claim, not a single-host one."""

    monkeypatch.setenv("WORLD_SIZE", str(CAPACITY["target_world_size"]))

    with pytest.raises(SystemExit, match="span more than one host"):
        producer.main(
            [
                "--role",
                "capacity-completion",
                "--release-manifest",
                str(RELEASE_MANIFEST),
                "--premise",
                str(PREMISE_ARTIFACT),
                "--local-world-size",
                str(CAPACITY["target_world_size"]),
                "--node-count",
                "1",
            ]
        )


def test_the_completion_role_discloses_a_release_world_the_envelope_cannot_carry(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A world the envelope cannot describe is disclosed before it is measured.

    The frozen sixteen-rank world is carriable since API change proposal 065
    added `EvidenceScope.MULTI_NODE_SCALE`, so the predicate is forced false here:
    the branch is a live invariant over a derived question, not a statement about
    the checked-in manifest, and the value of the warning is that an operator
    learns before spending the cluster that the payload cannot be sealed. The
    accelerator check is forced to fail too, so the test stops at the warning on
    any host instead of entering a distributed launch.
    """

    monkeypatch.setenv("WORLD_SIZE", str(CAPACITY["target_world_size"]))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(producer, "envelope_carries_world", lambda world: False)

    with pytest.raises(SystemExit, match="needs the measured devices"):
        producer.main(
            [
                "--role",
                "capacity-completion",
                "--release-manifest",
                str(RELEASE_MANIFEST),
                "--premise",
                str(PREMISE_ARTIFACT),
                "--local-world-size",
                "8",
                "--node-count",
                "2",
            ]
        )

    assert (
        "release_world_size_not_carriable_by_evidence_envelope"
        in capsys.readouterr().err
    )


def test_the_speed_roles_are_refused_because_no_mps_timing_has_been_taken() -> None:
    """No ladder is frozen, so a timed comparison would be chosen after the fact."""

    assert producer.UNFROZEN_SPEED_ROLES == ("matched-speed", "speed-summary")
    assert FROZEN["speed_workload"]["configuration_ladder"] is None

    for role in producer.UNFROZEN_SPEED_ROLES:
        with pytest.raises(SystemExit, match="no matched-speed configuration ladder"):
            producer.main(["--role", role, "--release-manifest", str(RELEASE_MANIFEST)])


def test_the_release_payload_role_projects_the_contract_the_role_assembled(
    tmp_path: Path,
) -> None:
    contract = {"acceptance_case": "multi_gpu_capacity_completion", "world_size": 16}
    document = tmp_path / "role.json"
    document.write_text(json.dumps({"measurements": contract}), encoding="utf-8")
    output = tmp_path / "evidence.json"

    assert (
        producer.main(
            [
                "--role",
                "release-payload",
                "--document",
                str(document),
                "--measurements",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8")) == contract

    # A role document without a measurements block has nothing to project, and a
    # projection that silently produced an empty payload would seal a run as
    # evidence for a contract nobody assembled.
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"role": "capacity-completion"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="no measurements block"):
        producer.main(
            [
                "--role",
                "release-payload",
                "--document",
                str(empty),
                "--measurements",
                str(tmp_path / "never.json"),
            ]
        )


def test_the_launcher_hands_the_topology_and_the_shape_to_the_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The host layout and the frozen shape travel as recorded inputs."""

    captured: list[argparse.Namespace] = []

    def record(arguments: argparse.Namespace) -> int:
        captured.append(arguments)
        return 0

    monkeypatch.setattr(producer, "_role_capacity_completion", record)

    assert (
        producer.main(
            [
                "--role",
                "capacity-completion",
                "--premise",
                str(PREMISE_ARTIFACT),
                "--bond-dimension",
                str(CAPACITY["trained_max_bond"]),
                "--steps",
                str(CAPACITY["steps"]),
                "--local-world-size",
                "8",
                "--node-count",
                "2",
            ]
        )
        == 0
    )

    arguments = captured[0]
    assert arguments.release_manifest == GATE_MANIFEST
    assert arguments.premise == PREMISE_ARTIFACT
    assert arguments.bond_dimension == CAPACITY["trained_max_bond"]
    assert arguments.steps == CAPACITY["steps"]
    assert arguments.local_world_size == 8
    assert arguments.node_count == 2
