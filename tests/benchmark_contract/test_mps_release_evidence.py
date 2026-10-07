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
import importlib
import json
from pathlib import Path

import pytest
import torch

from benchmarks import mps_release_evidence as producer
from benchmarks.internal.evidence import mps_release_gate as gate
from benchmarks.internal.evidence.mps_release_gate import MANIFEST as GATE_MANIFEST
from benchmarks.internal.evidence.mps_shardable_ceiling import (
    shardable_ceiling_speedup,
)
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

    The contract is one literal dictionary bound to ``evidence`` and returned
    after the kind-specific fields are folded into it, so its keys can be read
    without building a run: this is what lets the producer be checked against the
    release manifest before a campaign spends a cluster on it. The keys folded in
    afterwards are read too, because a field stated only by one payload kind is
    still a field the contract states.
    """

    tree = ast.parse(Path(producer.__file__).read_text(encoding="utf-8"))
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_sharded_contract"
    )
    literals: list[ast.Dict] = []
    for node in ast.walk(function):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "evidence"
            and isinstance(node.value, ast.Dict)
        ):
            literals.append(node.value)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "update"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "evidence"
            and node.args
            and isinstance(node.args[0], ast.Dict)
        ):
            literals.append(node.args[0])
    if not literals:
        raise AssertionError("_sharded_contract states no evidence literal")
    return {
        key.value
        for literal in literals
        for key in literal.keys
        if isinstance(key, ast.Constant)
    }


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


def test_freezing_a_workload_leaves_the_importable_module_at_its_own_shape() -> None:
    """A frozen leg is built in a namespace of its own, not by patching a global.

    The shape the contract freezes lives in module constants, so a producer that
    patched the imported module would leave the last leg's shape behind for every
    later reader: a second leg at another shape would inherit the first one's
    constants, and an unrelated test that reads the module would see a workload
    nobody asked it for. The frozen leg is therefore executed from the verified
    file into a module object of its own, and the importable module is not
    touched.
    """

    module = "benchmarks.internal.evidence.general_mps_capacity_16"
    imported = importlib.import_module(module)
    before = (imported.N_SITES, imported.MAX_BOND, imported.TARGET_WORLD)
    frozen = (
        CAPACITY["n_sites"],
        CAPACITY["trained_max_bond"],
        CAPACITY["target_world_size"],
    )
    frozen_parameters = CAPACITY["parameter_count"]

    workload = producer._frozen_workload(_contract())

    after = (imported.N_SITES, imported.MAX_BOND, imported.TARGET_WORLD)
    built = (workload.N_SITES, workload.MAX_BOND, workload.TARGET_WORLD)
    assert after == before
    assert workload is not imported
    assert built == frozen
    assert workload.parameter_count() == frozen_parameters


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


def test_a_rank_site_run_is_stated_by_its_ends() -> None:
    """The run is the fact; the enumeration is the same fact written per site."""

    span = producer._site_span(range(8192, 16384))

    assert span == {"first_site": 8192, "last_site": 16383, "site_count": 8192}
    # A single site is a run of one rather than a special case.
    assert producer._site_span([7]) == {
        "first_site": 7,
        "last_site": 7,
        "site_count": 1,
    }


def test_a_rank_site_run_does_not_depend_on_the_order_it_was_measured_in() -> None:
    """The run is the set of sites; the measurement order is not part of it."""

    assert producer._site_span([9, 7, 8]) == {
        "first_site": 7,
        "last_site": 9,
        "site_count": 3,
    }


def test_a_rank_site_set_with_a_gap_is_reported_in_full() -> None:
    """Ends that rounded over a gap would be a smaller file saying something false."""

    span = producer._site_span([0, 1, 4])

    assert span["first_site"] == 0
    assert span["last_site"] == 4
    assert span["sites"] == [0, 1, 4]


def test_the_frozen_capacity_payload_states_every_rank_inside_the_hygiene_cap() -> None:
    """The map is present for every rank, and small enough to be committed.

    The gate requires a non-empty map, so the compact spelling has to keep every
    rank in it. The size is asserted rather than assumed because the enumeration
    this replaces is what took the assembled payload past the repository's
    two-million-byte per-file limit.
    """

    world = 16
    ownership = [list(range(rank * 8192, (rank + 1) * 8192)) for rank in range(world)]
    emitted = {
        f"rank:{rank}": producer._site_span(sites)
        for rank, sites in enumerate(ownership)
    }

    assert len(emitted) == world
    assert all(emitted.values())
    assert len(json.dumps(emitted)) < 4096


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
        parameter_count=CAPACITY["parameter_count"],
    )

    assert baseline["distribution_semantics"] == "single_device_fast_path"
    assert baseline["mps_forward_distribution_semantics"] == "single_device_fast_path"
    assert baseline["site_ownership"]["sharded"] is False
    # The circuit the device was exhausted by is named, not assumed: the digest
    # of the launcher is not enough to identify a parameterization.
    assert baseline["workload_body_sha256"] == producer._workload_body_digest(
        _contract()
    )
    assert baseline["parameter_count"] == CAPACITY["parameter_count"]
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
    # The premise's own reading is whatever the artifact it cites recorded. That
    # artifact was measured in a checkout this repository does not contain and
    # before the frozen body was redesigned, so the manifest discloses its peak
    # separately and freezes the reading the frozen body reaches where the
    # contract is measured now. An inherited number from another environment is
    # not a shape a freshly sealed baseline could reproduce.
    recorded_peak = CAPACITY["premise_provenance"][
        "recorded_single_gpu_peak_memory_bytes"
    ]
    assert baseline["single_gpu_peak_memory_bytes"] == recorded_peak
    assert recorded_peak != CAPACITY["measured_single_device_peak_memory_bytes"]
    assert (
        baseline["single_gpu_device_total_memory_bytes"]
        == CAPACITY["single_device_total_memory_bytes"]
    )
    assert baseline["capacity_failure_reason"].startswith(FAILURE_REASON)
    # The checked-in run recorded no device name, so the premise states the
    # measured total memory instead of inventing a model for the hardware.
    assert "not recorded" in baseline["capacity_baseline_device"]
    # The raw log and the device telemetry were recovered from the recording
    # host's /tmp, where the run's own recorded command line had written them,
    # and they hash to the digests the premise recorded while they were missing.
    # The workload body on disk has drifted from the digest the premise recorded,
    # so that one source cannot be checked here and the premise says so rather
    # than implying that it re-verified it.
    integrity = premise["source_integrity"]
    assert integrity["finalized"] is True
    assert CAPACITY["premise_provenance"]["absent_sources"] == []
    assert integrity["unverifiable_paths"] == [
        CAPACITY["workload_definition_sources"][1]["path"]
    ]


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


def _acceptance_rung() -> dict:
    speed = FROZEN["speed_workload"]
    return next(
        rung
        for rung in speed["configuration_ladder"]
        if rung["name"] == speed["acceptance_configuration"]
    )


def _configurations(seconds: float) -> list[dict]:
    """Return the timings one speed leg publishes for the whole frozen ladder.

    The samples are a real spread around the median rather than one repeated
    reading, because the confidence interval is resampled from them: a leg whose
    samples were constant would bracket a ratio with a degenerate interval that
    says nothing about the latency it measured.
    """

    spread = (1.0, 1.02, 0.98, 1.01, 0.99, 1.03, 0.97, 1.0, 1.02, 0.98, 1.01, 0.99)
    count = int(FROZEN["speed_workload"]["measured_steps"])
    samples = [round(seconds * factor, 6) for factor in spread[:count]]
    return [
        {
            "name": str(rung["name"]),
            "n_sites": int(rung["n_sites"]),
            "trained_max_bond": int(rung["trained_max_bond"]),
            "parameter_count": int(rung["parameter_count"]),
            "steps": 1,
            "warmup": int(FROZEN["speed_workload"]["warmup_steps"]),
            "iterations": count,
            "seconds": samples,
            "minimum_seconds": min(samples),
            "median_seconds": seconds,
            "logical_mps_bytes": 4096,
            "checkpoint_budget_bytes": 1 << 20,
            "peak_memory_bytes": 1024,
            "training_step_count": 1,
            "final_loss": 0.5,
        }
        for rung in FROZEN["speed_workload"]["configuration_ladder"]
    ]


def _boundaries(world: int) -> list[list[dict]]:
    """Return, for every rank, the adjacent-rank boundaries that rank took part in.

    A boundary is owned by two ranks and published by both, which is what makes a
    one-sided report the signature of a transport that never crossed the cut.
    """

    records = [
        {
            "owner_ranks": [rank, rank + 1],
            "bond": rank,
            "operation_id": f"bond:{rank}",
            "payload_bytes": 4096,
            "forward_transport": "batched_isend_irecv",
            "reverse_transport": "batched_isend_irecv",
        }
        for rank in range(world - 1)
    ]
    return [
        [dict(record) for record in records if rank in record["owner_ranks"]]
        for rank in range(world)
    ]


def _summary(rank: int, world: int) -> dict:
    """Return one rank's training summary in the shape the engine publishes it.

    The ownership map is the whole workload's, stated by every rank, and the one
    field that is a property of the reporting rank rather than of the workload --
    whether that rank holds optimizer state -- is set the way the engine sets it,
    to ``owner_rank == rank``. A fixture that made that field agree across ranks
    would be a fixture no real sharded run can produce.
    """

    ownership = [
        {
            "parameter_index": index,
            "owner_rank": index % world,
            "optimizer_state_local": (index % world) == rank,
            "gradient_route": "owner_reduce",
            "update_route": "owner_step",
        }
        for index in range(int(CAPACITY["parameter_count"]))
    ]
    metric = {
        "useful_work_completed": True,
        "bond_updates": _boundaries(world)[rank],
        "boundary_bytes": 4096 * max(0, world - 1),
        "gradient_collective_bytes": 0,
        "optimizer_collective_bytes": 0,
        "layer_halo_intra_node_bytes": 0,
        "layer_halo_inter_node_bytes": 0,
        "boundary_forward_exchanges": 1,
        "boundary_reverse_exchanges": 1,
        "peak_memory_bytes": 1024,
        "reverse_peak_memory_bytes": 512,
        "forward_peak_memory_bytes": 512,
        "end_to_end_seconds": 1.0,
        # The per-rank byte account the executor keeps while it builds the
        # reverse tape. The frozen workload is left-canonical and asks for no
        # canonicalization bond, so the sweep term is a measured zero rather than
        # an omission -- which is the shape the strict audit reads as
        # ``canonicalization_not_required``.
        "forward_tensor_bytes": 2048,
        "adjoint_tensor_bytes": 1024,
        "boundary_gradient_buffer_bytes": 256,
        "canonicalization_temporary_bytes": 0,
        "truncation_temporary_bytes": 0,
    }
    return {
        "executor": "pytorch_native_sharded_mps_training_v1",
        "optimizer": "adam",
        "rank": rank,
        "world_size": world,
        "local_world_size": 1,
        "site_ownership_policy": "balanced",
        "completed_steps": 1,
        "losses": [0.5],
        "site_ownership": tuple((index,) for index in range(world)),
        "optimizer_ownership": ownership,
        "step_metrics": [dict(metric)],
    }


def _speed_leg(*, world: int, seconds: float) -> dict:
    """Return one timed leg of the matched-speed ladder as the role records it.

    The two legs differ in the world they ran at, which is the whole comparison,
    and the sharded leg carries one record per rank because the ownership and
    timing evidence is read from every rank rather than from rank 0 alone.
    """

    configurations = _configurations(seconds)
    record = {
        "schema": "flagquantum.mps_release_measurements.v2",
        "role": "matched-speed",
        "rank": 0,
        "world_size": world,
        "local_world_size": 1,
        "node_count": world,
        "state_mode": "mps",
        "hostname": "bm-baai-dx-zone1-lc-a800-80g-15-172",
        "commit": "a" * 40,
        "collective_backend": "nccl",
        "warmup": int(FROZEN["speed_workload"]["warmup_steps"]),
        "iterations": int(FROZEN["speed_workload"]["measured_steps"]),
        "steps": 1,
        "ladder_fingerprint": str(FROZEN["speed_workload"]["ladder_fingerprint"]),
        "acceptance_configuration": str(
            FROZEN["speed_workload"]["acceptance_configuration"]
        ),
        "configurations": configurations,
        "acceptance_summary": _summary(0, world),
        "measured_peak_memory_bytes": 1024,
        "device_name": "NVIDIA A800-SXM4-80GB",
        "device_total_memory_bytes": int(CAPACITY["single_device_total_memory_bytes"]),
        "software": {"torch": "2.13.0+cu130"},
    }
    return {
        **record,
        "ranks": [
            {
                **record,
                "rank": index,
                "acceptance_summary": _summary(index, world),
            }
            for index in range(world)
        ],
    }


def test_the_assembled_speed_summary_states_the_whole_release_contract(
    tmp_path: Path,
) -> None:
    """The ratio the release publishes is assembled from two measured legs.

    The two legs are timed at the frozen ladder, so the document this role writes
    has to carry the ladder's identity, the acceptance rung's shape and the
    interval the ratio was bracketed by, on top of every field the capacity roles
    state. It is checked against the manifest's own required-field list rather
    than a list copied here, so a field added to the contract and not to this role
    fails this test.
    """

    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_speed_leg(world=1, seconds=4.0)), encoding="utf-8")
    sharded = tmp_path / "sharded.json"
    sharded.write_text(json.dumps(_speed_leg(world=2, seconds=2.0)), encoding="utf-8")
    summary = tmp_path / "summary.json"

    assert (
        producer.main(
            [
                "--role",
                "speed-summary",
                "--release-manifest",
                str(RELEASE_MANIFEST),
                "--premise",
                str(PREMISE_ARTIFACT),
                "--baseline",
                str(baseline),
                "--sharded",
                str(sharded),
                "--measurements",
                str(summary),
            ]
        )
        == 0
    )
    document = json.loads(summary.read_text(encoding="utf-8"))
    measurements = document["measurements"]

    assert document["state_mode"] == "mps"
    assert measurements["acceptance_case"] == "matched_speed"
    # Four of the required fields are read off the machine that ran the rank and
    # are carried by the envelope, so the requirement is taken from the gate
    # rather than restated here.
    envelope = {
        "devices": ("GPU-0000",),
        "raw_log_sha256": "a" * 64,
        "commit": "b" * 40,
        "rank_mapping": ("rank=0:device_uuid=GPU-0000",),
    }
    required_from_payload = set(
        gate._has_required_fields({}, envelope, FROZEN["required_artifact_fields"])
    )
    assert required_from_payload - set(measurements) == set()
    # The envelope names the ladder the leg timed and the evidence names the
    # capacity premise the rung rests on, which are two different digests.
    assert (
        measurements["workload_sha256"] == FROZEN["speed_workload"]["workload_sha256"]
    )
    assert (
        measurements["capacity_premise_workload_sha256"] == CAPACITY["workload_sha256"]
    )
    assert (
        measurements["ladder_fingerprint"]
        == FROZEN["speed_workload"]["ladder_fingerprint"]
    )
    assert measurements["acceptance_configuration"] == _acceptance_rung()["name"]
    # The acceptance rung's shape is the rung's, not the capacity workload's, and
    # it is a shape the ladder froze.
    assert measurements["n_sites"] == _acceptance_rung()["n_sites"]
    assert measurements["max_bond_dimension"] == _acceptance_rung()["trained_max_bond"]
    assert measurements["parameter_count"] == _acceptance_rung()["parameter_count"]
    assert measurements["n_sites"] != int(CAPACITY["n_sites"])
    # 4.0 s against 2.0 s over a world of two, from samples the interval is
    # resampled from, so a payload whose ratio drifted from its timings is caught.
    assert measurements["speedup"] == pytest.approx(2.0)
    # The efficiency is a ratio against the speedup the timed workload's own
    # arithmetic permits rather than against the world size, so at a world of two
    # the two denominators differ and the payload has to state which it divided
    # by, what the frozen fraction is, and what ceiling that fraction implies. The
    # world-size ratio is still reported beside it so the two stay comparable.
    serial_fraction = float(FROZEN["speed_workload"]["measured_serial_fraction"])
    ceiling = shardable_ceiling_speedup(serial_fraction, 2)
    assert measurements["scaling_efficiency"] == pytest.approx(2.0 / ceiling)
    assert measurements["linear_scaling_efficiency"] == pytest.approx(1.0)
    assert measurements["measured_serial_fraction"] == pytest.approx(serial_fraction)
    assert measurements["shardable_ceiling_speedup"] == pytest.approx(ceiling)
    assert measurements["scaling_efficiency_definition"] == (
        FROZEN["speed_workload"]["scaling_efficiency_definition"]
    )
    # Against this workload the two ratios are not interchangeable, which is the
    # whole reason the contract names one: 2.0 / 2 is the efficiency of a workload
    # that partitions perfectly, and this one does not.
    assert measurements["scaling_efficiency"] != pytest.approx(
        measurements["linear_scaling_efficiency"]
    )
    lower, upper = measurements["speedup_confidence_interval"]
    assert lower < measurements["speedup"] < upper
    # Every rung is reported with its own ratio, not only the acceptance one.
    assert [item["name"] for item in measurements["configurations"]] == [
        str(rung["name"]) for rung in FROZEN["speed_workload"]["configuration_ladder"]
    ]

    # The sealer reads the projected measurements as the envelope's evidence, so
    # the contract has to survive that projection unchanged.
    evidence = tmp_path / "evidence.json"
    assert (
        producer.main(
            [
                "--role",
                "release-payload",
                "--document",
                str(summary),
                "--measurements",
                str(evidence),
            ]
        )
        == 0
    )
    assert json.loads(evidence.read_text(encoding="utf-8")) == measurements

    # The distribution audit is a required release-tier command, so a payload the
    # release gate accepts while the audit refuses it is not releasable. The strict
    # audit reads the backward memory and backward communication plans through its
    # own predicates, and it reads the optimizer step through the ownership
    # semantics rather than through the ownership maps the gate already checks, so
    # the three of them are checked here through those predicates. A producer
    # change that stopped stating one of them then fails this test rather than the
    # promotion it would otherwise have reached.
    from flagquantum.runtime.audit.mps_readiness import (
        _mps_communication_plan_reported,
        _mps_memory_plan_reported,
    )

    sharded_leg = _speed_leg(world=2, seconds=2.0)
    world = int(measurements["world_size"])
    memory_plan = measurements["mps_backward_memory_plan"]
    communication_plan = measurements["mps_backward_communication_plan"]
    assert _mps_memory_plan_reported(memory_plan, world_size=world)
    assert _mps_communication_plan_reported(communication_plan)
    assert measurements["optimizer_update_semantics"] == "sharded_across_ranks"
    assert measurements["optimizer_update_ownership_semantics"] == (
        "sharded_across_ranks"
    )
    # Every vector in the plan is the rank's own recorded counter rather than a
    # number restated from the workload, so the plan is a reading of the run.
    steps = [
        record["acceptance_summary"]["step_metrics"][-1]
        for record in sharded_leg["ranks"]
    ]
    assert memory_plan["forward_tensor_bytes_by_rank"] == [
        metric["forward_tensor_bytes"] for metric in steps
    ]
    assert memory_plan["backward_adjoint_bytes_by_rank"] == [
        metric["adjoint_tensor_bytes"] for metric in steps
    ]
    assert memory_plan["per_rank_peak_bytes"] == [
        metric["reverse_peak_memory_bytes"] for metric in steps
    ]
    # The plan states the sweep term it did not need rather than omitting it, and
    # says so in the two places the audit reads: the plan-level flag and the
    # per-rank flag.
    assert memory_plan["canonicalization_required"] is False
    assert all(
        item["canonicalization_not_required"] is True
        for item in memory_plan["rank_memory"]
    )
    # Each measured boundary is described by the ranks and bytes it was measured
    # with, so the audit can place it on a host rather than trusting a label.
    assert communication_plan["boundary_edge_count"] == len(
        communication_plan["boundary_edges"]
    )
    assert all(
        edge["communication_bytes"] == edge["payload_bytes"]
        and edge["right_rank"] == edge["left_rank"] + 1
        and edge["topology_tier"] in {"intra_node", "inter_node"}
        for edge in communication_plan["boundary_edges"]
    )


def test_a_speed_summary_refuses_legs_that_are_not_one_comparison(
    tmp_path: Path,
) -> None:
    """A ratio is only a ratio of one ladder, one transport and two real legs."""

    # The baseline leg is the world size 1 run the comparison divides by, so the
    # refusals below are about the sharded leg unless one of them says otherwise.
    solo = _speed_leg(world=1, seconds=4.0)

    def leg(**patches) -> dict:
        return {**_speed_leg(world=2, seconds=2.0), **patches}

    def run(baseline: dict, sharded: dict) -> None:
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
        sharded_path = tmp_path / "sharded.json"
        sharded_path.write_text(json.dumps(sharded), encoding="utf-8")
        producer.main(
            [
                "--role",
                "speed-summary",
                "--release-manifest",
                str(RELEASE_MANIFEST),
                "--premise",
                str(PREMISE_ARTIFACT),
                "--baseline",
                str(baseline_path),
                "--sharded",
                str(sharded_path),
                "--measurements",
                str(tmp_path / "never.json"),
            ]
        )

    # A denominator measured on one device is not a distributed speedup.
    with pytest.raises(SystemExit, match="sharded leg must run above world size 1"):
        run(solo, leg(world_size=1, node_count=1))
    # A leg timed at another protocol measured a comparison nobody froze.
    with pytest.raises(SystemExit, match="timed a ladder that is not the frozen one"):
        run(solo, leg(ladder_fingerprint="0" * 64))
    # A ratio across two transports is not a ratio of either of them.
    with pytest.raises(SystemExit, match="different collectives"):
        run(solo, leg(collective_backend="gloo"))
    # Both legs have to have timed the frozen ladder's own configurations.
    with pytest.raises(SystemExit, match="did not time the frozen ladder"):
        incomplete = leg()
        incomplete["configurations"] = incomplete["configurations"][1:]
        run(solo, incomplete)
    # A ratio needs a denominator that is a latency.
    with pytest.raises(SystemExit, match="not a latency a ratio can divide by"):
        zero = leg()
        for item in zero["configurations"]:
            item["median_seconds"] = 0.0
        run(solo, zero)
    # Every rank has to have measured something, or the world size the payload
    # states is not the number of ranks that ran.
    with pytest.raises(SystemExit, match="published .* rank records"):
        short = leg()
        short["ranks"] = short["ranks"][:1]
        run(solo, short)
    # Optimizer state is held by the rank that owns the parameter, and a rank
    # that states otherwise is reporting an optimizer that is not sharded.
    with pytest.raises(SystemExit, match="owns; the optimizer state"):
        drifted = leg()
        for entry in drifted["ranks"][1]["acceptance_summary"]["optimizer_ownership"]:
            entry["optimizer_state_local"] = True
        run(solo, drifted)


def test_the_frozen_ladder_is_the_protocol_the_manifest_declares() -> None:
    """The rungs the manifest lists are the rungs the ladder definition fixes.

    Two copies of one protocol is a drift hazard: the gate reads the manifest so
    that it never executes a workload, and the producer executes the definition,
    so the producer is where the two can be compared. The fingerprint is checked
    as well, because it is what the sealer's measured payloads name: a rung list
    that agreed while the fingerprint moved would let a payload attest to a
    protocol the manifest no longer describes.
    """

    speed = FROZEN["speed_workload"]
    module = producer._load_ladder_module(speed)

    assert producer._speed_ladder(speed) == [
        {
            "name": rung["name"],
            "n_sites": rung["n_sites"],
            "trained_max_bond": rung["trained_max_bond"],
            "parameter_count": FROZEN["capacity_workload"]["parameter_count"],
        }
        for rung in module.LADDER
    ]
    assert speed["acceptance_configuration"] == module.ACCEPTANCE
    assert producer._ladder_steps(speed) == module.STEPS
    assert speed["ladder_fingerprint"] == module.ladder_fingerprint()
    assert speed["workload_sha256"] == producer._sha256(
        Path(speed["ladder_definition_path"])
    )
    acceptance = next(
        rung
        for rung in speed["configuration_ladder"]
        if rung["name"] == speed["acceptance_configuration"]
    )
    assert (
        acceptance["parameter_count"] == FROZEN["capacity_workload"]["parameter_count"]
    )


def test_a_speed_role_refuses_a_protocol_the_manifest_did_not_freeze(
    tmp_path: Path,
) -> None:
    """A leg timed at another protocol measured a comparison nobody froze.

    The warmup and the measured call count are protocol rather than tuning: a
    cold rank and a warmed one measure different latencies, so a ratio computed
    across two different protocols is not the frozen acceptance measurement. The
    refusal comes before any device is touched, which is what makes it a launch
    check rather than a post-hoc one.
    """

    for warmup, iterations in ((0, 10), (2, 1)):
        with pytest.raises(SystemExit, match="the frozen matched-speed protocol"):
            producer.main(
                [
                    "--role",
                    "matched-speed",
                    "--release-manifest",
                    str(RELEASE_MANIFEST),
                    "--warmup",
                    str(warmup),
                    "--iterations",
                    str(iterations),
                ]
            )

    # A manifest whose ladder module is not the one it froze cannot be timed at
    # all: the rungs the leg would run are not the rungs the gate reads.
    unfrozen = copy.deepcopy(FROZEN)
    unfrozen["speed_workload"]["ladder_definition_path"] = str(
        tmp_path / "not_the_ladder.py"
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(unfrozen), encoding="utf-8")
    with pytest.raises(SystemExit, match="is not on disk"):
        producer.main(
            [
                "--role",
                "matched-speed",
                "--release-manifest",
                str(manifest),
                "--warmup",
                str(FROZEN["speed_workload"]["warmup_steps"]),
                "--iterations",
                str(FROZEN["speed_workload"]["measured_steps"]),
            ]
        )


def test_the_speed_summary_role_requires_both_legs_and_the_premise() -> None:
    """A ratio is not derivable from one leg, and a premise is not inferable."""

    with pytest.raises(SystemExit, match="requires --premise"):
        producer.main(
            ["--role", "speed-summary", "--release-manifest", str(RELEASE_MANIFEST)]
        )

    with pytest.raises(SystemExit, match="requires --baseline and --sharded"):
        producer.main(
            [
                "--role",
                "speed-summary",
                "--release-manifest",
                str(RELEASE_MANIFEST),
                "--premise",
                str(PREMISE_ARTIFACT),
            ]
        )


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


def test_the_timed_leg_exchanges_records_before_it_tears_the_group_down(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The ranks report what they timed while there is still a group to report on.

    Every rank times every rung, so the leg hands its own record to a gloo
    subgroup and reads rank 0's copy back. That exchange is a collective, and a
    collective after ``destroy_process_group`` has no default group to build its
    subgroup on: the leg died with ``Default process group has not been
    initialized`` from ``new_group``, and reported it as a matched-speed
    measurement failure. The order of these two events is therefore part of what
    the role has to get right, and it is observed here rather than inferred from
    the file the role happened to write.
    """

    events: list[str] = []

    class _Dist:
        @staticmethod
        def is_initialized() -> bool:
            return True

        @staticmethod
        def get_backend() -> str:
            return "nccl"

        @staticmethod
        def new_group(backend: str) -> object:
            events.append(f"new_group:{backend}")
            return object()

        @staticmethod
        def gather_object(value: object, gathered: list | None, **_: object) -> None:
            events.append("gather_object")
            if gathered is not None:
                gathered[:] = [dict(value), dict(value)]

        @staticmethod
        def destroy_process_group(group: object = None) -> None:
            events.append("destroy_process_group")

    class _Rung:
        N_SITES = 512
        MAX_BOND = 64

        @staticmethod
        def logical_mps_bytes(n_sites: int, max_bond: int) -> int:
            return 32_855_360

        @staticmethod
        def reverse_checkpoint_capacity_bytes(
            logical_bytes: int, world_size: int
        ) -> int:
            return 2 * (logical_bytes // world_size) + (1 << 30)

        @staticmethod
        def rank_owned_initial_mps(n_sites: int, max_bond: int, device: object) -> dict:
            return {}

        @staticmethod
        def frozen_parameters(device: object) -> dict:
            return {}

        @staticmethod
        def workload(parameters: dict) -> object:
            return object()

    class _Training:
        @staticmethod
        def summary() -> dict:
            return {"completed_steps": 1, "losses": [0.5]}

    class _Device:
        name = "NVIDIA A800-SXM4-80GB"
        total_memory = 85_093_777_408

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(producer, "dist", _Dist)
    monkeypatch.setattr(producer, "_initialize", lambda device: None)
    monkeypatch.setattr(producer, "_commit", lambda: "a" * 40)
    monkeypatch.setattr(producer, "_frozen_workload_at", lambda *a, **k: _Rung)
    monkeypatch.setattr(
        producer,
        "_speed_ladder",
        lambda speed: [
            {
                "name": FROZEN["speed_workload"]["acceptance_configuration"],
                "n_sites": 512,
                "trained_max_bond": 64,
                "parameter_count": 31,
            }
        ],
    )
    monkeypatch.setattr(
        producer.fqxd,
        "train_distributed_mps",
        lambda *a, **k: _Training(),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 1024)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: _Device())

    output = tmp_path / "mps_speed_pair.json"
    assert (
        producer.main(
            [
                "--role",
                "matched-speed",
                "--release-manifest",
                str(RELEASE_MANIFEST),
                "--warmup",
                str(FROZEN["speed_workload"]["warmup_steps"]),
                "--iterations",
                str(FROZEN["speed_workload"]["measured_steps"]),
                "--local-world-size",
                "2",
                "--node-count",
                "1",
                "--measurements",
                str(output),
            ]
        )
        == 0
    )

    assert events == [
        "new_group:gloo",
        "gather_object",
        "destroy_process_group",
        "destroy_process_group",
    ]
    written = json.loads(output.read_text(encoding="utf-8"))
    assert len(written["ranks"]) == 2


def test_the_premise_artifact_is_named_the_way_the_checkout_names_it():
    """A promoted payload must name a checked-in artifact a reader can open.

    The completion role is launched from the scratch checkout the measurement was
    taken in, and ``--premise`` is passed to it as an absolute path there. The
    projection records that argument as the payload's capacity baseline artifact,
    so without this the release payload names a checked-in artifact by a path that
    exists on the producing host and nowhere else -- and two payloads promoted
    from the same release would name the same repository file two different ways.
    A path that genuinely lies outside the repository is left alone, because then
    there is no repository name for it and a relative one would name nothing.
    """

    inside = producer.REPO_ROOT / "benchmarks/results/local/example.json"
    assert producer._repo_relative(inside) == "benchmarks/results/local/example.json"
    assert producer._repo_relative(Path("/tmp/example.json")) == "/tmp/example.json"
