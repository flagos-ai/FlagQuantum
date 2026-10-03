"""The release campaign's payload shape is machine-checked, not asserted in prose.

A release payload is only worth what the strict audit says it is worth, so these
tests take the documents the campaign runner actually emits, seal them exactly as
``tools/seal_runtime_evidence.py`` does, and require the sealed result to clear
both the distributed claim audit and the ISSUE-044 release gate. A change to the
runner that produced a payload the audit rejects fails here rather than on two
borrowed hosts.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from benchmarks.internal.evidence.statevector_release_gate import (
    evaluate_issue044_release,
    load_manifest,
)
from benchmarks.statevector_release_evidence import (
    _agreed_ownership,
    _bootstrap_ratio_interval,
    _capacity_circuit,
    _control_group,
    _gather,
    _production_fields,
    _role_speed_summary,
)
from flagquantum.runtime.audit.release_policy import validate_distributed_claim_evidence
from flagquantum.runtime.observability.evidence import (
    ArtifactClass,
    EvidenceScope,
    RuntimeProvenance,
    create_evidence_artifact,
)

pytestmark = [pytest.mark.benchmark_contract, pytest.mark.release_gate]

KEY = b"issue-044-campaign-test-key"
DEVICE = "NVIDIA A800-SXM4-80GB"


def _rank_record(rank: int, world: int, *, peak: int, steps: int = 2) -> dict:
    """One rank's record in the shape the campaign runner gathers."""

    return {
        "rank": rank,
        "world_size": world,
        "device_name": DEVICE,
        "measured_peak_memory_bytes": peak,
        "timings": [1.0, 1.1, 1.2, 1.3, 1.4],
        "measurements": {
            "completed_steps": steps,
            "communication_bytes": 4096 * world,
            "communication_events": 10 * world,
            "losses": [0.5, 0.25],
            "parameter_ownership_semantics": "sharded_across_ranks",
            "gradient_ownership_semantics": "sharded_across_ranks",
            "optimizer_update_ownership_semantics": "sharded_across_ranks",
            "optimizer_ownership": [
                {"parameter": f"parameter:{index}", "owner_rank": index % world}
                for index in range(world)
            ],
        },
    }


def _records(world: int) -> list[dict]:
    return [_rank_record(rank, world, peak=8 * 1024**3) for rank in range(world)]


def _provenance(world: int, *, warmup: int = 5, iterations: int = 30):
    return RuntimeProvenance(
        commit="a" * 40,
        workload_sha256=hashlib.sha256(
            Path(
                "benchmarks/manifests/statevector_capacity_workload_v2.json"
            ).read_bytes()
        ).hexdigest(),
        command=(
            "torchrun",
            "--nnodes=2",
            "benchmarks/statevector_release_evidence.py",
        ),
        devices=tuple(f"GPU-{rank}" for rank in range(world)),
        topology="NV12",
        rank_mapping=tuple(
            f"rank={rank}:node=host-{rank % 2}:device_uuid=GPU-{rank}"
            for rank in range(world)
        ),
        collective_backend="nccl",
        warmup=warmup,
        iterations=iterations,
        seeds=(440044,),
        raw_log_sha256="c" * 64,
        fallback_events=(),
    )


def _seal(
    evidence: dict,
    *,
    world: int,
    scope: EvidenceScope,
    release: bool,
    provenance_world: int | None = None,
):
    sealed = dict(evidence)
    if release:
        sealed.update(
            {
                "claim_evidence_type": "production_training_benchmark",
                "release_payload": True,
                "scalability_claim_allowed": True,
                "release_gate_allowed": True,
            }
        )
    artifact = create_evidence_artifact(
        artifact_class=ArtifactClass.MEASURED_PRODUCTION_RUN,
        evidence_scope=scope,
        provenance=_provenance(provenance_world or world),
        evidence=sealed,
        signing_key=KEY,
    )
    return artifact.summary()


def _capacity_completion(world: int, local_world_size: int) -> dict:
    manifest = load_manifest()
    digest = manifest["capacity_workload"]["workload_sha256"]
    return _production_fields(
        records=_records(world),
        capacity={"workload_sha256": digest},
        node_count=max(1, world // local_world_size),
        world=world,
        local_world_size=local_world_size,
        acceptance_case="multi_gpu_capacity_completion",
        seconds=[1.0, 1.1, 1.2, 1.3, 1.4],
        extra={"workload_name": "issue044_single_node_capacity_v2"},
    )


@pytest.mark.parametrize("world,local", [(2, 1), (4, 2), (8, 4)])
def test_capacity_completion_payload_clears_the_distributed_claim_audit(world, local):
    """The measured document must satisfy the audit once sealed.

    The two-host pairs are the point: a payload that reports a sharded world but
    cannot show multi-node production transport would be rejected here.
    """

    artifact = _seal(
        _capacity_completion(world, local),
        world=world,
        scope=(
            EvidenceScope.SCHEDULED_SCALE
            if world in (4, 8)
            else EvidenceScope.TWO_GPU_SEMANTIC
        ),
        release=True,
    )

    result = validate_distributed_claim_evidence(artifact["evidence"])

    assert result.errors == ()
    assert artifact["evidence"]["release_gate_allowed"] is True


def test_a_matched_speed_payload_states_the_capacity_premise_it_rests_on(tmp_path):
    """A release payload has to name the premise, not describe another workload.

    The matched-speed workload fits on one device by design, so a payload that
    described its own workload under ``capacity_failure_reason`` would claim a
    capacity failure that never happened. The premise is the frozen capacity
    workload, and the payload has to say so.
    """

    manifest = load_manifest()
    premise = manifest["capacity_workload"]["workload_sha256"]
    artifact = _speed_summary(tmp_path)
    evidence = artifact["evidence"]

    assert evidence["acceptance_case"] == "matched_speed"
    assert evidence["single_gpu_expected_oom"] is True
    assert evidence["capacity_premise_workload_sha256"] == premise
    assert premise in evidence["capacity_failure_reason"]
    assert evidence["workload_sha256"] != premise
    assert validate_distributed_claim_evidence(evidence).release_gate_allowed is True


def test_a_replicated_backward_publishes_no_gradient_ownership(tmp_path) -> None:
    """A payload may not claim ownership the measured reduction did not create.

    ``owner_reduce`` and ``reduce_scatter_sum`` leave each gradient on one rank;
    a replicated backward leaves every rank holding the whole gradient. The
    audit reads a non-empty gradient map as the evidence that the reduction was
    owner-scoped, so a leg that measured a replicated reduction must publish the
    semantics it measured and an empty map, and the release then fails closed.
    """

    artifact = _speed_summary(tmp_path, gradient_semantics="replicated_across_ranks")
    evidence = artifact["evidence"]

    assert evidence["gradient_ownership_semantics"] == "replicated_across_ranks"
    assert evidence["gradient_ownership"] == {}
    assert evidence["parameter_ownership"] != {}
    assert evidence["optimizer_update_ownership"] != {}
    audit = validate_distributed_claim_evidence(evidence)
    assert audit.release_gate_allowed is False


def test_a_matched_speed_payload_publishes_the_runtime_ownership_records(
    tmp_path,
) -> None:
    """The maps are the runtime's per-parameter records, rank by rank."""

    evidence = _speed_summary(tmp_path)["evidence"]

    assert evidence["parameter_ownership_semantics"] == "sharded_across_ranks"
    assert evidence["gradient_ownership_semantics"] == "sharded_across_ranks"
    assert set(evidence["parameter_ownership"]) == {"rank:0", "rank:1"}
    assert evidence["parameter_ownership"]["rank:0"] == ["parameter:0", "parameter:2"]
    for key in (
        "parameter_ownership",
        "gradient_ownership",
        "optimizer_update_ownership",
    ):
        owned = sorted(name for names in evidence[key].values() for name in names)
        assert owned == sorted(f"parameter:{index}" for index in range(4)), key


def test_ranks_that_disagree_about_ownership_produce_no_payload() -> None:
    """A map is only evidence if every rank of the leg published the same one.

    The runtime states the whole parameter-to-owner map on every rank, so a
    payload that merged the ranks would list each parameter once per rank, and a
    payload that trusted rank 0 alone would publish a map the rest of the leg
    contradicts. Both are refused rather than sealed.
    """

    records = [
        {
            "rank": rank,
            "measurements": {
                "optimizer_ownership": [
                    {
                        "parameter": f"parameter:{index}",
                        "owner_rank": (index + rank) % 2,
                    }
                    for index in range(4)
                ]
            },
        }
        for rank in range(2)
    ]

    with pytest.raises(SystemExit, match="disagree about which rank owns"):
        _agreed_ownership(
            [list(record["measurements"]["optimizer_ownership"]) for record in records],
            context="acceptance case 'matched_speed'",
        )


def test_a_capacity_completion_payload_names_the_same_premise() -> None:
    """Both payload kinds name one premise, so the release reads consistently."""

    manifest = load_manifest()
    evidence = _capacity_completion(4, 2)

    assert evidence["capacity_premise_workload_sha256"] == (
        manifest["capacity_workload"]["workload_sha256"]
    )
    assert evidence["workload_sha256"] == evidence["capacity_premise_workload_sha256"]


def test_a_matched_speed_payload_reports_the_communication_its_call_measured(
    tmp_path,
):
    """The bytes are read back from the timed call, not restated as a constant."""

    manifest = load_manifest()
    acceptance = manifest["speed_workload"]["acceptance_configuration"]
    evidence = _speed_summary(tmp_path)["evidence"]

    assert evidence["communication_bytes"] > 0
    assert len(evidence["communication_bytes_by_rank"]) == evidence["world_size"]
    assert evidence["communication_bytes"] == max(
        evidence["communication_bytes_by_rank"]
    )
    assert "rank-local collective total" in evidence["communication_scope"]
    assert evidence["configurations"][-1]["name"] == acceptance


def test_a_matched_speed_leg_that_measured_no_communication_reports_none(tmp_path):
    """A leg with no measured counter reports zero instead of a placeholder.

    The transport audit asks whether communication evidence was *reported*
    rather than how much of it there was, so this honesty lives in the producer:
    the payload must not carry a byte count that no timed call produced.
    """

    manifest = load_manifest()
    configurations = [
        {
            "name": manifest["speed_workload"]["acceptance_configuration"],
            "n_wires": 24,
            "depth": 8,
            "steps": 4,
            "parameter_count": 192,
            "seconds": [0.4, 0.42, 0.44, 0.46, 0.48],
            "minimum_seconds": 0.4,
            "median_seconds": 0.44,
            "optimizer_ownership": [
                {"parameter": f"parameter:{index}", "owner_rank": index % 2}
                for index in range(4)
            ],
            "parameter_ownership_semantics": "sharded_across_ranks",
            "gradient_ownership_semantics": "sharded_across_ranks",
            "optimizer_update_ownership_semantics": "sharded_across_ranks",
        }
    ]
    sharded = {
        "world_size": 2,
        "local_world_size": 1,
        "node_count": 2,
        "commit": "a" * 40,
        "device_name": DEVICE,
        "warmup": manifest["speed_workload"]["warmup_steps"],
        "iterations": manifest["speed_workload"]["measured_steps"],
        "workload_sha256": manifest["speed_workload"]["workload_sha256"],
        "software": {"torch": "2.13.0", "cuda": "13.0", "python": "3.12"},
        "ranks": [
            {
                "measured_peak_memory_bytes": 4 * 1024**3,
                "timings": [0.1],
                "measurements": {"configurations": copy.deepcopy(configurations)},
            },
            {
                "measured_peak_memory_bytes": 4 * 1024**3,
                "timings": [0.1],
                "measurements": {"configurations": copy.deepcopy(configurations)},
            },
        ],
        "measurements": {"configurations": configurations},
    }
    baseline = copy.deepcopy(sharded)
    baseline["world_size"] = 1
    baseline["node_count"] = 1
    baseline["ranks"] = [sharded["ranks"][0]]
    for item in baseline["measurements"]["configurations"]:
        item["seconds"] = [value * 2 for value in item["seconds"]]
        item["median_seconds"] *= 2
    args = SimpleNamespace(
        role="speed-summary",
        release_manifest=Path("benchmarks/manifests/statevector_release_v2.json"),
        baseline=_write_leg(tmp_path, "baseline", baseline),
        sharded=_write_leg(tmp_path, "sharded", sharded),
        measurements=tmp_path / "summary.json",
    )

    assert _role_speed_summary(args) == 0
    document = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    measurements = document["measurements"]

    assert document["world_size"] == 2
    assert measurements["communication_bytes"] == 0
    assert measurements["communication_bytes_by_rank"] == [0, 0]
    assert measurements["local_state_bytes_by_rank"] == [0, 0]
    assert measurements["inter_node_communication_bytes"] == 0
    assert measurements["communication_fraction"] == 0.0


def test_payload_without_a_multi_node_route_is_not_a_release_payload():
    """Declaring node_count is not transport evidence; the route has to be shown."""

    evidence = _capacity_completion(4, 2)
    evidence["node_count"] = 2
    del evidence["rank_placement"]
    del evidence["inter_node_communication_bytes"]
    artifact = _seal(
        evidence, world=4, scope=EvidenceScope.SCHEDULED_SCALE, release=True
    )

    result = validate_distributed_claim_evidence(artifact["evidence"])

    assert result.errors != ()
    assert result.release_gate_allowed is False


def test_full_campaign_of_sealed_artifacts_passes_the_issue044_gate(tmp_path):
    """Baseline outside the promoted set, three worlds inside it, and a ratio."""

    baseline_measurements = {
        "acceptance_case": "single_gpu_capacity_failure",
        "world_size": 1,
        "node_count": 1,
        "distribution_semantics": "single_device_fast_path",
        "single_device_oom_observed": True,
        "capacity_baseline_device": f"cuda:0 on {DEVICE}",
        "capacity_failure_reason": "CUDA out of memory. Tried to allocate 128.00 GiB",
        "measured_peak_memory_bytes": 68728000512,
        "full_state_materialized": True,
        "workload_sha256": load_manifest()["capacity_workload"]["workload_sha256"],
        "hardware_inventory": [DEVICE],
    }
    baseline = _seal(
        baseline_measurements,
        world=1,
        scope=EvidenceScope.ONE_GPU_LOCAL,
        release=False,
    )
    speed = _speed_summary(tmp_path)
    promoted = [
        speed,
        _seal(
            _capacity_completion(4, 2),
            world=4,
            scope=EvidenceScope.SCHEDULED_SCALE,
            release=True,
        ),
        _seal(
            _capacity_completion(8, 4),
            world=8,
            scope=EvidenceScope.SCHEDULED_SCALE,
            release=True,
        ),
    ]

    passed, blockers = evaluate_issue044_release(
        [baseline, *promoted], load_manifest(), signing_key=KEY
    )

    assert blockers == ()
    assert passed is True


def _speed_summary(
    tmp_path: Path, *, gradient_semantics: str = "sharded_across_ranks"
) -> dict:
    """Drive the frozen speed summary over synthetic legs and seal its output.

    The two legs carry deliberately different timings so the ratio is not a
    comparison of a run with itself, and both carry the same configuration list
    because the manifest's three shapes are what the ratio is allowed to cover.
    """

    manifest = load_manifest()
    acceptance = manifest["speed_workload"]["acceptance_configuration"]
    # The runtime publishes one ownership record per parameter. The payload
    # restates those records, so a synthetic leg has to carry them: this is the
    # evidence the release audit reads to accept that the update was sharded.
    ownership = [
        {
            "parameter": f"parameter:{index}",
            "owner_rank": index % 2,
            "gradient_reduction": "reduce_scatter_sum",
            "optimizer_state_local": index % 2 == 0,
            "update_route": "owner_step_then_broadcast",
        }
        for index in range(4)
    ]
    configurations = [
        {
            "name": f"matched_speed_{index}",
            "n_wires": 24 + 2 * index,
            "depth": 8 * 2**index,
            "steps": 4,
            "parameter_count": (24 + 2 * index) * 8 * 2**index,
            "seconds": [0.4, 0.42, 0.44, 0.46, 0.48],
            "minimum_seconds": 0.4,
            "median_seconds": 0.44,
            "communication_bytes": 4096 * (index + 1),
            "communication_events": 8,
            "completed_steps": 4,
            "local_state_bytes": 8 * 2**24 // 2,
            "communication_fraction": (4096 * (index + 1))
            / (4096 * (index + 1) + 8 * 2**24 // 2),
            "optimizer_ownership": copy.deepcopy(ownership),
            "parameter_ownership_semantics": "sharded_across_ranks",
            "gradient_ownership_semantics": gradient_semantics,
            "optimizer_update_ownership_semantics": "sharded_across_ranks",
        }
        for index in range(3)
    ]
    configurations[-1]["name"] = acceptance
    sharded_leg = {
        "world_size": 2,
        "local_world_size": 1,
        "node_count": 2,
        "commit": "a" * 40,
        "device_name": DEVICE,
        "warmup": manifest["speed_workload"]["warmup_steps"],
        "iterations": manifest["speed_workload"]["measured_steps"],
        "workload_sha256": manifest["speed_workload"]["workload_sha256"],
        "software": {"torch": "2.13.0", "cuda": "13.0", "python": "3.12"},
        "ranks": [
            {
                "measured_peak_memory_bytes": 4 * 1024**3,
                "timings": [0.1, 0.11, 0.12],
                "measurements": {"configurations": copy.deepcopy(configurations)},
            }
            for _ in range(2)
        ],
        "measurements": {"configurations": configurations},
    }
    baseline_leg = copy.deepcopy(sharded_leg)
    baseline_leg["world_size"] = 1
    baseline_leg["local_world_size"] = 1
    baseline_leg["node_count"] = 1
    baseline_leg["ranks"] = [sharded_leg["ranks"][0]]
    for item in baseline_leg["measurements"]["configurations"]:
        item["seconds"] = [value * 2 for value in item["seconds"]]
        item["minimum_seconds"] *= 2
        item["median_seconds"] *= 2

    baseline_path = _write_leg(tmp_path, "baseline", baseline_leg)
    sharded_path = _write_leg(tmp_path, "sharded", sharded_leg)
    output = tmp_path / "summary.json"
    args = SimpleNamespace(
        role="speed-summary",
        release_manifest=Path("benchmarks/manifests/statevector_release_v2.json"),
        baseline=baseline_path,
        sharded=sharded_path,
        measurements=output,
    )

    assert _role_speed_summary(args) == 0
    document = json.loads(output.read_text(encoding="utf-8"))
    return _seal(
        document["measurements"],
        world=2,
        scope=EvidenceScope.TWO_GPU_SEMANTIC,
        release=True,
        provenance_world=2,
    )


def _write_leg(tmp_path: Path, name: str, payload: dict) -> Path:
    path = tmp_path / f"issue044-leg-{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_speed_summary_refuses_a_baseline_that_is_not_single_device(tmp_path):
    """A ratio whose denominator is already sharded measures nothing."""

    leg = {"world_size": 4, "measurements": {"configurations": []}}
    baseline = tmp_path / "baseline.json"
    sharded = tmp_path / "sharded.json"
    baseline.write_text(json.dumps(leg), encoding="utf-8")
    sharded.write_text(json.dumps(leg), encoding="utf-8")
    args = SimpleNamespace(
        role="speed-summary",
        release_manifest=Path("benchmarks/manifests/statevector_release_v2.json"),
        baseline=baseline,
        sharded=sharded,
        measurements=tmp_path / "summary.json",
    )

    with pytest.raises(SystemExit, match="must be a world size 1 run"):
        _role_speed_summary(args)


def test_bootstrap_interval_orders_the_paired_legs():
    """Sampling the ratio keeps the direction of the comparison visible."""

    fast = [1.0, 1.01, 0.99, 1.02, 1.0]
    slow = [2.0, 2.02, 1.98, 2.04, 2.0]

    lower, upper = _bootstrap_ratio_interval(fast, slow)

    assert 0.4 < lower < 0.6
    assert 0.4 < upper < 0.6


def test_a_non_zero_rank_gathers_nothing_and_contributes_its_record(monkeypatch):
    """Only rank 0 writes a document; the others must return an empty result."""

    contributed: list[object] = []
    destroyed: list[object] = []
    control = object()
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setattr(
        "benchmarks.statevector_release_evidence._control_group", lambda world: control
    )
    monkeypatch.setattr(
        "benchmarks.statevector_release_evidence.dist.gather_object",
        lambda record, buffer, dst, group: contributed.append(
            (record, buffer, dst, group)
        ),
    )
    monkeypatch.setattr(
        "benchmarks.statevector_release_evidence.dist.destroy_process_group",
        destroyed.append,
    )

    assert _gather({"rank": 3}, 8) == []
    assert contributed == [({"rank": 3}, None, 0, control)]
    assert destroyed == [control]


def test_rank_zero_returns_every_rank_it_gathered(monkeypatch):
    records = [{"rank": index} for index in range(4)]
    destroyed: list[object] = []
    control = object()

    def gather(record, buffer, dst, group):
        assert group is control
        buffer[:] = records

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(
        "benchmarks.statevector_release_evidence._control_group", lambda world: control
    )
    monkeypatch.setattr(
        "benchmarks.statevector_release_evidence.dist.gather_object", gather
    )
    monkeypatch.setattr(
        "benchmarks.statevector_release_evidence.dist.destroy_process_group",
        destroyed.append,
    )

    assert _gather({"rank": 0}, 4) == records
    assert destroyed == [control]


def test_a_short_gather_raises_instead_of_sealing_a_partial_document(monkeypatch):
    """A record that never arrived must fail the leg, not shrink the payload."""

    def gather(record, buffer, dst, group):
        buffer[0] = {"rank": 0}
        buffer[1] = {"rank": 1}

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(
        "benchmarks.statevector_release_evidence._control_group", lambda world: object()
    )
    monkeypatch.setattr(
        "benchmarks.statevector_release_evidence.dist.gather_object", gather
    )
    monkeypatch.setattr(
        "benchmarks.statevector_release_evidence.dist.destroy_process_group",
        lambda group: None,
    )

    with pytest.raises(RuntimeError, match="for rank 2 of 4"):
        _gather({"rank": 0}, 4)


def test_the_record_gather_runs_on_a_cpu_group_not_the_device_group(monkeypatch):
    """NCCL cannot carry the records after a long training leg; gloo can."""

    opened: list[dict[str, object]] = []
    monkeypatch.delenv("GLOO_SOCKET_IFNAME", raising=False)
    monkeypatch.setenv("NCCL_SOCKET_IFNAME", "ens22f0")
    monkeypatch.setattr(
        "benchmarks.statevector_release_evidence.dist.new_group",
        lambda **kwargs: opened.append(kwargs) or object(),
    )

    assert _control_group(8) is not None
    assert opened[0]["backend"] == "gloo"
    assert opened[0]["timeout"].total_seconds() == 300
    assert os.environ["GLOO_SOCKET_IFNAME"] == "ens22f0"


def test_the_record_gather_single_device_run_needs_no_group(monkeypatch):
    """A one-device leg writes its own record and opens nothing."""

    def refuse(**kwargs):
        raise AssertionError("a single-device leg must not open a group")

    monkeypatch.setattr(
        "benchmarks.statevector_release_evidence.dist.new_group", refuse
    )

    assert _gather({"rank": 0}, 1) == [{"rank": 0}]
    assert _control_group(1) is None


def test_the_frozen_capacity_workload_is_the_one_the_runner_builds():
    """The circuit the runner builds is the circuit the manifest froze."""

    manifest = json.loads(
        Path("benchmarks/manifests/statevector_capacity_workload_v2.json").read_text()
    )

    circuit, parameters = _capacity_circuit(manifest, torch.device("cpu"))

    assert circuit.n_wires == manifest["n_wires"]
    assert [item.numel() for item in parameters] == [1, 1]


def test_the_frozen_capacity_workload_rejects_an_edited_gate_list():
    manifest = json.loads(
        Path("benchmarks/manifests/statevector_capacity_workload_v2.json").read_text()
    )
    manifest["gates"] = ["RY(n-1,theta)"]

    with pytest.raises(SystemExit, match="unsupported frozen gate list"):
        _capacity_circuit(manifest, torch.device("cpu"))
