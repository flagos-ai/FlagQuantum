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
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from benchmarks.internal.evidence.statevector_release_gate import (
    evaluate_issue044_release,
    load_manifest,
)
from benchmarks.statevector_release_evidence import (
    _bootstrap_ratio_interval,
    _capacity_circuit,
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


def _speed_summary(tmp_path: Path) -> dict:
    """Drive the frozen speed summary over synthetic legs and seal its output.

    The two legs carry deliberately different timings so the ratio is not a
    comparison of a run with itself, and both carry the same configuration list
    because the manifest's three shapes are what the ratio is allowed to cover.
    """

    manifest = load_manifest()
    acceptance = manifest["speed_workload"]["acceptance_configuration"]
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
