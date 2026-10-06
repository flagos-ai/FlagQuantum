"""Fail-closed readiness gate for the distributed tensor-network capability.

The gate separates two evidence roles, for the same reason the statevector gate
does. A *release payload* is a signed ``measured_production_run`` artifact that
shards one logical tensor-network workload across ranks and sets
``release_gate_allowed``; only those contribute release worlds and must carry the
per-rank field contract. A *single-device baseline* is the signed capacity
failure that proves the frozen workload exceeds one device, and it is read as
provenance rather than as a release payload.

The tensor-network lane has one boundary the statevector lane does not: the
measured pair is two hosts, and no third host exists on this cluster. The
manifest therefore freezes a *pair* topology instead of a single-node ladder, and
the gate requires the multi-node route to be observed in the payload rather than
assumed from the rank count.

The signature is not a capability. The runtime-evidence envelope carries no
capability key, so the gate admits a payload only when the payload states a state
mode the audit vocabulary recognizes as tensor-network, and a payload of another
capability is counted and reported as foreign evidence instead of being read as a
set that is one shape-specific blocker away from passing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from flagquantum.runtime.audit.vocabulary import TENSOR_NETWORK_STATE_MODES
from flagquantum.runtime.observability.evidence import verify_evidence_artifact

MANIFEST = Path("benchmarks/manifests/tensor_network_release_v1.json")
RESULTS = Path("benchmarks/results/scalability")


def load_manifest(path: Path = MANIFEST) -> dict[str, Any]:
    """Read and check the frozen tensor-network release manifest."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") not in {
        "flagquantum.tensor_network.release_manifest.v1",
    }:
        raise ValueError("invalid tensor-network release manifest schema")
    if not payload.get("frozen_before_release_run"):
        raise ValueError("tensor-network thresholds must be frozen before measurement")
    capacity = payload["capacity_workload"]
    declared = str(capacity["workload_sha256"])
    frozen = Path(str(capacity["workload_manifest_path"])).read_bytes()
    if hashlib.sha256(frozen).hexdigest() != declared:
        raise ValueError(
            "the frozen tensor-network capacity workload does not match the digest "
            "the manifest declares"
        )
    return payload


def _is_single_device_baseline(
    evidence: Mapping[str, Any], contract: Mapping[str, Any]
) -> bool:
    """Return whether a signed artifact is the capacity-failure baseline.

    The baseline is identified by the acceptance case it declares rather than by
    its world size: a world-size-1 artifact that did not fail would be a
    different, and much weaker, statement.
    """

    return (
        evidence.get("acceptance_case") == "single_gpu_capacity_failure"
        and evidence.get("single_device_oom_observed") is True
        and evidence.get("world_size") == 1
        and str(evidence.get("workload_sha256")) == str(contract["workload_sha256"])
    )


def _is_tensor_network_evidence(evidence: Mapping[str, Any]) -> bool:
    """Return whether a payload is evidence for this capability at all.

    The runtime-evidence envelope carries no capability key, and the statevector
    release payloads are ``measured_production_run`` artifacts with
    ``release_gate_allowed`` set, so without a capability test another
    capability's payload would satisfy this contract's release world, multi-node
    and speed requirements while only the tensor-network shape checks remained. A
    statevector payload either declares no state mode at all or declares one the
    vocabulary assigns to another family, so the test is whether the payload
    declares one of the modes the audit vocabulary recognizes as tensor-network.
    Which of those the frozen contract accepts is a separate question, asked by
    the contract check and answered from the manifest.
    """

    return str(evidence.get("state_mode", "")).lower() in TENSOR_NETWORK_STATE_MODES


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _is_sharded_training(evidence: Mapping[str, Any]) -> bool:
    """Return whether the payload measured a sharded training update.

    A forward-only sharded run is not a training claim, and a payload that
    reports ownership semantics without the maps behind them has asserted the
    structure rather than measured it, so both are required.
    """

    if evidence.get("distribution_semantics") != "sharded_across_ranks":
        return False
    if _as_int(evidence.get("training_step_count")) <= 0:
        return False
    for name in (
        "parameter_ownership_semantics",
        "gradient_ownership_semantics",
        "optimizer_update_ownership_semantics",
    ):
        if evidence.get(name) != "sharded_across_ranks":
            return False
    for name in (
        "parameter_ownership",
        "gradient_ownership",
        "optimizer_update_ownership",
    ):
        value = evidence.get(name)
        if not isinstance(value, Mapping) or not value:
            return False
    return True


def _has_required_fields(
    evidence: Mapping[str, Any],
    provenance: Mapping[str, Any],
    required: Sequence[str],
) -> tuple[str, ...]:
    """Return the required fields that are absent or empty.

    ``hardware_inventory`` is the one field the envelope may carry instead of the
    payload: the sealer reads the device inventory from the machine that ran the
    rank, so a payload that omits it is not missing the evidence.
    """

    missing: list[str] = []
    for name in required:
        value = evidence.get(name)
        if name == "hardware_inventory" and value in (None, "", (), []):
            value = provenance.get("devices")
        if name == "raw_log_sha256" and value in (None, "", (), []):
            value = provenance.get("raw_log_sha256")
        if name == "commit" and value in (None, "", (), []):
            value = provenance.get("commit")
        if name == "rank_mapping" and value in (None, "", (), []):
            value = provenance.get("rank_mapping")
        if value in (None, "", (), []):
            missing.append(name)
    return tuple(missing)


def _confidence_interval_lower(evidence: Mapping[str, Any]) -> float:
    """Return the lower bound of the reported speedup interval."""

    for container in (evidence, evidence.get("measurements")):
        if not isinstance(container, Mapping):
            continue
        for key in ("speedup_confidence_interval", "confidence_interval"):
            interval = container.get(key)
            if isinstance(interval, (list, tuple)) and interval:
                return _as_float(interval[0])
    return float("nan")


def evaluate_tensor_network_release(
    artifacts: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    signing_key: bytes | None = None,
) -> tuple[bool, tuple[str, ...]]:
    """Return whether the artifacts certify the frozen tensor-network release."""

    blockers: list[str] = []
    capacity = manifest["capacity_workload"]
    baseline_contract = dict(capacity["single_gpu_baseline"])
    baseline_contract["workload_sha256"] = capacity["workload_sha256"]
    required = tuple(manifest["required_artifact_fields"])
    baseline_required = tuple(baseline_contract["required_artifact_fields"])

    production: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    baselines: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    rejected = False
    single_device_claimants = False
    foreign_evidence = 0
    for artifact in artifacts:
        if artifact.get("artifact_class") != "measured_production_run":
            continue
        evidence = artifact.get("evidence")
        provenance = artifact.get("provenance")
        if not isinstance(evidence, Mapping) or not isinstance(provenance, Mapping):
            rejected = True
            continue
        if signing_key is not None:
            valid, _ = verify_evidence_artifact(artifact, signing_key=signing_key)
            if not valid:
                rejected = True
                continue
        # The envelope carries no capability key, so a payload of another
        # capability can otherwise satisfy this contract's requirements. It is
        # counted and reported rather than dropped: a directory holding five
        # files of which none is admissible must not read as an empty set that
        # only the shape-specific blockers are missing from.
        if not _is_tensor_network_evidence(evidence):
            foreign_evidence += 1
            continue
        if _is_single_device_baseline(evidence, baseline_contract):
            baselines.append((evidence, provenance))
        if evidence.get("release_gate_allowed") is not True:
            continue
        if _as_int(evidence.get("world_size")) > 1:
            production.append((evidence, provenance))
        else:
            single_device_claimants = True

    if rejected:
        blockers.append("invalid_or_unsigned_production_artifact")
    if single_device_claimants:
        blockers.append("single_device_world_cannot_be_a_release_payload")
    if foreign_evidence:
        blockers.append("production_artifact_is_not_tensor_network_evidence")

    worlds = {_as_int(evidence.get("world_size")) for evidence, _ in production}
    required_worlds = set(manifest["topologies"]["release_world_sizes"])
    if not required_worlds <= worlds:
        blockers.append("missing_release_world_sizes")

    # A pair-only cluster can still show that the ranks are on two hosts. The
    # gate requires the observation, not the rank count: a payload that reports
    # node_count 2 without an observed multi-node route has not measured one.
    multi_node = [
        evidence
        for evidence, _ in production
        if _as_int(evidence.get("node_count")) > 1
        and evidence.get("topology_scope") == "multi_node_production_transport"
    ]
    if not multi_node:
        blockers.append("missing_multinode_correctness_artifact")

    if not any(_is_sharded_training(evidence) for evidence, _ in production):
        blockers.append("missing_sharded_training_ownership")

    speed = manifest["speed_workload"]
    qualified = [
        evidence
        for evidence, _ in production
        if evidence.get("acceptance_case") == "matched_speed"
        and _as_float(evidence.get("speedup")) >= float(speed["minimum_speedup"])
        and _confidence_interval_lower(evidence)
        > float(speed["confidence_interval_must_exclude_speedup"])
    ]
    if not qualified:
        blockers.append("missing_statistically_significant_speedup_artifact")

    if not capacity["premise_established"]:
        blockers.append("capacity_premise_not_established")
    elif not baselines:
        blockers.append("missing_single_gpu_measured_oom_artifact")
    elif any(
        _has_required_fields(evidence, provenance, baseline_required)
        for evidence, provenance in baselines
    ):
        blockers.append("single_gpu_baseline_missing_required_fields")

    completions = [
        evidence
        for evidence, _ in production
        if evidence.get("acceptance_case") == "multi_gpu_capacity_completion"
        and str(evidence.get("workload_sha256")) == str(capacity["workload_sha256"])
    ]
    if capacity["premise_established"] and not completions:
        blockers.append("missing_multi_gpu_capacity_completion_artifact")

    # Being tensor-network evidence is not the same as being evidence for the
    # frozen contract: a payload may name any of the modes the vocabulary
    # recognizes while the manifest froze the distributed one, and a payload that
    # names another mode was measured by another execution path.
    expected_state_mode = str(manifest["runtime"]["state_mode"]).lower()
    for evidence, _ in production:
        if str(evidence.get("state_mode", "")).lower() != expected_state_mode:
            blockers.append("missing_tensor_network_state_mode_evidence")

    for evidence, provenance in production:
        if _has_required_fields(evidence, provenance, required):
            blockers.append("production_artifact_missing_required_fields")
            break

    ordered = tuple(dict.fromkeys(blockers))
    return (not ordered, ordered)


def baseline_results(manifest: Mapping[str, Any]) -> Path:
    """Return the declared directory the single-device baseline is read from.

    The strict promotion audit requires every file under ``RESULTS`` to be
    release-grade sharded scalability evidence, which a single-device run can
    never be, so the baseline declares its own directory.
    """

    declared = Path(
        str(manifest["capacity_workload"]["single_gpu_baseline"]["results_path"])
    )
    if declared == RESULTS or RESULTS in declared.parents:
        raise ValueError(
            f"the single-device baseline must be read outside {RESULTS}; a "
            "single-device run cannot be release-grade sharded evidence"
        )
    return declared


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument(
        "--candidate",
        type=Path,
        action="append",
        default=None,
        help="directory holding candidate artifacts; repeatable",
    )
    parser.add_argument(
        "--baseline-directory",
        type=Path,
        default=None,
        help=(
            "directory holding the single-device capacity baseline, overriding "
            "the manifest's declared path. A candidate set that carries its own "
            "baseline names it here so the gate reads the baseline the set was "
            "measured against rather than whichever one is checked in."
        ),
    )
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    baseline = (
        args.baseline_directory
        if args.baseline_directory is not None
        else baseline_results(manifest)
    )
    directories = (baseline, *args.candidate) if args.candidate else (RESULTS, baseline)
    artifacts = [
        json.loads(path.read_text(encoding="utf-8"))
        for directory in directories
        for path in sorted(directory.glob("*.json"))
    ]
    signing_key = os.environ.get("FQ_EVIDENCE_SIGNING_KEY", "").encode()
    passed, blockers = evaluate_tensor_network_release(
        artifacts, manifest, signing_key=signing_key
    )
    print(
        json.dumps(
            {
                "capability": "distributed_tensor_network",
                "artifact_count": len(artifacts),
                "evaluated": [str(item) for item in directories],
                "passed": passed,
                "blockers": list(blockers),
            }
        )
    )
    if not passed:
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    main()
