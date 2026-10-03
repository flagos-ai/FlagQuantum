"""Fail-closed readiness gate for the distributed matrix-product-state capability.

The gate separates two evidence roles, for the same reason the statevector and
tensor-network gates do. A *release payload* is a signed
``measured_production_run`` artifact that shards one logical MPS across ranks and
sets ``release_gate_allowed``; only those contribute release worlds and must carry
the per-rank field contract. A *single-device baseline* is the signed capacity
failure that proves the frozen workload exceeds one device, and it is read as
provenance rather than as a release payload.

The MPS lane has two boundaries the tensor-network lane does not. First, MPS
training is approximate: a sharded MPS claim is only a claim about the
matrix-product-state path, so the payload has to identify that path by state mode
and declare that it fell back to nothing -- a statevector fallback, a replicated
per-rank autograd replay, or a full local state view all compute the same numbers
by a different route and none of them is sharded MPS evidence. Second, MPS
training is sharded along two axes: sites and bonds. A payload that reports how
work was spread over sites but not over bonds has described half the execution,
so site, bond, boundary-gradient and parameter-gradient ownership are each
required, together with the backward memory and per-boundary communication
measurements taken while the backward pass actually ran.

Every check below is stated against the frozen manifest rather than against
numbers written into this file, so the contract moves only by re-freezing the
manifest. ``benchmarks/results/scalability/README.md`` is the published statement
of the same requirements, and the contract tests map each of its bullets to the
blockers that enforce it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from flagquantum.runtime.observability.evidence import (
    ArtifactClass,
    EvidenceScope,
    RuntimeProvenance,
    create_evidence_artifact,
    verify_evidence_artifact,
)

MANIFEST = Path("benchmarks/manifests/mps_release_v1.json")
RESULTS = Path("benchmarks/results/scalability")

# The status vocabularies are shared with the MPS readiness audit in
# ``flagquantum.runtime.audit.mps_readiness``: a plan that has not reached one of
# these states describes an intention, not a measurement.
_EXECUTED_BACKWARD_MEMORY_STATUS = frozenset(
    {"measured", "production_measured", "executed"}
)
_EXECUTED_BACKWARD_COMMUNICATION_STATUS = frozenset(
    {"executed", "production_executed", "multi_node_production_transport"}
)
_NON_EXECUTED_BACKWARD_LABELS = (
    "unknown",
    "pending",
    "planned",
    "not_executed",
    "local_simulated",
    "estimated",
)
# Most required fields are absent when they are empty: an empty ownership map
# describes nothing. A declared blocker list is the exception, because an empty
# one is exactly the result a clean payload reports, so it counts as present when
# the key is there.
_EMPTY_IS_A_DECLARATION = frozenset({"blockers"})
_FORBIDDEN_FALLBACKS = (
    "replicated_mps_autograd",
    "full_local_mps_state_view",
    "full_local_mps_replay",
    "statevector_fallback",
)


def load_manifest(path: Path = MANIFEST) -> dict[str, Any]:
    """Read and check the frozen MPS release manifest.

    Every file that defines the capacity workload is digest-bound rather than
    merely named, because the definition decides the site count, the bond
    schedule, the rank boundaries and the circuit the completion artifact has to
    match. The workload is defined by a launcher that pins the site count and a
    body that builds the circuit, so both are checked: digesting only the
    launcher would let the work the contract claims to have measured change under
    a stable filename.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") not in {
        "flagquantum.mps.release_manifest.v1",
    }:
        raise ValueError("invalid MPS release manifest schema")
    if not payload.get("frozen_before_release_run"):
        raise ValueError("MPS thresholds must be frozen before measurement")
    capacity = payload["capacity_workload"]
    declared_sources = {
        str(capacity["workload_definition_path"]): str(
            capacity["workload_definition_sha256"]
        )
    }
    for source in capacity.get("workload_definition_sources", ()):
        declared_sources[str(source["path"])] = str(source["sha256"])
    for source_path, digest in declared_sources.items():
        frozen = Path(source_path).read_bytes()
        if hashlib.sha256(frozen).hexdigest() != digest:
            raise ValueError(
                "the frozen MPS capacity workload does not match the digest the "
                f"manifest declares for {source_path}"
            )
    return payload


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


def envelope_carries_world(world_size: int) -> bool:
    """Return whether the evidence envelope can describe a run over N ranks.

    A release payload is a sealed artifact, so a frozen release world that no
    evidence scope can carry is a world no run can ever certify. The answer is
    obtained by asking the envelope to build a provenance record for that many
    devices rather than by repeating its scope table here, so a scope added later
    widens the contract by itself instead of leaving a stale copy of the rule in
    this file.
    """

    for scope in EvidenceScope:
        try:
            create_evidence_artifact(
                artifact_class=ArtifactClass.MEASURED_PRODUCTION_RUN,
                evidence_scope=scope,
                provenance=RuntimeProvenance(
                    commit="0" * 40,
                    workload_sha256="0" * 64,
                    command=("scope carriage probe",),
                    devices=tuple(f"probe-device-{rank}" for rank in range(world_size)),
                    topology="scope carriage probe",
                    rank_mapping=tuple(f"rank={rank}" for rank in range(world_size)),
                    collective_backend="nccl",
                    warmup=1,
                    iterations=1,
                    seeds=(0,),
                    raw_log_sha256="0" * 64,
                    fallback_events=(),
                ),
                evidence={"measured_peak_memory_bytes": 0},
                signing_key=b"scope carriage probe",
            )
        except ValueError:
            continue
        return True
    return False


def _is_absent(value: Any) -> bool:
    return value is None or value == "" or value == () or value == []


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
        and _as_int(evidence.get("world_size")) == 1
        and str(evidence.get("workload_sha256")) == str(contract["workload_sha256"])
    )


def _has_required_fields(
    evidence: Mapping[str, Any],
    provenance: Mapping[str, Any],
    required: Sequence[str],
) -> tuple[str, ...]:
    """Return the required fields that are absent or empty.

    ``hardware_inventory``, ``raw_log_sha256``, ``commit`` and ``rank_mapping``
    are the fields the envelope may carry instead of the payload: the sealer reads
    them from the machine that ran the rank, so a payload that omits one is not
    missing the evidence. A field whose empty value is itself a statement, such as
    a blocker list declaring that there are none, counts as present when the key
    is there.
    """

    provenance_fields = {
        "hardware_inventory": "devices",
        "raw_log_sha256": "raw_log_sha256",
        "commit": "commit",
        "rank_mapping": "rank_mapping",
    }
    missing: list[str] = []
    for name in required:
        if name in _EMPTY_IS_A_DECLARATION:
            if name not in evidence:
                missing.append(name)
            continue
        value = evidence.get(name)
        if _is_absent(value) and name in provenance_fields:
            value = provenance.get(provenance_fields[name])
        if _is_absent(value):
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


def _fallback_text(evidence: Mapping[str, Any]) -> str:
    """Return the declared fallback semantics as one lower-case string.

    A mapping is read by key because the forbidden routes are named by the route
    that was taken, and a sequence is read element by element because the routes
    that were *not* taken are sometimes named that way.
    """

    declared = evidence.get("fallback_semantics")
    if isinstance(declared, Mapping):
        values = [str(key) for key in declared]
    elif isinstance(declared, (list, tuple)):
        values = [str(item) for item in declared]
    elif _is_absent(declared):
        values = []
    else:
        values = [str(declared)]
    return " ".join(values).lower()


def _is_sharded_training(evidence: Mapping[str, Any]) -> bool:
    """Return whether the payload measured a sharded MPS training update.

    A forward-only sharded run is not a training claim, and a payload that reports
    ownership semantics without the maps behind them has asserted the structure
    rather than measured it, so both are required.
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


def _mps_contract_blockers(
    evidence: Mapping[str, Any], manifest: Mapping[str, Any]
) -> tuple[str, ...]:
    """Return the requirement items one release payload fails to meet.

    Every blocker names the requirement it fails rather than the field that is
    missing, because the requirement is the claim under review and the field is
    only how the claim is carried.
    """

    blockers: list[str] = []
    capacity = manifest["capacity_workload"]
    expected_state_mode = str(manifest["runtime"]["state_mode"]).lower()

    if str(evidence.get("state_mode", "")).lower() != expected_state_mode:
        blockers.append("missing_mps_state_mode_evidence")

    forward_semantics = evidence.get(
        "mps_forward_distribution_semantics",
        evidence.get("forward_distribution_semantics"),
    )
    if (
        evidence.get("distribution_semantics") != "sharded_across_ranks"
        or forward_semantics != "sharded_across_ranks"
    ):
        blockers.append("missing_sharded_mps_forward_evidence")

    backward_semantics = evidence.get(
        "mps_backward_distribution_semantics",
        evidence.get("backward_distribution_semantics"),
    )
    if backward_semantics != "sharded_across_ranks":
        blockers.append("missing_sharded_mps_backward_evidence")

    backward_execution = str(evidence.get("backward_execution", "unknown")).lower()
    if (
        not backward_execution
        or any(token in backward_execution for token in _NON_EXECUTED_BACKWARD_LABELS)
        or not (
            evidence.get("backward_execution_measured") is True
            or "executed" in backward_execution
            or "production" in backward_execution
        )
    ):
        blockers.append("missing_executed_production_backward_evidence")

    for field, blocker in (
        ("site_shard_ownership", "missing_site_shard_ownership"),
        ("bond_shard_ownership", "missing_bond_shard_ownership"),
        ("boundary_gradient_ownership", "missing_boundary_gradient_ownership"),
        ("parameter_gradient_ownership", "missing_parameter_gradient_ownership"),
    ):
        value = evidence.get(field)
        if not isinstance(value, Mapping) or not value:
            blockers.append(blocker)

    exchange = evidence.get("boundary_adjoint_exchange")
    routes = evidence.get("boundary_gradient_routes")
    if (
        not isinstance(exchange, Mapping)
        or not exchange
        or not isinstance(routes, Mapping)
        or not routes
        or str(exchange.get("execution_status", "")).lower() != "executed"
    ):
        blockers.append("missing_executed_boundary_adjoint_exchange")

    memory_plan = evidence.get("mps_backward_memory_plan")
    backward_peak = (
        memory_plan.get("measured_backward_peak_memory_bytes")
        if isinstance(memory_plan, Mapping)
        else None
    ) or evidence.get("rank_backward_peak_memory_bytes")
    if (
        not isinstance(memory_plan, Mapping)
        or str(memory_plan.get("status", "")).lower()
        not in _EXECUTED_BACKWARD_MEMORY_STATUS
        or _is_absent(backward_peak)
    ):
        blockers.append("missing_production_backward_memory_evidence")

    communication_plan = evidence.get("mps_backward_communication_plan")
    if not isinstance(communication_plan, Mapping):
        blockers.append("missing_production_boundary_communication_evidence")
    else:
        status = str(communication_plan.get("status", "")).lower()
        edges = tuple(communication_plan.get("boundary_edges") or ())
        per_boundary = evidence.get("boundary_communication_bytes")
        if (
            status not in _EXECUTED_BACKWARD_COMMUNICATION_STATUS
            or not edges
            or any(
                str(edge.get("execution_status", "")).lower() != "executed"
                for edge in edges
                if isinstance(edge, Mapping)
            )
            or any(not isinstance(edge, Mapping) for edge in edges)
            or not isinstance(per_boundary, Mapping)
            or not per_boundary
        ):
            blockers.append("missing_production_boundary_communication_evidence")

    if not _is_sharded_training(evidence):
        blockers.append("missing_sharded_training_ownership")

    if _as_int(evidence.get("training_step_count")) <= 0:
        blockers.append("missing_positive_training_step_count")

    if (
        evidence.get("single_gpu_expected_oom") is not True
        or _is_absent(evidence.get("capacity_baseline_device"))
        or _is_absent(evidence.get("capacity_failure_reason"))
    ):
        blockers.append("missing_single_gpu_expected_oom_declaration")

    fallback_text = _fallback_text(evidence)
    if (
        not fallback_text
        or any(token in fallback_text for token in _FORBIDDEN_FALLBACKS)
        or evidence.get("fallback_semantics")
        != manifest["runtime"]["fallback_semantics"]
    ):
        blockers.append("missing_no_fallback_semantics")
    if evidence.get("fallback_events") not in ([], ()):
        blockers.append("production_payload_declares_fallback_events")

    declared_blockers = evidence.get("blockers")
    if not isinstance(declared_blockers, (list, tuple)) or declared_blockers:
        blockers.append("production_payload_declares_blockers")

    if _is_absent(evidence.get("capacity_baseline_device")):
        blockers.append("missing_capacity_baseline_device")
    if _is_absent(evidence.get("capacity_failure_reason")):
        blockers.append("missing_capacity_failure_reason")
    if evidence.get("topology_scope") != "multi_node_production_transport" or _as_int(
        evidence.get("node_count")
    ) < _as_int(manifest["topologies"]["minimum_multi_node_count"]):
        blockers.append("missing_multinode_transport_scope")

    # The shape is what makes a completion artifact evidence for *this* frozen
    # workload, so a missing shape field and a wrong one are reported apart: the
    # first is an incomplete record and the second is a different workload.
    for field, expected in (
        ("n_sites", _as_int(capacity["n_sites"])),
        ("max_bond_dimension", _as_int(capacity["trained_max_bond"])),
        ("batch_size", _as_int(capacity["batch_size"])),
    ):
        value = evidence.get(field)
        if value is None:
            blockers.append("missing_capacity_workload_shape_evidence")
        elif _as_int(value) != expected:
            blockers.append("capacity_workload_shape_mismatch")
    if evidence.get("dtype") is None:
        blockers.append("missing_capacity_workload_shape_evidence")
    elif str(evidence["dtype"]) != str(capacity["dtype"]):
        blockers.append("capacity_workload_shape_mismatch")

    return tuple(blockers)


def evaluate_mps_release(
    artifacts: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    signing_key: bytes | None = None,
) -> tuple[bool, tuple[str, ...]]:
    """Return whether the artifacts certify the frozen MPS release.

    ``signing_key=None`` skips envelope verification, which is what a caller that
    only wants the contract read does. The command-line entry point always passes
    a key -- the environment's, or the empty bytes the sealer is configured with
    when signing is disabled -- so a promoted artifact is never accepted on the
    strength of a claim its envelope does not carry.
    """

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

    worlds = {_as_int(evidence.get("world_size")) for evidence, _ in production}
    required_worlds = set(manifest["topologies"]["release_world_sizes"])
    if not required_worlds <= worlds:
        blockers.append("missing_release_world_sizes")
    # Sealing is part of the protocol, so a frozen world the evidence envelope
    # cannot describe is reported before a campaign spends a cluster run on it.
    if any(
        not envelope_carries_world(int(world_size)) for world_size in required_worlds
    ):
        blockers.append("release_world_size_not_carriable_by_evidence_envelope")

    if not any(
        _as_int(evidence.get("node_count"))
        >= _as_int(manifest["topologies"]["minimum_multi_node_count"])
        and evidence.get("topology_scope") == "multi_node_production_transport"
        for evidence, _ in production
    ):
        blockers.append("missing_multinode_correctness_artifact")

    if not any(_is_sharded_training(evidence) for evidence, _ in production):
        blockers.append("missing_sharded_training_ownership")

    # The ladder is part of the protocol: a timed comparison against a ladder that
    # was chosen after the numbers were known is not a frozen acceptance test. The
    # manifest freezes it as null while no MPS timing exists, and the gate reports
    # that state rather than reading the null as an empty ladder.
    speed = manifest["speed_workload"]
    if not speed.get("configuration_ladder"):
        blockers.append("speed_configuration_ladder_not_frozen")
    qualified = [
        evidence
        for evidence, _ in production
        if evidence.get("acceptance_case") == "matched_speed"
        and _as_float(evidence.get("speedup")) >= float(speed["minimum_speedup"])
        and _confidence_interval_lower(evidence)
        > float(speed["confidence_interval_must_exclude_speedup"])
        and _as_float(evidence.get("scaling_efficiency"))
        >= float(speed["minimum_scaling_efficiency"])
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

    completion = any(
        evidence.get("acceptance_case") == "multi_gpu_capacity_completion"
        and str(evidence.get("workload_sha256")) == str(capacity["workload_sha256"])
        for evidence, _ in production
    )
    if capacity["premise_established"] and not completion:
        blockers.append("missing_multi_gpu_capacity_completion_artifact")

    for evidence, provenance in production:
        blockers.extend(_mps_contract_blockers(evidence, manifest))
        if _has_required_fields(evidence, provenance, required):
            blockers.append("production_artifact_missing_required_fields")

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
    passed, blockers = evaluate_mps_release(
        artifacts, manifest, signing_key=signing_key
    )
    print(
        json.dumps(
            {
                "capability": "distributed_matrix_product_state",
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
