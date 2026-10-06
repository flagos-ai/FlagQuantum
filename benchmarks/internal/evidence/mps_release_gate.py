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
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.internal.evidence.mps_shardable_ceiling import (
    SERIAL_FRACTION_KEY,
    calibration_errors,
    frozen_serial_fraction,
    shardable_ceiling_speedup,
)
from flagquantum.runtime.audit.mps_readiness import (
    _FORBIDDEN_FALLBACKS as _AUDITED_FORBIDDEN_FALLBACKS,
)
from flagquantum.runtime.audit.vocabulary import MPS_STATE_MODES
from flagquantum.runtime.observability.evidence import (
    ArtifactClass,
    EvidenceScope,
    RuntimeProvenance,
    create_evidence_artifact,
    verify_evidence_artifact,
)

MANIFEST = Path("benchmarks/manifests/mps_release_v1.json")
RESULTS = Path("benchmarks/results/scalability")
# The frozen contract names its calibration artifact relative to the repository
# rather than to the invocation, because the threshold it declares is a property
# of the source tree a reader can re-read from any directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

# The status vocabularies are shared with the MPS readiness audit in
# ``flagquantum.runtime.audit.mps_readiness``: a plan that has not reached one of
# these states describes an intention, not a measurement.
_EXECUTED_BACKWARD_MEMORY_STATUS = frozenset(
    {"measured", "production_measured", "executed"}
)
_EXECUTED_BACKWARD_COMMUNICATION_STATUS = frozenset(
    {"executed", "production_executed", "multi_node_production_transport"}
)
# Most required fields are absent when they are empty: an empty ownership map
# describes nothing. A declared blocker list is the exception, because an empty
# one is exactly the result a clean payload reports, so it counts as present when
# the key is there.
_EMPTY_IS_A_DECLARATION = frozenset({"blockers"})
# The forbidden-route rule is owned by the MPS readiness audit, which already
# maps each route to the code that refuses it. Reading its keys here rather than
# repeating the four names keeps one source of truth for which fallbacks a claim
# may not rest on.
_FORBIDDEN_FALLBACKS = tuple(_AUDITED_FORBIDDEN_FALLBACKS)


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
    # The statuses the gate matches against are part of the frozen contract, and a
    # manifest that named none of them would refuse every payload for a reason no
    # reader could find in the document, so the declaration is checked here.
    accepted_backward = payload["runtime"].get("accepted_backward_execution_statuses")
    if not isinstance(accepted_backward, list) or not accepted_backward:
        raise ValueError(
            "the manifest must freeze the backward-execution statuses it accepts"
        )
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


def premise_evidence_errors(
    capacity: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return the frozen premise files that are missing or have drifted.

    ``capacity_workload`` names every file the premise rests on and digests it, so
    a premise can be re-read rather than believed. An entry this manifest already
    knows to be unresolvable is returned as its own name rather than raised,
    because a manifest that disclosed it must not be able to hide behind a passing
    gate. This contract currently discloses none: the raw log and the telemetry it
    once recorded as absent are on disk at the digests it recorded for them. The
    caller turns a non-empty result into a blocker, so any future drift is reported
    and the premise stays unestablished until the files agree.
    """

    errors: list[str] = []
    for entry in capacity.get("frozen_evidence", ()):
        path = Path(str(entry["path"]))
        if not path.is_file():
            errors.append(f"capacity_workload.frozen_evidence[{path}]: absent")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != str(entry["sha256"]):
            errors.append(f"capacity_workload.frozen_evidence[{path}]: drifted")
    provenance = capacity.get("premise_provenance", {})
    for path_text in provenance.get("absent_sources", ()):
        path = Path(str(path_text))
        if not path.is_file():
            errors.append(f"capacity_workload.premise_provenance[{path}]: absent")
    artifact_text = provenance.get("artifact")
    if artifact_text is not None:
        path = Path(str(artifact_text))
        if not path.is_file():
            errors.append(f"capacity_workload.premise_provenance[{path}]: absent")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != str(
            provenance["artifact_sha256"]
        ):
            errors.append(f"capacity_workload.premise_provenance[{path}]: drifted")
    return tuple(errors)


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


def _efficiency_matches_frozen_ceiling(
    evidence: Mapping[str, Any],
    speed: Mapping[str, Any],
    blockers: list[str],
) -> bool:
    """Return whether a payload's efficiency is the ceiling-relative one.

    The scaling threshold is compared against the speedup the timed workload's own
    arithmetic permits, and the payload states both the fraction it was computed
    from and the ceiling it divides by. Trusting the payload's own efficiency would
    let any denominator pass the gate, so the ceiling is recomputed from the frozen
    fraction and the payload's world size, and the reported efficiency has to be
    the ratio it implies. A payload that names the definition this contract
    replaced is refused by name, so the refusal says which denominator was wrong
    rather than reporting a number below a threshold.
    """

    serial_fraction = frozen_serial_fraction(speed)
    # A payload reaches here only after the production-world binding admitted it,
    # so its world is at least two and the ceiling is always computable; a world
    # the ceiling cannot divide by is refused by that binding rather than here.
    world = _as_int(evidence.get("world_size"))
    ceiling = shardable_ceiling_speedup(serial_fraction, world)
    expected_definition = str(speed["scaling_efficiency_definition"])
    if str(evidence.get("scaling_efficiency_definition", "")) != expected_definition:
        blockers.append("matched_speed_scaling_efficiency_uses_a_replaced_definition")
        return False
    if _as_float(evidence.get(SERIAL_FRACTION_KEY)) != serial_fraction:
        blockers.append("matched_speed_payload_serial_fraction_disagrees_with_contract")
        return False
    if not _close(_as_float(evidence.get("shardable_ceiling_speedup")), ceiling):
        blockers.append("matched_speed_ceiling_disagrees_with_frozen_fraction")
        return False
    speedup = _as_float(evidence.get("speedup"))
    reported = _as_float(evidence.get("scaling_efficiency"))
    if not _close(reported, speedup / ceiling):
        blockers.append("matched_speed_scaling_efficiency_disagrees_with_its_ceiling")
        return False
    linear = _as_float(evidence.get("linear_scaling_efficiency"))
    if not _close(linear, speedup / world):
        blockers.append("matched_speed_linear_efficiency_disagrees_with_world")
        return False
    return True


def _close(left: float, right: float) -> bool:
    """Return whether two ratios agree to the precision a payload is written at.

    The efficiency is a quotient of two measured numbers, so it is compared with a
    relative tolerance rather than by equality: reading a payload is not the same
    as re-sealing it, and a bit-exact comparison would refuse a document that was
    rounded on the way through JSON.
    """

    if math.isnan(left) or math.isnan(right):
        return False
    return abs(left - right) <= 1e-9 * max(1.0, abs(right))


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


def _is_mps_evidence(evidence: Mapping[str, Any]) -> bool:
    """Return whether a payload is evidence for this capability at all.

    The runtime-evidence envelope carries no capability key, and the statevector
    release payloads are ``measured_production_run`` artifacts with
    ``release_gate_allowed`` set, so without a capability test another
    capability's payload would satisfy this contract's world requirement. A
    statevector payload declares no state mode at all, so the test is whether the
    payload declares one of the modes the audit vocabulary recognizes as MPS.
    Which of those the frozen contract accepts is a separate question, asked by
    the contract check and answered from the manifest.
    """

    return str(evidence.get("state_mode", "")).lower() in MPS_STATE_MODES


def frozen_workload_body_sha256(capacity: Mapping[str, Any]) -> str:
    """Return the digest of the body the manifest says builds the circuit.

    The launcher entry fixes the site count; the entry whose role names the
    workload body fixes the rank boundaries and the parameterization. A manifest
    that named no such entry would leave the circuit unbound, so this is a
    refusal rather than an empty string.
    """

    for source in capacity.get("workload_definition_sources", ()):
        if str(source["role"]).startswith("workload body"):
            return str(source["sha256"])
    raise ValueError(
        "capacity_workload.workload_definition_sources names no workload body, so "
        "the circuit a payload measured is not bound by this manifest"
    )


def _world_is_carried_by_envelope(
    evidence: Mapping[str, Any], provenance: Mapping[str, Any]
) -> bool:
    """Return whether the envelope attests the world the payload declares.

    The envelope's device inventory and rank mapping are what the sealer recorded
    about the machine that ran the rank, so they are the only reason to believe a
    payload's ``world_size``. A payload that asserts a wider world than its own
    signed provenance describes has described a run that was never sealed, and a
    world of zero devices is not a world at all.
    """

    world = _as_int(evidence.get("world_size"))
    if world <= 0:
        return False
    for name in ("devices", "rank_mapping"):
        value = provenance.get(name)
        if not isinstance(value, (list, tuple)) or len(value) != world:
            return False
    return True


def matched_speed_workload_sha256(manifest: Mapping[str, Any]) -> str:
    """Return the digest of the definition that freezes the timed ladder.

    The capacity workload cannot be timed on one device -- it exhausts one, which
    is what the premise asserts -- so a matched-speed payload is a comparison at
    reduced shapes of the same construction. Those shapes are frozen by the speed
    block, and the file that defines them is the workload a matched-speed payload
    measured, so its digest is what such a payload's envelope names.
    """

    return str(manifest["speed_workload"].get("workload_sha256", ""))


def _check_provenance_binding(
    evidence: Mapping[str, Any],
    provenance: Mapping[str, Any],
    capacity: Mapping[str, Any],
    blockers: list[str],
    *,
    speed_workload_sha256: str,
) -> None:
    """Report where the sealed provenance disagrees with the frozen contract.

    The envelope is signed over its own provenance, so the digest and revision it
    records are the ones the sealer attested. A payload whose envelope names a
    different workload, or a different revision than the payload does, is not
    evidence about the frozen workload however well the payload reads.

    A matched-speed payload is the one case where the envelope legitimately names
    a workload other than the capacity launcher, because the capacity workload
    cannot be timed on one device. The ladder digest is accepted only when the
    payload also carries the frozen capacity workload's own digest as the premise
    it rests on, so the exemption binds the payload to the premise instead of
    releasing it from the contract.
    """

    declared = str(evidence.get("capacity_premise_workload_sha256", ""))
    attested = str(provenance.get("workload_sha256", ""))
    names_the_ladder = (
        bool(speed_workload_sha256)
        and attested == speed_workload_sha256
        and declared == str(capacity["workload_sha256"])
    )
    if attested != str(capacity["workload_sha256"]) and not names_the_ladder:
        blockers.append("production_payload_provenance_workload_mismatch")
    if str(evidence.get("workload_body_sha256", "")) != str(
        frozen_workload_body_sha256(capacity)
    ):
        # The launcher digest pins the site count and nothing else. The body
        # builds the rank boundaries and the parameterization, so a payload that
        # named a launcher this manifest froze while measuring a different
        # circuit is not evidence about the frozen workload either.
        blockers.append("production_payload_workload_body_mismatch")
    declared_commit = evidence.get("commit")
    if not _is_absent(declared_commit) and str(provenance.get("commit", "")) != str(
        declared_commit
    ):
        blockers.append("production_payload_provenance_revision_mismatch")


def _is_fraction(value: Any) -> bool:
    """Return whether a value is a number in the unit interval a fraction lives in."""

    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and 0.0 <= float(value) <= 1.0
    )


def _ill_shaped_evidence(evidence: Mapping[str, Any], world: int) -> tuple[str, ...]:
    """Return the measurements that are present but do not shape as claims.

    Presence is not measurement. ``rank_gradients: [0.0, 0.0]``, a collective
    backend of ``"unknown"`` and a communication fraction of ``"unknown"`` all
    satisfy a key test while reporting nothing, and the per-rank memory and
    communication records are the claims this contract exists to check. Each
    blocker names the requirement rather than the field, and a field that is
    absent is reported by the required-field check instead, so this function only
    judges what is there.
    """

    per_rank = ("rank_outputs", "rank_gradients", "rank_timings")
    memory = ("rank_peak_memory_bytes", "measured_peak_memory_bytes_by_rank")
    not_well_formed = False
    for name in per_rank:
        value = evidence.get(name)
        if _is_absent(value):
            continue
        if (
            not isinstance(value, (list, tuple))
            or len(value) != world
            or any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                for item in value
            )
        ):
            not_well_formed = True
    for name in memory:
        value = evidence.get(name)
        if _is_absent(value):
            continue
        if (
            not isinstance(value, (list, tuple))
            or len(value) != world
            or any(
                isinstance(item, bool) or not isinstance(item, (int, float)) or item < 0
                for item in value
            )
        ):
            not_well_formed = True
    blockers: list[str] = []
    if not_well_formed:
        blockers.append("production_per_rank_measurements_not_well_formed")
    fraction = evidence.get("communication_fraction")
    if not _is_absent(fraction) and not _is_fraction(fraction):
        blockers.append("production_communication_fraction_not_well_formed")
    activity = evidence.get("gpu_activity")
    if not _is_absent(activity) and not _is_fraction(activity):
        blockers.append("production_gpu_activity_not_well_formed")
    return tuple(blockers)


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

    # The status has to be one the manifest froze as an executed reverse pass.
    # Matching on substrings of the label would accept "production_projected" and
    # "executed_never_actually", which report an intention in the vocabulary of a
    # measurement, so the manifest names the status the producer emits instead.
    backward_execution = str(evidence.get("backward_execution", "")).lower()
    accepted_backward = {
        str(status).lower()
        for status in manifest["runtime"]["accepted_backward_execution_statuses"]
    }
    if (
        backward_execution not in accepted_backward
        or evidence.get("backward_execution_measured") is not True
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

    # The collective backend is what actually carried the sharded state, and the
    # vocabulary of recognized backends is not the question: the manifest freezes
    # the one the release run must use, so a payload that names another one --
    # "unknown" included -- was measured on a different transport than this
    # contract is about.
    expected_backend = str(manifest["runtime"]["collective_backend"]).lower()
    declared_backend = evidence.get("collective_backend")
    if (
        _is_absent(declared_backend)
        or str(declared_backend).lower() != expected_backend
    ):
        blockers.append("collective_backend_mismatch")

    # A rank count and a host count that do not divide is not a placement, and a
    # payload whose per-host rank count differs from the frozen one was measured
    # on a different deployment from the one this contract is about.
    topology = manifest["topologies"]
    world = _as_int(evidence.get("world_size"))
    local = _as_int(evidence.get("local_world_size"))
    nodes = _as_int(evidence.get("node_count"))
    if local != _as_int(topology["local_world_size"]):
        blockers.append("local_world_size_mismatch")
    if local <= 0 or world <= 0 or world % local or world // local != nodes:
        blockers.append("rank_placement_inconsistent")

    # The shape is what makes a completion artifact evidence for *this* frozen
    # workload, so a missing shape field and a wrong one are reported apart: the
    # first is an incomplete record and the second is a different workload. A
    # matched-speed payload is the one payload whose shape is not the capacity
    # workload's, because the capacity workload cannot be timed on one device, so
    # its shape is checked against the frozen ladder instead.
    blockers.extend(_workload_shape_blockers(evidence, capacity, manifest))

    return tuple(blockers)


def _workload_shape_blockers(
    evidence: Mapping[str, Any],
    capacity: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return where a payload measured a shape this manifest did not freeze.

    Every payload this contract accepts is evidence about a frozen shape: the
    capacity workload's, or one rung of the frozen matched-speed ladder. Both
    checks report the same two blockers, because the requirement they serve is
    one requirement -- the payload measured a shape this manifest freezes -- and
    a reader of the report should not have to know which payload kind produced
    it to know what was wrong. The dtype is protocol rather than shape and is
    checked against the contract for both kinds, since a payload that trained in
    another precision trained another workload whatever its size.
    """

    blockers: list[str] = []
    if evidence.get("dtype") is None:
        blockers.append("missing_capacity_workload_shape_evidence")
    elif str(evidence["dtype"]) != str(capacity["dtype"]):
        blockers.append("capacity_workload_shape_mismatch")
    if evidence.get("acceptance_case") == "matched_speed":
        blockers.extend(_ladder_shape_blockers(evidence, capacity, manifest))
    else:
        blockers.extend(_capacity_shape_blockers(evidence, capacity))
    return tuple(blockers)


def _capacity_shape_blockers(
    evidence: Mapping[str, Any], capacity: Mapping[str, Any]
) -> tuple[str, ...]:
    """Return where a payload's shape is not the frozen capacity workload's."""

    blockers: list[str] = []
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
    return tuple(blockers)


def _ladder_shape_blockers(
    evidence: Mapping[str, Any],
    capacity: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return where a matched-speed payload did not time a frozen ladder rung.

    The comparison is only meaningful at a shape the manifest froze before the
    legs ran, so the payload's shape has to be one of the ladder's. The rung is
    identified by all three of its shape numbers together -- site count, bond
    dimension and trainable parameter count -- because a payload that matched a
    rung's site count while training a different parameterization would be a
    different circuit of the same size.
    """

    blockers: list[str] = []
    if _as_int(evidence.get("batch_size")) != _as_int(capacity["batch_size"]):
        blockers.append("capacity_workload_shape_mismatch")
    if evidence.get("n_sites") is None or evidence.get("max_bond_dimension") is None:
        blockers.append("missing_capacity_workload_shape_evidence")
        return tuple(blockers)
    ladder = tuple(manifest["speed_workload"].get("configuration_ladder") or ())
    if not ladder:
        blockers.append("missing_capacity_workload_shape_evidence")
        return tuple(blockers)
    claimed = (
        _as_int(evidence.get("n_sites")),
        _as_int(evidence.get("max_bond_dimension")),
        _as_int(evidence.get("parameter_count")),
    )
    frozen = {
        (
            _as_int(rung.get("n_sites")),
            _as_int(rung.get("trained_max_bond")),
            _as_int(rung.get("parameter_count")),
        )
        for rung in ladder
    }
    if claimed not in frozen:
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
        # counted and reported rather than dropped, because a directory holding
        # five files of which none is admissible must not read as an empty set
        # one blocker away from passing.
        if not _is_mps_evidence(evidence):
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
        blockers.append("production_artifact_is_not_mps_evidence")

    # A payload counts towards a release world only when its signed envelope
    # carries that many devices, so the frozen world is bound to an attestation
    # rather than to a number the payload declares about itself.
    bound_production = [
        (evidence, provenance)
        for evidence, provenance in production
        if _world_is_carried_by_envelope(evidence, provenance)
    ]
    if len(bound_production) != len(production):
        blockers.append("production_payload_world_size_not_carried_by_envelope")

    worlds = {_as_int(evidence.get("world_size")) for evidence, _ in bound_production}
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
    # The scaling threshold is a ratio against a ceiling the timed workload's own
    # fitted serial fraction implies, so that fraction has to be re-readable from
    # the artifact this contract is digest-bound to. A contract whose calibration
    # moved is not one whose threshold can be evaluated, and the gate names that
    # rather than comparing every payload against a number nobody can reproduce.
    calibration = calibration_errors(speed, REPOSITORY_ROOT)
    if calibration:
        blockers.append("speed_scaling_calibration_not_verifiable")
    qualified = []
    for evidence, _ in production:
        if evidence.get("acceptance_case") != "matched_speed":
            continue
        if _as_float(evidence.get("speedup")) < float(speed["minimum_speedup"]):
            continue
        if _confidence_interval_lower(evidence) <= float(
            speed["confidence_interval_must_exclude_speedup"]
        ):
            continue
        if _as_float(evidence.get("scaling_efficiency")) < float(
            speed["minimum_scaling_efficiency"]
        ):
            continue
        # The efficiency is a ratio against a ceiling the workload's fitted serial
        # fraction implies, so a payload reporting the definition this contract
        # replaced is refused outright rather than read as a large number: against
        # the old denominator a shardable workload and this one are not
        # distinguished, and a payload cannot clear a measured threshold by
        # choosing the denominator that flatters it.
        if calibration:
            continue
        if not _efficiency_matches_frozen_ceiling(evidence, speed, blockers):
            continue
        qualified.append(evidence)
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

    # The premise is a claim about files, so it is re-read rather than believed.
    # ``premise_provenance`` discloses which of the sources it cites resolve and
    # which have drifted; reporting that here is what keeps the disclosure from
    # being decorative. This contract's raw log and device telemetry now resolve
    # at the digests the premise recorded, so the only source left unverifiable is
    # the workload body the premise was assembled from, which was redesigned after
    # that run and cannot resolve against a digest recorded before the redesign.
    if premise_evidence_errors(capacity):
        blockers.append("capacity_premise_evidence_not_verifiable")

    for evidence, _ in baselines:
        if any(
            _as_int(evidence.get(name)) != _as_int(capacity[name])
            for name in ("n_sites", "logical_mps_bytes")
        ) or _as_int(evidence.get("max_bond_dimension")) != _as_int(
            capacity["trained_max_bond"]
        ):
            blockers.append("single_gpu_baseline_disagrees_with_frozen_premise")
        if _as_int(evidence.get("measured_peak_memory_bytes")) != _as_int(
            capacity["measured_single_device_peak_memory_bytes"]
        ):
            blockers.append("single_gpu_baseline_disagrees_with_frozen_premise")
        if _as_int(evidence.get("device_total_memory_bytes")) != _as_int(
            capacity["single_device_total_memory_bytes"]
        ):
            blockers.append("single_gpu_baseline_disagrees_with_frozen_premise")
        # The baseline has to have failed on the circuit the premise is about: a
        # device exhausted by a different parameterization is a measurement of a
        # different workload, whatever its site count and bond dimension say.
        if str(evidence.get("workload_body_sha256", "")) != (
            frozen_workload_body_sha256(capacity)
        ) or _as_int(evidence.get("parameter_count")) != _as_int(
            capacity["parameter_count"]
        ):
            blockers.append("single_gpu_baseline_disagrees_with_frozen_premise")

    completion = any(
        evidence.get("acceptance_case") == "multi_gpu_capacity_completion"
        and str(evidence.get("workload_sha256")) == str(capacity["workload_sha256"])
        for evidence, _ in production
    )
    if capacity["premise_established"] and not completion:
        blockers.append("missing_multi_gpu_capacity_completion_artifact")

    for evidence, provenance in production:
        blockers.extend(_mps_contract_blockers(evidence, manifest))
        _check_provenance_binding(
            evidence,
            provenance,
            capacity,
            blockers,
            speed_workload_sha256=matched_speed_workload_sha256(manifest),
        )
        if _has_required_fields(evidence, provenance, required):
            blockers.append("production_artifact_missing_required_fields")
        blockers.extend(
            _ill_shaped_evidence(evidence, _as_int(evidence.get("world_size")))
        )
        # The sealer records observed fallbacks in the envelope's provenance and
        # exempts that field from its completeness check, so a payload whose
        # evidence reports none is not the same as a run that took none.
        if provenance.get("fallback_events") not in ((), [], None):
            blockers.append("production_payload_declares_fallback_events")

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
