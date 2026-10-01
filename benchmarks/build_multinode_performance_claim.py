#!/usr/bin/env python
"""Build the two-node performance-claim artifact from the recorded evidence.

The three recorded multi-node artifacts each carry a measured forward leg, and
each of them reports the measurement inside a payload whose job is to describe a
correctness run. A capability claim needs a smaller thing: one artifact whose
selectors a reader can resolve, whose ``commit`` names the revision the numbers
were taken at, and whose scope says what the timer covered. This builds that
artifact, and it builds it only from a measurement it can prove was taken.

Nothing here re-measures. Every number is copied out of a recorded artifact and
checked before it is copied, so a claim cannot describe a run that failed to
produce one:

* the artifact digest matches its own evidence;
* the route was read from the NCCL debug log as an observed fabric route rather
  than the socket fallback or an unobserved default;
* the measurement declares itself measured and synchronized, and carries a
  warmup plus at least three samples whose recorded minimum, median and maximum
  are consistent with the samples themselves;
* the run recorded one rank record per rank, each naming the work that rank
  owned and the memory and traffic it carried on the forward leg the timer
  covered, rather than around it;
* both claim flags are false, so the artifact cannot be mistaken for scalability
  or release evidence;
* the blockers do not include ``production_performance_not_measured``, the one
  blocker that directly contradicts a measurement claim;
* all three artifacts name the same source revision, because a claim that mixed
  revisions would not describe one thing.

The result is a non-release benchmark payload in its own right: it reports
``distribution_semantics``, the world, local world and node counts, per-rank
ownership, memory and communication, and its blockers, so the checked-in result
audit inspects it instead of skipping it, and it still cannot be promoted.

Repro commands
-------------
Rebuild the claim artifact from the recorded evidence:
  python benchmarks/build_multinode_performance_claim.py

Fail unless the committed artifact matches what the evidence implies:
  python benchmarks/build_multinode_performance_claim.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

SCHEMA = "flagquantum.multinode_performance_claim.v1"
BENCHMARK = "cuda_multinode_two_node_performance"
DEFAULT_OUTPUT = "benchmarks/results/local/multinode_two_node_performance_20261001.json"

EVIDENCE_CLASS = "local_non_release"
CLAIM_EVIDENCE_TYPE = "development_smoke"
DISTRIBUTION_SEMANTICS = "sharded_across_ranks"
OBSERVED_ROUTE = "infiniband"
# The blocker a measurement retracts, and therefore the blocker whose presence
# would make this artifact describe a run that never measured anything.
CONTRADICTED_BLOCKER = "production_performance_not_measured"
MINIMUM_WARMUP_ITERATIONS = 1
MINIMUM_MEASURED_ITERATIONS = 3


class ClaimBuildError(RuntimeError):
    """Raised when the recorded evidence cannot support a performance claim."""


@dataclass(frozen=True)
class RankEvidence:
    """What one rank of one measured workload owned and carried."""

    local_memory_bytes: int
    communication_bytes: int
    owned_work: str


@dataclass(frozen=True)
class Workload:
    """One measured workload and the selectors that read its rank records."""

    name: str
    artifact: str
    metric_key: str
    rank_evidence: Callable[[str, Mapping[str, Any], int], RankEvidence]


def _mapping(relative: str, field: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ClaimBuildError(f"{relative}: {field} is not an object")
    return value


def _positive_int(relative: str, field: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ClaimBuildError(
            f"{relative}: {field} is not a positive byte count: {value!r}"
        )
    return value


def _nonempty_text(relative: str, field: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ClaimBuildError(f"{relative}: {field} is not a description: {value!r}")
    return value


def _statevector_rank_evidence(
    relative: str, record: Mapping[str, Any], rank: int
) -> RankEvidence:
    return RankEvidence(
        local_memory_bytes=_positive_int(
            relative, "local_state_bytes", record.get("local_state_bytes")
        ),
        communication_bytes=_positive_int(
            relative,
            "inter_node_communication_bytes",
            record.get("inter_node_communication_bytes"),
        ),
        owned_work=_nonempty_text(
            relative,
            "owned_global_basis_indices",
            record.get("owned_global_basis_indices"),
        ),
    )


def _mps_rank_evidence(
    relative: str, record: Mapping[str, Any], rank: int
) -> RankEvidence:
    ownership = _mapping(relative, "rank_ownership", record.get("rank_ownership"))
    communication = _mapping(relative, "communication", record.get("communication"))
    sites = record.get("owned_sites")
    if (
        not isinstance(sites, list)
        or not sites
        or not all(
            isinstance(site, int) and not isinstance(site, bool) for site in sites
        )
    ):
        raise ClaimBuildError(f"{relative}: rank {rank} reports no owned sites")
    return RankEvidence(
        local_memory_bytes=_positive_int(
            relative,
            "rank_ownership.local_tensor_bytes",
            ownership.get("local_tensor_bytes"),
        ),
        communication_bytes=_positive_int(
            relative,
            "communication.forward_communication_bytes",
            communication.get("forward_communication_bytes"),
        ),
        owned_work="sites " + ",".join(str(site) for site in sites),
    )


def _tensor_network_rank_evidence(
    relative: str, record: Mapping[str, Any], rank: int
) -> RankEvidence:
    amplitudes = _mapping(relative, "amplitudes", record.get("amplitudes"))
    tiers = _mapping(
        relative,
        "amplitudes.communication_tiers",
        amplitudes.get("communication_tiers"),
    )
    memory = amplitudes.get("local_memory_bytes_by_rank")
    if not isinstance(memory, list) or len(memory) <= rank:
        raise ClaimBuildError(
            f"{relative}: no local memory reported for rank {rank} of {len(memory) if isinstance(memory, list) else 0}"
        )
    tasks = _mapping(
        relative, "amplitudes.tasks_by_rank", amplitudes.get("tasks_by_rank")
    )
    owned = tasks.get(str(rank))
    slice_tasks = amplitudes.get("slice_tasks")
    if isinstance(owned, bool) or not isinstance(owned, int) or owned <= 0:
        raise ClaimBuildError(f"{relative}: rank {rank} owns no slice tasks: {owned!r}")
    if (
        isinstance(slice_tasks, bool)
        or not isinstance(slice_tasks, int)
        or slice_tasks <= 0
    ):
        raise ClaimBuildError(
            f"{relative}: the slice task count is not a positive integer: {slice_tasks!r}"
        )
    return RankEvidence(
        local_memory_bytes=_positive_int(
            relative,
            f"amplitudes.local_memory_bytes_by_rank[{rank}]",
            memory[rank],
        ),
        communication_bytes=_positive_int(
            relative,
            "amplitudes.communication_tiers.inter_node_collective_bytes",
            tiers.get("inter_node_collective_bytes"),
        ),
        owned_work=f"{owned} of {slice_tasks} slice tasks",
    )


# One row per workload. `metric_key` names the prefix that workload's latency
# selectors use, so each capability's claim can render its own numbers out of
# one shared artifact.
WORKLOADS: tuple[Workload, ...] = (
    Workload(
        name="statevector",
        artifact="artifacts/cuda_multinode_statevector_a800_jp171_jp172_20260930.json",
        metric_key="statevector_forward",
        rank_evidence=_statevector_rank_evidence,
    ),
    Workload(
        name="mps",
        artifact="artifacts/cuda_multinode_mps_a800_jp171_jp172_20260930.json",
        metric_key="mps_forward",
        rank_evidence=_mps_rank_evidence,
    ),
    Workload(
        name="tensor_network",
        artifact="artifacts/cuda_multinode_tn_a800_jp171_jp172_20260930.json",
        metric_key="tensor_network_amplitudes",
        rank_evidence=_tensor_network_rank_evidence,
    ),
)


def _canonical_sha256(evidence: dict[str, Any]) -> str:
    encoded = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ClaimBuildError(f"recorded artifact is missing: {relative}") from error
    except json.JSONDecodeError as error:
        raise ClaimBuildError(
            f"recorded artifact is not valid JSON: {relative}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ClaimBuildError(f"recorded artifact is not a JSON object: {relative}")
    return payload


def _performance(relative: str, evidence: dict[str, Any]) -> dict[str, Any]:
    """Return the measurement, having checked every property a claim relies on."""

    observations = evidence.get("observations")
    if not isinstance(observations, dict):
        raise ClaimBuildError(f"{relative}: evidence carries no observations")
    performance = observations.get("performance")
    if not isinstance(performance, dict):
        raise ClaimBuildError(f"{relative}: the run recorded no performance leg")
    if performance.get("measured") is not True:
        raise ClaimBuildError(f"{relative}: the performance leg did not measure")
    if performance.get("synchronized") is not True:
        raise ClaimBuildError(f"{relative}: the samples were not synchronized")
    warmup = performance.get("warmup_iterations")
    iterations = performance.get("measured_iterations")
    if not isinstance(warmup, int) or warmup < MINIMUM_WARMUP_ITERATIONS:
        raise ClaimBuildError(
            f"{relative}: a measurement needs at least "
            f"{MINIMUM_WARMUP_ITERATIONS} warmup iteration, recorded {warmup!r}"
        )
    if not isinstance(iterations, int) or iterations < MINIMUM_MEASURED_ITERATIONS:
        raise ClaimBuildError(
            f"{relative}: a measurement needs at least "
            f"{MINIMUM_MEASURED_ITERATIONS} samples, recorded {iterations!r}"
        )
    seconds = performance.get("seconds")
    if not isinstance(seconds, list) or len(seconds) != iterations:
        raise ClaimBuildError(
            f"{relative}: recorded {len(seconds) if isinstance(seconds, list) else 0} "
            f"samples against {iterations} measured iterations"
        )
    if not all(isinstance(value, (int, float)) and value > 0 for value in seconds):
        raise ClaimBuildError(f"{relative}: every sample must be a positive number")
    minimum = performance.get("minimum_seconds")
    median = performance.get("median_seconds")
    maximum = performance.get("maximum_seconds")
    if minimum != min(seconds) or maximum != max(seconds):
        raise ClaimBuildError(
            f"{relative}: the recorded extremes disagree with the samples"
        )
    if not isinstance(median, (int, float)) or not minimum <= median <= maximum:
        raise ClaimBuildError(f"{relative}: the median is outside the sample range")
    return performance


def _route(relative: str, evidence: dict[str, Any]) -> str:
    """Return the observed fabric route, failing closed on anything weaker."""

    observations = evidence.get("observations")
    network = observations.get("network") if isinstance(observations, dict) else None
    if not isinstance(network, dict):
        raise ClaimBuildError(f"{relative}: the run recorded no network observation")
    if network.get("evidence_level") != "observed_debug_log":
        raise ClaimBuildError(
            f"{relative}: the route is not observed evidence, it is "
            f"{network.get('evidence_level')!r}"
        )
    if network.get("route") != OBSERVED_ROUTE:
        raise ClaimBuildError(
            f"{relative}: the observed route is {network.get('route')!r}, "
            f"not {OBSERVED_ROUTE!r}"
        )
    if network.get("socket_transport_observed") is not False:
        raise ClaimBuildError(f"{relative}: the socket fallback was observed")
    if network.get("infiniband_transport_observed") is not True:
        raise ClaimBuildError(f"{relative}: no fabric transport was observed")
    return OBSERVED_ROUTE


def _rank_records(
    relative: str, evidence: dict[str, Any], expected: Any
) -> list[Mapping[str, Any]]:
    """Return one rank record per rank, or fail closed."""

    observations = evidence.get("observations")
    records = (
        observations.get("rank_records") if isinstance(observations, dict) else None
    )
    if not isinstance(records, list) or not records:
        raise ClaimBuildError(f"{relative}: the run recorded no per-rank evidence")
    if not isinstance(expected, int) or len(records) != expected:
        raise ClaimBuildError(
            f"{relative}: recorded {len(records)} rank records for a world size of "
            f"{expected!r}"
        )
    if not all(isinstance(record, Mapping) for record in records):
        raise ClaimBuildError(f"{relative}: a rank record is not an object")
    return records


def _row(workload: Workload) -> dict[str, Any]:
    relative = workload.artifact
    payload = _load(relative)
    evidence = payload.get("evidence")
    if not isinstance(evidence, dict):
        raise ClaimBuildError(f"{relative}: the artifact carries no evidence")
    digest = _canonical_sha256(evidence)
    if payload.get("evidence_sha256") != digest:
        raise ClaimBuildError(f"{relative}: the evidence digest does not match")
    if evidence.get("status") != "passed":
        raise ClaimBuildError(
            f"{relative}: the run did not pass, it reported {evidence.get('status')!r}"
        )
    for flag in ("scalability_claim_allowed", "release_gate_allowed"):
        if evidence.get(flag) is not False:
            raise ClaimBuildError(f"{relative}: {flag} is not false")
    blockers = evidence.get("claim_blockers")
    if not isinstance(blockers, list) or not all(
        isinstance(item, str) for item in blockers
    ):
        raise ClaimBuildError(f"{relative}: the claim blockers are not a list of names")
    if CONTRADICTED_BLOCKER in blockers:
        raise ClaimBuildError(
            f"{relative}: the run still reports {CONTRADICTED_BLOCKER}"
        )
    performance = _performance(relative, evidence)
    route = _route(relative, evidence)
    scope = evidence.get("scope")
    environment = evidence.get("environment")
    if not isinstance(scope, dict) or not isinstance(environment, dict):
        raise ClaimBuildError(
            f"{relative}: the artifact carries no scope or environment"
        )
    world_size = scope.get("world_size")
    records = _rank_records(relative, evidence, world_size)
    rank_evidence = [
        {
            "rank": rank,
            "local_memory_bytes": item.local_memory_bytes,
            "communication_bytes": item.communication_bytes,
            "owned_work": item.owned_work,
        }
        for rank, record in enumerate(records)
        for item in (workload.rank_evidence(relative, record, rank),)
    ]
    device_names = {record.get("device_name") for record in records}
    if len(device_names) != 1:
        raise ClaimBuildError(
            f"{relative}: the ranks disagree on the device: {sorted(map(str, device_names))}"
        )
    return {
        "name": workload.name,
        "artifact": relative,
        "artifact_sha256": hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
        "metric_key": workload.metric_key,
        "captured_at": evidence.get("captured_at"),
        "measurement": performance.get("measurement"),
        "n_wires": performance.get("n_wires"),
        "max_bond": performance.get("max_bond"),
        "device_name": device_names.pop(),
        "world_size": world_size,
        "local_world_size": scope.get("local_world_size"),
        "node_count": scope.get("node_count"),
        "dtype": scope.get("dtype"),
        "distribution_semantics": scope.get("distribution_semantics"),
        "route": route,
        "synchronized": performance.get("synchronized"),
        "measured": performance.get("measured"),
        "warmup_iterations": performance.get("warmup_iterations"),
        "measured_iterations": performance.get("measured_iterations"),
        "seconds": [float(value) for value in performance["seconds"]],
        "minimum_seconds": float(performance["minimum_seconds"]),
        "median_seconds": float(performance["median_seconds"]),
        "maximum_seconds": float(performance["maximum_seconds"]),
        "claim_blockers": sorted(blockers),
        "rank_evidence": rank_evidence,
        "source_revision": environment.get("source_revision"),
        "torch": environment.get("torch"),
        "nccl": environment.get("nccl"),
        "cuda_runtime": environment.get("cuda_runtime"),
        "python": environment.get("python"),
    }


def _uniform(rows: list[dict[str, Any]], key: str) -> Any:
    """Return the value every row agrees on, or fail closed."""

    values = {row[key] for row in rows}
    if len(values) != 1:
        raise ClaimBuildError(
            f"the recorded artifacts disagree on {key}: {sorted(map(str, values))}"
        )
    return rows[0][key]


def _positive_int_field(field: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ClaimBuildError(
            f"the recorded {field} is not a positive integer: {value!r}"
        )
    return value


def build_payload() -> dict[str, Any]:
    rows = [_row(workload) for workload in WORKLOADS]
    revisions = {str(row["source_revision"]) for row in rows}
    if len(revisions) != 1:
        raise ClaimBuildError(
            "the recorded artifacts name different source revisions: "
            + ", ".join(sorted(revisions))
        )
    revision = revisions.pop()
    if len(revision) != 40 or not all(char in "0123456789abcdef" for char in revision):
        raise ClaimBuildError(
            f"the source revision is not a full commit sha: {revision}"
        )
    routers = {str(row["route"]) for row in rows}
    if routers != {OBSERVED_ROUTE}:
        raise ClaimBuildError(f"the recorded routes disagree: {sorted(routers)}")
    semantics = _uniform(rows, "distribution_semantics")
    if semantics != DISTRIBUTION_SEMANTICS:
        raise ClaimBuildError(
            f"the recorded distribution semantics are {semantics!r}, "
            f"not {DISTRIBUTION_SEMANTICS!r}"
        )
    accelerator = _uniform(rows, "device_name")
    if not isinstance(accelerator, str) or not accelerator:
        raise ClaimBuildError(
            f"the recorded device name is not a name: {accelerator!r}"
        )
    world_size = _positive_int_field("world size", _uniform(rows, "world_size"))
    local_world_size = _positive_int_field(
        "local world size", _uniform(rows, "local_world_size")
    )
    node_count = _positive_int_field("node count", _uniform(rows, "node_count"))
    if world_size < 2:
        raise ClaimBuildError(
            f"a multi-node claim needs a world size above one, recorded {world_size}"
        )
    if local_world_size * node_count < world_size:
        raise ClaimBuildError(
            f"the recorded layout cannot host a world size of {world_size}: "
            f"{local_world_size} devices on each of {node_count} nodes"
        )
    blockers = sorted({blocker for row in rows for blocker in row["claim_blockers"]})
    latency: dict[str, float] = {}
    for row in rows:
        latency[f"{row['metric_key']}_minimum"] = row["minimum_seconds"]
        latency[f"{row['metric_key']}_median"] = row["median_seconds"]
        latency[f"{row['metric_key']}_maximum"] = row["maximum_seconds"]
    sample_seconds = sorted({float(value) for row in rows for value in row["seconds"]})
    rank_shards = [
        {
            "rank": rank,
            "local_rank": rank % local_world_size,
            "node_rank": rank // local_world_size,
            "local_memory_bytes": max(
                row["rank_evidence"][rank]["local_memory_bytes"] for row in rows
            ),
            "communication_bytes": sum(
                row["rank_evidence"][rank]["communication_bytes"] for row in rows
            ),
            "owned_work": {
                row["name"]: row["rank_evidence"][rank]["owned_work"] for row in rows
            },
        }
        for rank in range(world_size)
    ]
    return {
        "schema": SCHEMA,
        "benchmark": BENCHMARK,
        "generated_by": "benchmarks/build_multinode_performance_claim.py",
        "benchmark_evidence_class": EVIDENCE_CLASS,
        "claim_evidence_type": CLAIM_EVIDENCE_TYPE,
        "non_release_evidence": True,
        "distribution_semantics": semantics,
        "world_size": world_size,
        "local_world_size": local_world_size,
        "node_count": node_count,
        "commit": revision,
        "latency_seconds": latency,
        "workloads": [
            {key: value for key, value in row.items() if key != "metric_key"}
            for row in rows
        ],
        "rank_shards": rank_shards,
        "communication_bytes": sum(
            shard["communication_bytes"] for shard in rank_shards
        ),
        "scalability_blockers": blockers,
        "environment": {
            "accelerator": accelerator,
            "cuda_runtime": rows[0]["cuda_runtime"],
            "dtype": rows[0]["dtype"],
            "local_world_size": local_world_size,
            "measured_latencies": len(latency) // 3,
            "measured_workloads": len(rows),
            "nccl": rows[0]["nccl"],
            "node_count": node_count,
            "python": rows[0]["python"],
            "route": OBSERVED_ROUTE,
            "sample_count_minimum": min(row["measured_iterations"] for row in rows),
            "sample_count_maximum": max(row["measured_iterations"] for row in rows),
            "slowest_sample_seconds": sample_seconds[-1],
            "torch": rows[0]["torch"],
            "warmup_iterations": rows[0]["warmup_iterations"],
            "world_size": world_size,
        },
        "scope": (
            "One sharded forward pass of each recorded two-node workload, timed "
            "between CUDA synchronizations on the recorded pair. This is a "
            "latency of a small circuit, not a scaling, throughput, or capacity "
            "claim, and both claim flags are false."
        ),
        "derivation": (
            "Each latency is copied from the recorded artifact named in the "
            "matching workloads[] entry, whose digest is recorded there as "
            "artifact_sha256. rank_shards[].local_memory_bytes is the largest "
            "local footprint that rank recorded across the three measured "
            "workloads, and communication_bytes sums the recorded per-rank "
            "traffic of all three."
        ),
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
    }


def serialize(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail unless the committed artifact matches the recorded evidence",
    )
    args = parser.parse_args(argv)
    try:
        text = serialize(build_payload())
    except ClaimBuildError as error:
        print(f"performance claim refused: {error}", file=sys.stderr)
        return 2
    path = ROOT / args.output
    if args.check:
        try:
            committed = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"performance claim is not committed: {args.output}", file=sys.stderr)
            return 2
        if committed != text:
            print(
                f"performance claim is stale: rerun without --check to rebuild "
                f"{args.output}",
                file=sys.stderr,
            )
            return 2
        print(f"performance claim matches {args.output}")
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
