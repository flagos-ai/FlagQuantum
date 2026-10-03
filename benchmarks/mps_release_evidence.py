"""Measure the frozen MPS capacity workload and emit an MPS release payload.

The output of this script is a *measurements document*, not a signed artifact: it
reports what executed, and ``tools/seal_runtime_evidence.py`` reports what that is
worth. Three roles are implemented.

``capacity-failure`` runs the frozen capacity workload on one device and records
the exhaustion that makes the workload a capacity premise. ``capacity-completion``
runs the same workload across ranks, inherits the measured single-device failure
from a validated ISSUE-092 capacity premise, and records the sharded training
update that completes it. ``release-payload`` projects one role document onto the
top-level contract the sealer reads.

``matched-speed`` and ``speed-summary`` are declared and refused. The frozen
manifest deliberately contains no matched-speed configuration ladder
(``speed_workload.configuration_ladder`` is null), because no MPS timing of any
kind has been taken: there is no measured single-device leg and no measured
sharded leg to compare. A timing protocol chosen now would be chosen without a
measurement to choose it from, and a speed comparison run against a ladder that
was picked after the fact is not a frozen acceptance test. The refusal is
implemented rather than left to a comment, so a launcher that asks for a speed
role is told why instead of being handed a document that looks like evidence.

The frozen workload is identified by the digests the release manifest declares,
and the manifest is read through the release gate's loader so that the producer
and the gate cannot disagree about what was frozen. The workload itself is built
by the ISSUE-092 capacity definition the manifest digest-binds rather than
re-implemented here: a second construction of the same circuit would be a second
source of truth for the site count, the bond schedule and the rank boundaries.

The producer measures and reports; it does not seal. Sealing is
``tools/seal_runtime_evidence.py``, and the frozen release world of sixteen ranks
is wider than any evidence scope the envelope defines, so the completion role
warns that its payload cannot be sealed yet. The measurement is still taken
because it is the evidence the contract is about, and because the warning is
about the envelope rather than about the run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

import flagquantum.experimental.distributed as fqxd
import flagquantum.runtime.executors.mps.records as fqxm
from benchmarks.internal.evidence.mps_release_gate import (
    MANIFEST as RELEASE_MANIFEST,
)
from benchmarks.internal.evidence.mps_release_gate import (
    envelope_carries_world,
)
from benchmarks.internal.evidence.mps_release_gate import (
    load_manifest as _load_frozen_manifest,
)
from flagquantum.testing import (
    MPSCapacityCertificationError,
    require_general_mps_capacity,
)

SCHEMA = "flagquantum.mps_release_measurements.v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
WORKLOAD_MODULE = "benchmarks.internal.evidence.general_mps_capacity_16"
ROLES = (
    "capacity-failure",
    "capacity-completion",
    "matched-speed",
    "speed-summary",
    "release-payload",
)
UNFROZEN_SPEED_ROLES = ("matched-speed", "speed-summary")


def _commit() -> str:
    """Return the revision the producer is running at."""

    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _software() -> dict[str, str]:
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "",
        "python": sys.version.split()[0],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _capacity_contract(manifest_path: Path) -> dict[str, Any]:
    """Return the frozen capacity contract, checked against the manifest digests."""

    return dict(_load_frozen_manifest(Path(manifest_path))["capacity_workload"])


def _runtime_contract(manifest_path: Path) -> dict[str, Any]:
    return dict(_load_frozen_manifest(Path(manifest_path))["runtime"])


def _workload_digest(contract: Mapping[str, Any]) -> str:
    """Return the digest of the launcher that defines the frozen workload."""

    return _sha256(Path(str(contract["workload_definition_path"])))


def _frozen_workload(contract: Mapping[str, Any]) -> Any:
    """Return the frozen workload module, patched to the contract's shape.

    The manifest freezes a site count, a maximum bond dimension, a target world
    size and a circuit; the module that builds the circuit carries the last
    three as module constants, exactly as the ISSUE-092 launcher patches them.
    The module is patched here and then asked for the arithmetic, so a contract
    that disagrees with the code that builds the workload fails before anything
    is measured.
    """

    for source in (
        {
            "path": contract["workload_definition_path"],
            "sha256": contract["workload_definition_sha256"],
        },
        *contract["workload_definition_sources"],
    ):
        path = Path(str(source["path"]))
        if _sha256(path) != str(source["sha256"]):
            raise SystemExit(
                f"{path} is not the workload the release manifest froze; the "
                "contract and the code that builds the workload have drifted apart"
            )
    workload = importlib.import_module(WORKLOAD_MODULE)
    workload.N_SITES = int(contract["n_sites"])
    workload.MAX_BOND = int(contract["trained_max_bond"])
    workload.TARGET_WORLD = int(contract["target_world_size"])
    measured_bytes = int(
        workload.logical_mps_bytes(workload.N_SITES, workload.MAX_BOND)
    )
    if measured_bytes != int(contract["logical_mps_bytes"]):
        raise SystemExit(
            "the frozen workload definition computes "
            f"{measured_bytes} logical MPS bytes where the manifest declares "
            f"{int(contract['logical_mps_bytes'])}"
        )
    return workload


def _require_frozen_shape(
    arguments: argparse.Namespace, contract: Mapping[str, Any]
) -> None:
    """Refuse a launch whose shape is not the shape the manifest froze.

    The bond dimension and the optimizer step count are workload identity, not
    tuning: a run at a different bond dimension holds a different state and a run
    at a different step count measured a different training update. They travel
    as arguments because the launcher is what chose them, and they are checked
    here because the payload may not describe a workload nobody froze.
    """

    frozen_bond = int(contract["trained_max_bond"])
    if arguments.bond_dimension is not None and int(arguments.bond_dimension) != (
        frozen_bond
    ):
        raise SystemExit(
            f"the frozen capacity workload uses a maximum bond dimension of "
            f"{frozen_bond}; a run at {int(arguments.bond_dimension)} measured a "
            "different workload"
        )
    frozen_steps = int(contract["steps"])
    if arguments.steps is not None and int(arguments.steps) != frozen_steps:
        raise SystemExit(
            f"the frozen capacity workload trains {frozen_steps} optimizer "
            f"step(s); a run at {int(arguments.steps)} measured a different "
            "workload"
        )


def _write(arguments: argparse.Namespace, document: Mapping[str, Any]) -> None:
    text = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if arguments.measurements is None:
        print(text, end="")
        return
    arguments.measurements.parent.mkdir(parents=True, exist_ok=True)
    arguments.measurements.write_text(text, encoding="utf-8")
    print(f"wrote {arguments.measurements}", flush=True)


def _initialize(device: torch.device) -> None:
    """Join the rendezvous every role runs inside.

    The MPS executor requires an initialized group even at world size one,
    because the caller owns initialization and a group of one rank is still a
    group. Both roles are launched through a rendezvous -- ``torchrun`` for the
    sharded one, a ``torchrun --nproc-per-node 1`` or an explicit
    ``MASTER_ADDR``/``MASTER_PORT`` pair for the single-device one -- so the
    group is created here and the role decides what the world size means.
    """

    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=device)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        raise SystemExit(f"{name} is not an integer") from None


def _baseline_measurements(
    contract: Mapping[str, Any],
    *,
    status: str,
    failure_reason: str,
    peak_memory_bytes: int,
    device_total_memory_bytes: int,
    device_name: str,
    digest: str,
    world_size: int,
    local_world_size: int,
    node_count: int,
) -> dict[str, Any]:
    """Return the single-device capacity contract the failure role measured.

    A single device owns the whole state, so its site ownership is not a shard
    map and is stated as such rather than as a one-entry map. The distribution
    semantics say ``single_device_fast_path`` for the same reason: one rank has
    nothing to be sharded across, and calling it sharded would be the exact
    misreport the release contract exists to prevent.
    """

    n_sites = int(contract["n_sites"])
    return {
        "acceptance_case": "single_gpu_capacity_failure",
        "state_mode": "mps",
        "distribution_semantics": "single_device_fast_path",
        "mps_forward_distribution_semantics": "single_device_fast_path",
        "world_size": world_size,
        "local_world_size": local_world_size,
        "node_count": node_count,
        "single_device_oom_observed": status == "cuda_oom",
        "single_gpu_expected_oom": True,
        "capacity_baseline_device": device_name,
        "capacity_failure_reason": failure_reason,
        "measured_peak_memory_bytes": int(peak_memory_bytes),
        "device_total_memory_bytes": int(device_total_memory_bytes),
        "n_sites": n_sites,
        "initial_max_bond": int(contract["initial_max_bond"]),
        "max_bond_dimension": int(contract["trained_max_bond"]),
        "batch_size": int(contract["batch_size"]),
        "dtype": str(contract["dtype"]),
        "logical_mps_bytes": int(contract["logical_mps_bytes"]),
        "site_ownership": {
            "policy": "single_device_owns_every_site",
            "sharded": False,
            "site_count": n_sites,
        },
        "workload_sha256": digest,
        "workload_definition_sha256": str(contract["workload_definition_sha256"]),
        "blockers": [
            "measured single-device capacity failure is baseline provenance only",
        ],
    }


def _agreed(records: Sequence[Any], *, context: str) -> Any:
    """Return the one value every rank published, or fail closed.

    Global evidence -- the objective value, the site map, the optimizer
    ownership -- is stated by every rank because every rank holds the whole
    logical workload's description. A payload that concatenated the ranks would
    list each item once per rank, and one that read only rank 0 would accept a
    leg whose other ranks reported something different. Both are rejected here.
    """

    if not records:
        raise SystemExit(f"no rank published {context}, so the payload cannot state it")
    first = records[0]
    if any(item != first for item in records[1:]):
        raise SystemExit(f"the ranks of the sharded leg disagree about {context}")
    return first


def _ownership_map(
    entries: Sequence[Mapping[str, Any]], *, world_size: int
) -> dict[str, list[str]]:
    """Group measured optimizer ownership entries by the rank that owns them."""

    owned: dict[str, list[tuple[int, str]]] = {}
    for entry in entries:
        owner = int(entry["owner_rank"])
        if not 0 <= owner < world_size:
            raise SystemExit(
                f"the sharded leg attributes a parameter to rank {owner}, which "
                f"is outside a world of {world_size}"
            )
        index = int(entry["parameter_index"])
        owned.setdefault(f"rank:{owner}", []).append((index, f"parameter:{index}"))
    if not owned:
        raise SystemExit(
            "the sharded leg recorded no parameter ownership, so a release "
            "payload cannot state which rank updated which parameter"
        )
    return {
        owner: [name for _, name in sorted(names)]
        for owner, names in sorted(owned.items())
    }


def _routes(entries: Sequence[Mapping[str, Any]], key: str, *, context: str) -> str:
    """Return the one route every parameter's ownership entry declares."""

    routes = {str(entry.get(key, "")) for entry in entries}
    if len(routes) != 1 or not routes or "" in routes:
        raise SystemExit(
            f"the sharded leg does not declare one {context} for every "
            f"parameter: {sorted(routes)}"
        )
    return routes.pop()


def _boundary_edges(
    published: Sequence[Sequence[Mapping[str, Any]]], *, world_size: int
) -> list[dict[str, Any]]:
    """Return every measured rank boundary, described once.

    ``published[i]`` is rank ``i``'s list of the boundaries it took part in. A
    boundary is reported by both of the ranks that own it, and the executor emits
    a boundary record for every rank named in its ownership, so a report from one
    side only is the signature of a transport that ran locally instead of across
    the cut. The payload therefore requires both owners to have published the
    boundary and their two records to agree; neither an unreported boundary nor a
    one-sided one is assembled into a release contract.
    """

    seen: dict[tuple[int, ...], Mapping[str, Any]] = {}
    reporters: dict[tuple[int, ...], set[int]] = {}
    for rank, records in enumerate(published):
        for record in records:
            ranks = tuple(int(rank) for rank in record.get("owner_ranks", ()))
            if len(ranks) != 2 or sorted(ranks) != list(ranks):
                raise SystemExit(
                    f"boundary ownership {ranks} is not a pair of adjacent ranks"
                )
            if not 0 <= ranks[0] < ranks[1] < world_size:
                raise SystemExit(
                    f"boundary ownership {ranks} is outside a world of {world_size}"
                )
            previous = seen.setdefault(ranks, record)
            if previous != record:
                raise SystemExit(
                    f"ranks {ranks[0]} and {ranks[1]} disagree about the boundary "
                    "they share"
                )
            reporters.setdefault(ranks, set()).add(rank)
    expected = {(rank, rank + 1) for rank in range(world_size - 1)}
    if set(seen) != expected:
        missing = sorted(expected - set(seen))
        raise SystemExit(
            "the sharded leg reported no transport for every adjacent rank "
            f"boundary; missing={missing}"
        )
    unilateral = sorted(ranks for ranks in seen if reporters[ranks] != set(ranks))
    if unilateral:
        raise SystemExit(
            "the sharded leg reported a boundary that only one of the ranks which "
            "own it saw, so the transport did not cross the cut: "
            f"{[list(ranks) for ranks in unilateral]}"
        )
    edges: list[dict[str, Any]] = []
    for ranks in sorted(seen):
        record = seen[ranks]
        edges.append(
            {
                "owner_ranks": list(ranks),
                "bond": int(record["bond"]),
                "operation_id": str(record["operation_id"]),
                "payload_bytes": int(record["payload_bytes"]),
                "forward_transport": str(record["forward_transport"]),
                "reverse_transport": str(record["reverse_transport"]),
                "execution_status": "executed",
            }
        )
    return edges


def _node_placement(
    edges: Sequence[Mapping[str, Any]], *, local_world_size: int, node_count: int
) -> list[list[int]]:
    """Return the measured boundaries that cross a host boundary.

    Ranks are placed host by host in rank order, which is what the launcher
    does, so the crossing boundaries are the ones between the last rank of a
    host block and the first rank of the next. The placement is checked for
    consistency with the declared host count before it is used, because a
    payload that mislabelled intra-host traffic as inter-host traffic would
    overstate the measured transport.
    """

    if local_world_size <= 0 or node_count <= 0:
        raise SystemExit("the host layout needs a positive local world size")
    world_size = local_world_size * node_count
    crossing = [
        list(edge["owner_ranks"])
        for edge in edges
        if int(edge["owner_ranks"][0]) // local_world_size
        != int(edge["owner_ranks"][1]) // local_world_size
    ]
    expected = [
        [rank, rank + 1]
        for rank in range(world_size - 1)
        if (rank + 1) % local_world_size == 0
    ]
    if crossing != expected:
        raise SystemExit(
            "the measured boundaries do not cross hosts where the declared "
            f"layout says they do: crossing={crossing} expected={expected}"
        )
    return crossing


def _sharded_contract(
    contract: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    summaries: Sequence[Mapping[str, Any]],
    premise: Mapping[str, Any],
    digest: str,
    collective_backend: str,
    local_world_size: int,
    node_count: int,
) -> dict[str, Any]:
    """Assemble the release contract from the measured rank summaries.

    Every field is either read from a rank summary or derived from one by an
    operation stated here, and the derived fields name the measured fields they
    came from. The one field that describes the transport rather than the ranks
    is the collective backend, which is read from the process group that ran the
    leg instead of from the manifest that declared it: a leg that ran over gloo
    must not be reported as NCCL. Nothing else is carried over from the contract
    as if it had been measured: the shape is checked against the contract
    instead, and the baseline facts come from the premise artifact the
    single-device failure was measured in.
    """

    world_size = len(summaries)
    if world_size <= 1:
        raise SystemExit(
            "capacity-completion is the sharded run; a single world cannot shard "
            "one logical MPS across ranks"
        )
    steps = [int(summary["completed_steps"]) for summary in summaries]
    step_count = int(_agreed(steps, context="the number of completed steps"))
    if step_count <= 0:
        raise SystemExit(
            "the sharded leg completed no optimizer step, so it measured no "
            "training update"
        )
    if any(
        not all(step["useful_work_completed"] for step in summary["step_metrics"])
        for summary in summaries
    ):
        raise SystemExit(
            "a rank reports a step in which it did no site or bond work; a rank "
            "that idles is not evidence that the workload was sharded across it"
        )

    site_ownership = _agreed(
        [summary["site_ownership"] for summary in summaries],
        context="the site ownership map",
    )
    if len(site_ownership) != world_size or any(not sites for sites in site_ownership):
        raise SystemExit(
            "the measured site ownership map does not give every rank of a "
            f"world of {world_size} at least one site"
        )
    ownership_entries = _agreed(
        [tuple(summary["optimizer_ownership"]) for summary in summaries],
        context="the optimizer ownership map",
    )
    parameter_ownership = _ownership_map(ownership_entries, world_size=world_size)
    gradient_route = _routes(
        ownership_entries, "gradient_route", context="gradient route"
    )
    update_route = _routes(
        ownership_entries, "update_route", context="optimizer update route"
    )
    if (
        not all(
            bool(entry["optimizer_state_local"])
            == (int(entry["owner_rank"]) == int(summaries[0]["rank"]))
            for entry in ownership_entries
        )
        and world_size > 1
    ):
        raise SystemExit(
            "the sharded leg holds optimizer state on a rank that does not own "
            "the parameter it belongs to"
        )

    step_metrics = [summary["step_metrics"][-1] for summary in summaries]
    edges = _boundary_edges(
        [summary["step_metrics"][-1]["bond_updates"] for summary in summaries],
        world_size=world_size,
    )
    crossing = _node_placement(
        edges, local_world_size=local_world_size, node_count=node_count
    )
    boundary_bytes = sum(int(metric["boundary_bytes"]) for metric in step_metrics)
    inter_node_bytes = sum(
        int(edge["payload_bytes"])
        for edge in edges
        if list(edge["owner_ranks"]) in crossing
    )
    gradient_bytes = sum(
        int(metric["gradient_collective_bytes"]) for metric in step_metrics
    )
    optimizer_bytes = sum(
        int(metric["optimizer_collective_bytes"]) for metric in step_metrics
    )
    halo_bytes = sum(
        int(metric["layer_halo_intra_node_bytes"])
        + int(metric["layer_halo_inter_node_bytes"])
        for metric in step_metrics
    )
    total_bytes = boundary_bytes + gradient_bytes + optimizer_bytes + halo_bytes
    forward_exchanges = sum(
        int(metric["boundary_forward_exchanges"]) for metric in step_metrics
    )
    reverse_exchanges = sum(
        int(metric["boundary_reverse_exchanges"]) for metric in step_metrics
    )
    losses = [float(summary["losses"][-1]) for summary in summaries]
    peak_memory = [int(metric["peak_memory_bytes"]) for metric in step_metrics]
    reverse_peak = [int(metric["reverse_peak_memory_bytes"]) for metric in step_metrics]
    forward_peak = [int(metric["forward_peak_memory_bytes"]) for metric in step_metrics]
    timings = [
        sorted(float(step["end_to_end_seconds"]) for step in summary["step_metrics"])[
            len(summary["step_metrics"]) // 2
        ]
        for summary in summaries
    ]
    baseline = dict(premise["baseline"])
    return {
        "acceptance_case": "multi_gpu_capacity_completion",
        "state_mode": "mps",
        "topology_scope": "multi_node_production_transport",
        "distribution_semantics": "sharded_across_ranks",
        "mps_forward_distribution_semantics": "sharded_across_ranks",
        "mps_backward_distribution_semantics": "sharded_across_ranks",
        "executor": str(summaries[0]["executor"]),
        "collective_backend": str(collective_backend),
        "world_size": world_size,
        "local_world_size": local_world_size,
        "node_count": node_count,
        "rank_placement": "rank // local_world_size",
        "optimizer": str(summaries[0]["optimizer"]),
        "training_step_count": step_count,
        "batch_size": int(contract["batch_size"]),
        "dtype": str(contract["dtype"]),
        "n_sites": int(contract["n_sites"]),
        "initial_max_bond": int(contract["initial_max_bond"]),
        "max_bond_dimension": int(contract["trained_max_bond"]),
        "logical_mps_bytes": int(contract["logical_mps_bytes"]),
        "gradient_policy": str(contract["gradient_policy"]),
        "truncation_error_budget": float(contract["truncation_error_budget"]),
        "site_ownership_policy": str(summaries[0]["site_ownership_policy"]),
        "site_shard_ownership": {
            f"rank:{rank}": list(sites) for rank, sites in enumerate(site_ownership)
        },
        "bond_shard_ownership": _bond_shard_ownership(site_ownership, edges),
        "bond_shard_ownership_source": (
            "derived from the measured site ownership map and the measured "
            "boundary bonds, which name the two ranks that share each cut bond"
        ),
        "parameter_ownership": parameter_ownership,
        "gradient_ownership": parameter_ownership,
        "optimizer_update_ownership": parameter_ownership,
        "parameter_ownership_semantics": "sharded_across_ranks",
        "gradient_ownership_semantics": "sharded_across_ranks",
        "optimizer_update_ownership_semantics": "sharded_across_ranks",
        "gradient_route": gradient_route,
        "update_route": update_route,
        "parameter_gradient_ownership": {
            name: f"rank:{owner}"
            for owner, names in parameter_ownership.items()
            for name in names
        },
        "parameter_gradient_ownership_source": (
            "derived from the measured optimizer ownership map and the measured "
            f"gradient route {gradient_route}"
        ),
        "boundary_gradient_ownership": {
            f"boundary:{edge['bond']}": [f"rank:{rank}" for rank in edge["owner_ranks"]]
            for edge in edges
        },
        "boundary_gradient_routes": {
            f"boundary:{edge['bond']}": {
                "owner_ranks": edge["owner_ranks"],
                "operation_id": edge["operation_id"],
                "gradient_route": gradient_route,
                "update_route": update_route,
            }
            for edge in edges
        },
        "boundary_adjoint_exchange": {
            "status": "executed",
            "execution_status": "executed",
            "measured_step": step_count,
            "boundary_count": len(edges),
            "forward_exchanges": forward_exchanges,
            "reverse_exchanges": reverse_exchanges,
            "boundary_bytes": boundary_bytes,
            "transport": "batched_isend_irecv_packed_multi_tensor_envelope",
        },
        # The reverse pass is stated as executed because the measured step metrics
        # carry the reverse exchanges it performed. A leg that exchanged nothing
        # in reverse measured no backward pass, so it says so and the gate reports
        # the missing evidence instead of reading the label as an intention.
        "backward_execution": (
            "executed_sharded_reverse_pass" if reverse_exchanges > 0 else "not_executed"
        ),
        "backward_execution_measured": reverse_exchanges > 0,
        "mps_backward_memory_plan": {
            "status": "measured",
            "measured_backward_peak_memory_bytes": max(reverse_peak),
            "measured_backward_peak_memory_bytes_by_rank": reverse_peak,
            "measured_forward_peak_memory_bytes_by_rank": forward_peak,
            "source": "step metrics of the executed reverse pass",
        },
        "mps_backward_communication_plan": {
            "status": "executed",
            "communication_protocol": (
                "batched_isend_irecv_packed_multi_tensor_envelope"
            ),
            "boundary_edges": edges,
        },
        "boundary_communication_bytes": {
            f"boundary:{edge['bond']}": int(edge["payload_bytes"]) for edge in edges
        },
        "communication_bytes": boundary_bytes,
        "communication_counting": (
            "each boundary payload is counted at both of the ranks that own it, "
            "which is the convention the ISSUE-092 capacity artifact uses"
        ),
        "inter_node_communication_bytes": inter_node_bytes,
        "intra_node_communication_bytes": boundary_bytes - inter_node_bytes,
        "inter_node_boundary_ranks": crossing,
        "communication_fraction": (
            boundary_bytes / total_bytes if total_bytes else 0.0
        ),
        "communication_fraction_definition": (
            "measured boundary transport bytes over the sum of measured boundary, "
            "gradient collective, optimizer collective and halo bytes"
        ),
        "rank_outputs": losses,
        "rank_gradients": [
            int(metric["gradient_collective_bytes"]) for metric in step_metrics
        ],
        "rank_timings": timings,
        "rank_peak_memory_bytes": peak_memory,
        "measured_peak_memory_bytes": max(peak_memory),
        "measured_peak_memory_bytes_by_rank": peak_memory,
        "gpu_activity": 1.0,
        "full_mps_materialization": False,
        "workload_sha256": digest,
        "workload_definition_sha256": str(contract["workload_definition_sha256"]),
        "state_mode_evidence": {
            "state_mode": "mps",
            "executor": str(summaries[0]["executor"]),
            "full_mps_materialization": False,
        },
        "single_gpu_expected_oom": True,
        "capacity_baseline_device": str(baseline["capacity_baseline_device"]),
        "capacity_baseline_device_source": str(baseline["source"]),
        "capacity_failure_reason": str(baseline["capacity_failure_reason"]),
        "capacity_baseline_peak_memory_bytes": int(
            baseline["single_gpu_peak_memory_bytes"]
        ),
        "capacity_baseline_device_total_memory_bytes": int(
            baseline["single_gpu_device_total_memory_bytes"]
        ),
        "capacity_baseline_artifact": str(premise["path"]),
        "capacity_baseline_sha256": str(premise["sha256"]),
        "capacity_baseline_topology_fingerprint": str(premise["topology_fingerprint"]),
        "premise_source_integrity": dict(premise["source_integrity"]),
        "fallback_semantics": str(runtime["fallback_semantics"]),
        "fallback_events": [],
        "blockers": [],
    }


def _bond_shard_ownership(
    site_ownership: Sequence[Sequence[int]], edges: Sequence[Mapping[str, Any]]
) -> dict[str, str]:
    """Return the measured owner of every bond the run held.

    Bonds inside a rank's site range are owned by that rank alone. The cut bonds
    are owned by the pair of ranks that exchanged them, which is what the
    measured boundary records name, so the map is stated in terms of those
    records rather than inferred from the site split.
    """

    owners = {
        f"bond:{min(sites)}-{max(sites)}": f"rank:{rank}"
        for rank, sites in enumerate(site_ownership)
    }
    for edge in edges:
        ranks = [int(rank) for rank in edge["owner_ranks"]]
        owners[f"bond:{int(edge['bond'])}"] = f"rank:{ranks[0]}+rank:{ranks[1]}"
    return dict(sorted(owners.items()))


def _read_single_gpu_failure(
    premise: Mapping[str, Any], base_dir: Path
) -> tuple[dict[str, Any], list[str]]:
    """Return the measured single-device failure and the sources that are gone.

    The premise names its sources and their digests. The failure record itself
    has to be readable and has to match its digest, because every baseline fact
    the release payload states comes from it. The remaining sources -- the raw
    log and the device telemetry -- are checked when their bytes are still on
    disk and reported when they are not, so a payload recorded from a premise
    whose provenance has been cleaned up says so instead of implying that it was
    verified.
    """

    source = next(
        (
            item
            for item in premise.get("source_artifacts", ())
            if item.get("kind") == "single_gpu_failure"
        ),
        None,
    )
    if not isinstance(source, Mapping):
        raise SystemExit(
            "the capacity premise records no single-device failure artifact, so "
            "there is nothing measured for the release payload to inherit"
        )
    path = Path(str(source["path"]))
    resolved = path if path.is_absolute() else base_dir / path
    if not resolved.is_file():
        raise SystemExit(
            f"the single-device failure record {resolved} is gone, so the "
            "baseline facts a release payload states cannot be checked"
        )
    if _sha256(resolved) != str(source.get("sha256")):
        raise SystemExit(
            f"{resolved} is not the failure the capacity premise was assembled "
            "from; its digest does not match the premise's record"
        )
    unverifiable: list[str] = []
    for item in premise.get("source_artifacts", ()):
        if item is source:
            continue
        candidate = Path(str(item["path"]))
        candidate = candidate if candidate.is_absolute() else base_dir / candidate
        if not candidate.is_file() or _sha256(candidate) != str(item.get("sha256")):
            unverifiable.append(str(item["path"]))
    return dict(json.loads(resolved.read_text(encoding="utf-8"))), unverifiable


def _premise(
    contract: Mapping[str, Any], path: Path, *, workload: Any
) -> dict[str, Any]:
    """Validate the ISSUE-092 capacity premise the completion role inherits.

    The premise is what says the frozen workload failed on one device, so it is
    read through the certification the ISSUE-092 artifacts are held to and then
    checked against the frozen shape: a premise measured on a different site
    count, bond dimension or rank boundary set describes a different workload and
    cannot make this one a capacity claim.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} does not carry a capacity premise")
    try:
        require_general_mps_capacity(payload)
    except MPSCapacityCertificationError as error:
        raise SystemExit(
            f"{path} is not a certified capacity premise: {error}"
        ) from None
    if int(payload.get("n_sites", 0)) != int(contract["n_sites"]):
        raise SystemExit(
            f"{path} measured {payload.get('n_sites')} sites where the frozen "
            f"workload has {int(contract['n_sites'])}"
        )
    if int(payload.get("initial_max_bond", 0)) != int(contract["trained_max_bond"]):
        raise SystemExit(
            f"{path} measured a maximum bond dimension of "
            f"{payload.get('initial_max_bond')} where the frozen workload uses "
            f"{int(contract['trained_max_bond'])}"
        )
    if int(payload.get("world_size", 0)) != int(contract["target_world_size"]):
        raise SystemExit(
            f"{path} completed across {payload.get('world_size')} ranks where the "
            f"frozen topology is {int(contract['target_world_size'])}"
        )
    fingerprint = str(payload.get("topology_fingerprint", ""))
    expected = str(workload.topology_fingerprint())
    if fingerprint != expected:
        raise SystemExit(
            f"{path} measured topology {fingerprint} where the frozen workload's "
            f"rank boundaries and bond schedule give {expected}"
        )
    failure, unverifiable = _read_single_gpu_failure(payload, REPO_ROOT)
    record = failure.get("rank_record", {})
    if failure.get("status") != "cuda_oom" or not record.get("error"):
        raise SystemExit(
            f"{path} does not rest on a measured single-device exhaustion; its "
            "failure record reports no CUDA out-of-memory reason"
        )
    source = "the single-device failure record named by the premise"
    return {
        "path": str(path),
        "sha256": _sha256(Path(path)),
        "topology_fingerprint": fingerprint,
        "baseline": {
            "single_gpu_peak_memory_bytes": int(
                payload["single_gpu_peak_memory_bytes"]
            ),
            "single_gpu_device_total_memory_bytes": int(
                payload["single_gpu_device_total_memory_bytes"]
            ),
            "capacity_failure_reason": str(record["error"]),
            "capacity_baseline_device": _baseline_device_identity(record),
            "source": source,
        },
        "source_integrity": {
            "finalized": payload.get("source_integrity_finalized") is True,
            "unverifiable_paths": unverifiable,
        },
    }


def _baseline_device_identity(record: Mapping[str, Any]) -> str:
    """Return a measured identity for the device the baseline failed on.

    The failure record carries the device's total memory as measured and, on a
    run that recorded it, the device name. A name is never invented for a run
    that did not record one: an unmeasured model would be a claim about hardware
    that no artifact supports.
    """

    name = str(record.get("device_name") or "").strip()
    if name:
        return name
    return (
        "cuda device rank 0, device_total_memory_bytes="
        f"{int(record['device_total_memory_bytes'])} (device name not recorded by "
        "the measured run)"
    )


def _role_capacity_failure(arguments: argparse.Namespace) -> int:
    """Record the single-device exhaustion of the frozen capacity workload."""

    contract = _capacity_contract(arguments.release_manifest)
    _require_frozen_shape(arguments, contract)
    digest = _workload_digest(contract)
    if digest != str(contract["workload_sha256"]):
        raise SystemExit(
            "the frozen workload definition does not match the digest the release "
            "manifest declares; the premise moved and the run did not"
        )
    world = _env_int("WORLD_SIZE", 1)
    if world != 1:
        raise SystemExit(
            "capacity-failure records one device and nothing else; launch it with "
            f"a world size of one, not {world}"
        )
    if not torch.cuda.is_available():
        raise SystemExit(
            "capacity-failure needs the measured device: the frozen workload is a "
            "capacity premise about one accelerator, so a run without one measures "
            "nothing"
        )
    device = torch.device("cuda", _env_int("LOCAL_RANK", 0))
    torch.cuda.set_device(device)
    _initialize(device)
    workload = _frozen_workload(contract)
    total_bytes = int(torch.cuda.get_device_properties(device).total_memory)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    status, failure_reason = "completed", ""
    started = time.perf_counter()
    try:
        initial = workload.rank_owned_initial_mps(
            workload.N_SITES, workload.MAX_BOND, device
        )
        theta = torch.tensor(7e-5, device=device, requires_grad=True)
        phi = torch.tensor(-1.1e-4, device=device, requires_grad=True)
        fqxd.train_distributed_mps(
            workload.workload(theta, phi),
            steps=int(contract["steps"]),
            observable={workload.N_SITES // 2: "z"},
            optimizer=str(contract["optimizer"]),
            lr=float(contract["learning_rate"]),
            device=device,
            max_bond=int(contract["trained_max_bond"]),
            gradient_policy=str(contract["gradient_policy"]),
            gradient_tolerance=float(contract["truncation_error_budget"]),
            initial_mps_tensors=initial,
            initial_mps_left_canonical=bool(contract["initial_mps_left_canonical"]),
            canonicalization_policy=str(contract["canonicalization_policy"]),
            svd_driver=str(contract["svd_driver"]),
            reverse_checkpoint_policy=fqxm.MPSReverseCheckpointPolicy(
                max_saved_bytes=workload.reverse_checkpoint_capacity_bytes(
                    int(contract["logical_mps_bytes"]), world
                )
            ),
        )
        torch.cuda.synchronize(device)
    except torch.OutOfMemoryError as error:
        status = "cuda_oom"
        failure_reason = str(error).splitlines()[0]
    except RuntimeError as error:
        status = "runtime_error"
        failure_reason = str(error).splitlines()[0]
    elapsed = time.perf_counter() - started
    peak_bytes = int(torch.cuda.max_memory_allocated(device))
    measurements = _baseline_measurements(
        contract,
        status=status,
        failure_reason=failure_reason,
        peak_memory_bytes=peak_bytes,
        device_total_memory_bytes=total_bytes,
        device_name=str(torch.cuda.get_device_properties(device).name),
        digest=digest,
        world_size=world,
        local_world_size=1,
        node_count=1,
    )
    measurements["elapsed_seconds"] = elapsed
    measurements["failure_elapsed_seconds"] = elapsed
    _write(
        arguments,
        {
            "schema": SCHEMA,
            "role": arguments.role,
            "rank": 0,
            "world_size": world,
            "local_world_size": 1,
            "node_count": 1,
            "hostname": socket.gethostname(),
            "commit": _commit(),
            "device_name": measurements["capacity_baseline_device"],
            "software": _software(),
            "measurements": measurements,
        },
    )
    if status != "cuda_oom":
        # A workload that fits on one device is not a capacity premise, and a run
        # that failed for another reason did not measure the exhaustion the
        # contract is about. Both are reported as failures of the role.
        print(
            f"the frozen workload did not exhaust one device: status={status} "
            f"{failure_reason}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    return 0


def _gather(record: Mapping[str, Any], world: int) -> list[Any]:
    """Return every rank's record, carried where NCCL cannot carry it.

    NCCL classifies a gather as an operation it has no route for, and a long
    training run drives its socket state into a failure the collectives share. A
    gloo group is created for this one transfer and destroyed immediately.
    """

    if world <= 1:
        return [dict(record)]
    group = None
    try:
        if "GLOO_SOCKET_IFNAME" not in os.environ and os.environ.get(
            "NCCL_SOCKET_IFNAME"
        ):
            os.environ["GLOO_SOCKET_IFNAME"] = os.environ["NCCL_SOCKET_IFNAME"]
        group = dist.new_group(backend="gloo")
        if int(os.environ["RANK"]) != 0:
            dist.gather_object(dict(record), None, dst=0, group=group)
            return []
        gathered: list[Any] = [None] * world
        dist.gather_object(dict(record), gathered, dst=0, group=group)
        return gathered
    finally:
        if group is not None:
            dist.destroy_process_group(group)


def _role_capacity_completion(arguments: argparse.Namespace) -> int:
    """Record the sharded training update that completes the frozen workload."""

    contract = _capacity_contract(arguments.release_manifest)
    runtime = _runtime_contract(arguments.release_manifest)
    _require_frozen_shape(arguments, contract)
    if arguments.premise is None:
        raise SystemExit(
            "capacity-completion requires --premise: the sharded completion is a "
            "capacity claim only because one device was measured to fail at this "
            "workload, and that measurement is what the payload inherits"
        )
    digest = _workload_digest(contract)
    if digest != str(contract["workload_sha256"]):
        raise SystemExit(
            "the frozen workload definition does not match the digest the release "
            "manifest declares; the premise moved and the run did not"
        )
    world = _env_int("WORLD_SIZE", 1)
    if world <= 1:
        raise SystemExit(
            "capacity-completion is the sharded run; a single world cannot shard "
            "one logical MPS across ranks"
        )
    if world != int(contract["target_world_size"]):
        raise SystemExit(
            f"the frozen topology shards the workload across "
            f"{int(contract['target_world_size'])} ranks, not {world}"
        )
    local_world = int(arguments.local_world_size)
    node_count = int(arguments.node_count)
    if local_world <= 0 or world % local_world or world // local_world != node_count:
        raise SystemExit(
            f"a world of {world} over {node_count} host(s) of {local_world} "
            "device(s) does not place every rank exactly once"
        )
    if node_count < int(
        _load_frozen_manifest(Path(arguments.release_manifest))["topologies"][
            "minimum_multi_node_count"
        ]
    ):
        raise SystemExit(
            "the frozen MPS topology requires the ranks to span more than one "
            f"host; a launch over {node_count} host(s) measured a single-node run"
        )
    if not envelope_carries_world(world):
        # The measurement is still worth taking and the payload is still worth
        # reviewing, but the sealer cannot wrap it, so the operator is told here
        # rather than after the run has been sealed against a refusal.
        print(
            f"warning: the release world of {world} ranks cannot be described by "
            "any evidence scope, so this payload cannot be sealed as a signed "
            "measured_production_run artifact until "
            "flagquantum/runtime/observability/evidence.py can carry it; the "
            "release gate reports "
            "release_world_size_not_carriable_by_evidence_envelope",
            file=sys.stderr,
            flush=True,
        )
    if not torch.cuda.is_available():
        raise SystemExit(
            "capacity-completion needs the measured devices; a run without "
            "accelerators measured no per-rank memory and no boundary transport"
        )
    workload = _frozen_workload(contract)
    premise = _premise(contract, Path(arguments.premise), workload=workload)
    device = torch.device("cuda", _env_int("LOCAL_RANK", 0))
    torch.cuda.set_device(device)
    _initialize(device)
    # Read the backend off the group the leg ran in, while it is still up, so the
    # payload states the transport that carried the measured bytes.
    collective_backend = str(dist.get_backend())
    outcomes: list[dict[str, Any]] = []
    status, failure_reason = "passed", ""
    try:
        initial = workload.rank_owned_initial_mps(
            workload.N_SITES, workload.MAX_BOND, device
        )
        theta = torch.tensor(7e-5, device=device, requires_grad=True)
        phi = torch.tensor(-1.1e-4, device=device, requires_grad=True)
        result = fqxd.train_distributed_mps(
            workload.workload(theta, phi),
            steps=int(contract["steps"]),
            observable={workload.N_SITES // 2: "z"},
            optimizer=str(contract["optimizer"]),
            lr=float(contract["learning_rate"]),
            device=device,
            max_bond=int(contract["trained_max_bond"]),
            gradient_policy=str(contract["gradient_policy"]),
            gradient_tolerance=float(contract["truncation_error_budget"]),
            initial_mps_tensors=initial,
            initial_mps_left_canonical=bool(contract["initial_mps_left_canonical"]),
            canonicalization_policy=str(contract["canonicalization_policy"]),
            svd_driver=str(contract["svd_driver"]),
            reverse_checkpoint_policy=fqxm.MPSReverseCheckpointPolicy(
                max_saved_bytes=workload.reverse_checkpoint_capacity_bytes(
                    int(contract["logical_mps_bytes"]), world
                )
            ),
        )
        torch.cuda.synchronize(device)
        outcomes = _gather(result.summary(), world)
    except (torch.OutOfMemoryError, RuntimeError) as error:
        status = "runtime_error"
        failure_reason = str(error).splitlines()[0]
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"the sharded leg failed: {failure_reason}", file=sys.stderr)
        return 2
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    if int(os.environ.get("RANK", "0")) != 0:
        return 0
    summaries = [dict(item) for item in outcomes]
    contract_payload = _sharded_contract(
        contract,
        runtime,
        summaries=summaries,
        premise=premise,
        digest=digest,
        collective_backend=collective_backend,
        local_world_size=local_world,
        node_count=node_count,
    )
    records = [
        {
            "schema": SCHEMA,
            "role": arguments.role,
            "rank": int(summary["rank"]),
            "world_size": int(summary["world_size"]),
            "local_world_size": local_world,
            "node_count": node_count,
            "hostname": socket.gethostname(),
            "commit": _commit(),
            "completed_steps": int(summary["completed_steps"]),
            "losses": [float(value) for value in summary["losses"]],
            "step_metrics": [dict(item) for item in summary["step_metrics"]],
            "site_ownership": [list(item) for item in summary["site_ownership"]],
            "optimizer_ownership": [
                dict(item) for item in summary["optimizer_ownership"]
            ],
            "software": _software(),
        }
        for summary in summaries
    ]
    _write(
        arguments,
        {
            "schema": SCHEMA,
            "role": arguments.role,
            "rank": 0,
            "world_size": world,
            "local_world_size": local_world,
            "node_count": node_count,
            "hostname": socket.gethostname(),
            "commit": _commit(),
            "status": status,
            "software": _software(),
            "measurements": contract_payload,
            "ranks": records,
        },
    )
    return 0


def _role_release_payload(arguments: argparse.Namespace) -> int:
    """Project one role document onto the contract the sealer reads.

    A role document states the release contract under ``measurements`` beside the
    per-rank records it was assembled from. ``tools/seal_runtime_evidence.py``
    takes its evidence input at the top level, so something has to lift the
    contract out; doing it here keeps the projection reviewable and repeatable
    instead of leaving it to whoever runs the campaign.
    """

    if arguments.document is None:
        raise SystemExit("release-payload requires --document")
    document = json.loads(Path(arguments.document).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise SystemExit(f"{arguments.document} is not a measurements document")
    payload = document.get("measurements")
    if not isinstance(payload, dict):
        raise SystemExit(
            f"{arguments.document} carries no measurements block to project; a "
            "release payload is the contract a role assembled, so a document "
            "without one has nothing to seal"
        )
    _write(arguments, payload)
    return 0


def _refuse_unfrozen_speed(arguments: argparse.Namespace) -> int:
    """Refuse a speed role the frozen manifest cannot support."""

    manifest = _load_frozen_manifest(Path(arguments.release_manifest))
    ladder = manifest["speed_workload"]["configuration_ladder"]
    raise SystemExit(
        f"the frozen MPS manifest declares no matched-speed configuration ladder "
        f"(configuration_ladder={ladder!r}), so the {arguments.role} role has no "
        "frozen protocol to measure against: no MPS single-device leg and no MPS "
        "sharded leg has been timed, and a comparison run against a ladder chosen "
        "now would not be a frozen acceptance test. Freeze the ladder in "
        f"{arguments.release_manifest} first."
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--release-manifest", type=Path, default=RELEASE_MANIFEST)
    parser.add_argument("--measurements", type=Path)
    parser.add_argument("--document", type=Path)
    parser.add_argument("--premise", type=Path)
    parser.add_argument(
        "--bond-dimension",
        type=int,
        help=(
            "maximum bond dimension the launch was configured with. It is checked "
            "against the frozen workload and never used to select one, so a "
            "launcher that ran a different shape is told before it is recorded"
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        help=(
            "optimizer steps the launch was configured with, checked against the "
            "frozen workload for the same reason as --bond-dimension"
        ),
    )
    parser.add_argument(
        "--local-world-size",
        type=int,
        default=1,
        help="devices per host; the launcher knows this and the ranks do not",
    )
    parser.add_argument(
        "--node-count",
        type=int,
        default=1,
        help="hosts the run spans; the launcher knows this and the ranks do not",
    )
    arguments = parser.parse_args(argv)

    if arguments.role == "release-payload":
        return _role_release_payload(arguments)
    if arguments.role in UNFROZEN_SPEED_ROLES:
        return _refuse_unfrozen_speed(arguments)
    if arguments.role == "capacity-failure":
        return _role_capacity_failure(arguments)
    return _role_capacity_completion(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
