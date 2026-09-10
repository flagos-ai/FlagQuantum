"""Fail-closed contracts for matched FlagOS statevector capacity evidence."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

CAPACITY_PROFILE_SCHEMA = "flagquantum_flagos_statevector_capacity_f5_v1"
CAPACITY_RUN_SCHEMA = "flagquantum_flagos_statevector_capacity_run_v1"
CAPACITY_MODES = ("single", "replicated", "sharded")


class FlagOSCapacityProfileError(ValueError):
    """Raised when capacity evidence is malformed, mismatched, or overclaimed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FlagOSCapacityProfileError(message)


def _validate_boundary(run: Mapping[str, Any], mode: str) -> None:
    _require(run.get("schema") == CAPACITY_RUN_SCHEMA, f"{mode}: invalid schema")
    _require(run.get("mode") == mode, f"{mode}: mode mismatch")
    _require(run.get("logical_device_type") == "flagos", f"{mode}: not FlagOS")
    _require(run.get("flagcx_route_verified") is False, f"{mode}: route inferred")
    _require(run.get("host_staging_observed") is None, f"{mode}: host staging claim")
    for field in (
        "communication_claim_allowed",
        "scalability_claim_allowed",
        "production_support_claim_allowed",
        "release_gate_allowed",
    ):
        _require(run.get(field) is False, f"{mode}: {field} must remain false")


def _validate_topology(
    run: Mapping[str, Any], mode: str
) -> tuple[Mapping[str, Any], ...]:
    world_size = run.get("world_size")
    ranks = run.get("ranks")
    if not isinstance(world_size, int) or world_size < 1:
        raise FlagOSCapacityProfileError(f"{mode}: invalid world size")
    if not isinstance(ranks, list) or len(ranks) != world_size:
        raise FlagOSCapacityProfileError(f"{mode}: incomplete ranks")
    for item in ranks:
        if not isinstance(item, Mapping):
            raise FlagOSCapacityProfileError(f"{mode}: rank record must be an object")
    _require(
        {item.get("rank") for item in ranks} == set(range(world_size)),
        f"{mode}: missing or duplicate ranks",
    )
    _require(
        all(
            isinstance(item.get("total_memory_bytes"), int)
            and item["total_memory_bytes"] > 0
            and isinstance(item.get("peak_memory_bytes"), int)
            and item["peak_memory_bytes"] >= 0
            for item in ranks
        ),
        f"{mode}: invalid memory evidence",
    )
    return tuple(ranks)


def _oom_accepted(run: Mapping[str, Any], mode: str, world_size: int) -> bool:
    ranks = _validate_topology(run, mode)
    _require(run.get("world_size") == world_size, f"{mode}: unexpected world size")
    expected_distribution = (
        "single_device_fast_path"
        if mode == "single"
        else "replicated_full_state_per_rank"
    )
    return bool(
        run.get("status") == "expected_oom"
        and run.get("expected_outcome") == "capacity_failure"
        and run.get("distribution_semantics") == expected_distribution
        and run.get("full_state_materialization") is True
        and all(
            item.get("status") == "expected_oom"
            and item.get("oom_observed") is True
            and isinstance(item.get("error"), str)
            and "out of memory" in item["error"].lower()
            for item in ranks
        )
    )


def _sharded_accepted(run: Mapping[str, Any], world_size: int) -> bool:
    ranks = _validate_topology(run, "sharded")
    _require(run.get("world_size") == world_size, "sharded: unexpected world size")
    total_amplitudes = 2 ** int(run["workload"]["n_wires"])
    return bool(
        run.get("status") == "passed"
        and run.get("expected_outcome") == "completion"
        and run.get("distribution_semantics") == "sharded_across_ranks"
        and run.get("full_state_materialization") is False
        and all(
            item.get("status") == "passed"
            and item.get("oom_observed") is False
            and item.get("device_type") == "flagos"
            and item.get("local_amplitudes", 0) > 0
            and item["local_amplitudes"] * world_size == total_amplitudes
            and item.get("communication_count", 0) > 0
            and item.get("communication_bytes", 0) > 0
            and item.get("norm_error", float("inf")) <= item.get("tolerance", 0)
            and item.get("validation_method")
            == "single_full_width_forward_with_fp64_global_norm"
            and item.get("peak_memory_bytes", 0) < item.get("total_memory_bytes", 0)
            for item in ranks
        )
    )


@dataclass(frozen=True)
class FlagOSStatevectorCapacityProfile:
    """A matched single/replicated/sharded development capacity result."""

    workload: Mapping[str, Any]
    workload_sha256: str
    completion_world_size: int
    runs: tuple[Mapping[str, Any], ...]
    source_revision: str
    torch_fl_source_revision: str

    @property
    def run_map(self) -> dict[str, Mapping[str, Any]]:
        return {str(run["mode"]): run for run in self.runs}

    @property
    def accepted(self) -> bool:
        runs = self.run_map
        return bool(
            set(runs) == set(CAPACITY_MODES)
            and _oom_accepted(runs["single"], "single", 1)
            and _oom_accepted(
                runs["replicated"], "replicated", self.completion_world_size
            )
            and _sharded_accepted(runs["sharded"], self.completion_world_size)
        )

    def to_dict(self) -> dict[str, Any]:
        accepted = self.accepted
        return {
            "schema": CAPACITY_PROFILE_SCHEMA,
            "status": "passed" if accepted else "failed",
            "validation_scope": "flagos_single_node_matched_statevector_capacity",
            "artifact_classification": "development_capacity_evidence",
            "workload": dict(self.workload),
            "workload_sha256": self.workload_sha256,
            "completion_world_size": self.completion_world_size,
            "runs": [dict(run) for run in self.runs],
            "development_capacity_expansion_observed": accepted,
            "single_device_capacity_failure_measured": _oom_accepted(
                self.run_map["single"], "single", 1
            ),
            "replicated_capacity_failure_measured": _oom_accepted(
                self.run_map["replicated"],
                "replicated",
                self.completion_world_size,
            ),
            "sharded_capacity_completion_measured": _sharded_accepted(
                self.run_map["sharded"], self.completion_world_size
            ),
            "outer_backend": "flagos",
            "inner_communication_route": "unattributed",
            "flagcx_route_verified": False,
            "host_staging_observed": None,
            "communication_claim_allowed": False,
            "scalability_claim_allowed": False,
            "production_support_claim_allowed": False,
            "release_gate_allowed": False,
            "environment": {
                "source_revision": self.source_revision,
                "torch_fl_source_revision": self.torch_fl_source_revision,
            },
            "blockers": [
                "development_threshold_calibrated_not_preregistered",
                "inner_communication_route_unattributed",
                "host_staging_unverified",
                "single_node_capacity_evidence_only",
                "performance_convergence_and_multinode_not_certified",
            ]
            + ([] if accepted else ["matched_capacity_expansion_not_demonstrated"]),
        }

    def require_accepted(self) -> None:
        """Reject a profile that does not prove the matched capacity result."""

        if not self.accepted:
            raise FlagOSCapacityProfileError("F5 matched capacity gate failed")


def build_flagos_statevector_capacity_profile(
    runs: Sequence[Mapping[str, Any]],
) -> FlagOSStatevectorCapacityProfile:
    """Validate and aggregate exactly one matched run for each F5 mode."""

    _require(len(runs) == len(CAPACITY_MODES), "F5 requires exactly three runs")
    modes = [run.get("mode") for run in runs]
    _require(
        set(modes) == set(CAPACITY_MODES) and len(set(modes)) == len(modes),
        "F5 modes are incomplete",
    )
    for run in runs:
        _validate_boundary(run, str(run.get("mode")))
    hashes = {run.get("workload_sha256") for run in runs}
    workloads = {json_key(run.get("workload")) for run in runs}
    _require(len(hashes) == 1 and None not in hashes, "F5 workload hashes differ")
    _require(len(workloads) == 1, "F5 workload payloads differ")
    expected_hash = hashlib.sha256(
        json_key(runs[0].get("workload")).encode()
    ).hexdigest()
    _require(hashes == {expected_hash}, "F5 workload hash is not canonical")
    revisions = {run.get("source_revision") for run in runs}
    torch_fl_revisions = {run.get("torch_fl_source_revision") for run in runs}
    _require(
        len(revisions) == 1 and None not in revisions, "F5 source revisions differ"
    )
    _require(
        len(torch_fl_revisions) == 1 and None not in torch_fl_revisions,
        "F5 Torch-FL revisions differ",
    )
    run_map = {str(run["mode"]): run for run in runs}
    completion_world_size = int(run_map["sharded"]["world_size"])
    _require(
        completion_world_size in {2, 4, 8},
        "F5 completion world size must be 2, 4, or 8",
    )
    _require(
        run_map["replicated"].get("world_size") == completion_world_size,
        "F5 replicated and sharded world sizes differ",
    )
    return FlagOSStatevectorCapacityProfile(
        workload=dict(run_map["single"]["workload"]),
        workload_sha256=str(hashes.pop()),
        completion_world_size=completion_world_size,
        runs=tuple(run_map[mode] for mode in CAPACITY_MODES),
        source_revision=str(revisions.pop()),
        torch_fl_source_revision=str(torch_fl_revisions.pop()),
    )


def json_key(value: Any) -> str:
    """Return a canonical comparison key without accepting non-JSON payloads."""

    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


__all__ = (
    "CAPACITY_MODES",
    "CAPACITY_PROFILE_SCHEMA",
    "CAPACITY_RUN_SCHEMA",
    "FlagOSCapacityProfileError",
    "FlagOSStatevectorCapacityProfile",
    "build_flagos_statevector_capacity_profile",
)
