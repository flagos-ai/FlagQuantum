#!/usr/bin/env python
"""Verify one CUDA statevector shard per node without making a scale claim."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

import flagquantum as fq
from flagquantum.runtime.backends.statevector.forward_executor import (
    execute_torch_distributed_statevector,
)

ROOT = Path(__file__).resolve().parents[1]


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _git_revision() -> str:
    declared = os.environ.get("FLAGQUANTUM_SOURCE_REVISION")
    if declared:
        return declared
    try:
        return subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _circuit() -> fq.Circuit:
    return (
        fq.Circuit(5, dtype=torch.complex128)
        .h(0)
        .ry(4, 0.31)
        .cx(4, 1)
        .rx(3, -0.27)
        .cx(0, 4)
        .rz(4, 0.19)
        .cx(3, 2)
    )


def _validation_state(result: Any) -> torch.Tensor:
    """Materialize a tiny state solely to compare with the CPU reference."""

    local = result.shard_state.amplitudes[0]
    gathered = [torch.empty_like(local) for _ in range(result.plan.world_size)]
    dist.all_gather(gathered, local)
    internal = torch.empty(
        result.plan.total_amplitudes,
        dtype=local.dtype,
        device=local.device,
    )
    for rank, shard in enumerate(gathered):
        internal[rank :: result.plan.world_size] = shard
    mapping = result.logical_to_physical_wires
    canonical = torch.empty_like(internal)
    for logical_basis in range(internal.numel()):
        physical_basis = 0
        for logical_wire, physical_wire in enumerate(mapping):
            bit = (logical_basis >> (len(mapping) - logical_wire - 1)) & 1
            physical_basis |= bit << (len(mapping) - physical_wire - 1)
        canonical[logical_basis] = internal[physical_basis]
    return canonical


def _network_observation(path: Path | None) -> dict[str, Any]:
    configured_interface = os.environ.get("NCCL_SOCKET_IFNAME")
    ib_disabled = os.environ.get("NCCL_IB_DISABLE") == "1"
    observation: dict[str, Any] = {
        "backend": "nccl",
        "configured_interface": configured_interface,
        "ib_disabled": ib_disabled,
        "route": "socket" if ib_disabled else "nccl_auto",
        "evidence_level": "configured_only",
    }
    if path is None:
        return observation
    content = path.read_text(encoding="utf-8", errors="replace")
    socket_observed = "NET/Socket" in content
    interface_observed = bool(configured_interface and configured_interface in content)
    observation.update(
        {
            "debug_log_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "socket_transport_observed": socket_observed,
            "configured_interface_observed": interface_observed,
            "evidence_level": "observed_debug_log",
        }
    )
    if ib_disabled and not (socket_observed and interface_observed):
        raise RuntimeError(
            "NCCL debug log does not confirm the configured socket route"
        )
    return observation


def _artifact(
    *,
    rank_records: list[dict[str, Any]],
    metrics: dict[str, float],
    network: dict[str, Any],
) -> dict[str, Any]:
    evidence = {
        "schema": "flagquantum.cuda_multinode_statevector_probe.v1",
        "status": "passed",
        "environment": {
            "source_revision": _git_revision(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda_runtime": str(torch.version.cuda),
            "nccl": ".".join(str(item) for item in torch.cuda.nccl.version()),
        },
        "scope": {
            "dtype": "complex128",
            "n_wires": 5,
            "world_size": 2,
            "local_world_size": 1,
            "node_count": 2,
            "distribution_semantics": "sharded_across_ranks",
            "execution": "forward_statevector",
        },
        "observations": {
            "numerical_metrics": metrics,
            "rank_records": rank_records,
            "network": network,
            "validation_full_state_materialization": True,
            "production_full_state_materialization": False,
        },
        "claim_blockers": [
            "validation_only_tiny_full_state_gather",
            "hidden_host_staging_not_audited",
            "rdma_not_tested",
            "distributed_gradient_not_tested",
            "checkpoint_restart_not_tested",
            "production_performance_not_measured",
        ],
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    digest = _canonical_sha256(evidence)
    return {
        "schema": "flagquantum.cuda_multinode_statevector_artifact.v1",
        "evidence_sha256": digest,
        "evidence": evidence,
    }


def probe(*, network_log: Path | None) -> dict[str, Any] | None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise RuntimeError("probe requires exactly two torchrun ranks")
    if int(os.environ.get("LOCAL_WORLD_SIZE", "1")) != 1:
        raise RuntimeError("probe requires exactly one rank per node")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    payload: dict[str, Any] | None = None
    try:
        circuit = _circuit()
        result = execute_torch_distributed_statevector(
            circuit,
            device=device,
            dtype=torch.complex128,
            local_world_size=1,
        )
        torch.cuda.synchronize(device)
        if result.shard_state.amplitudes.dtype != torch.complex128:
            raise RuntimeError("distributed shard lost complex128 precision")
        if result.shard_state.amplitudes.device.type != "cuda":
            raise RuntimeError("distributed shard left CUDA")
        if result.inter_node_communication_count < 1:
            raise RuntimeError("workload did not exercise inter-node communication")

        candidate = _validation_state(result)
        reference = _circuit().state()[0].to(device=device)
        max_abs_error = float(torch.max(torch.abs(candidate - reference)).item())
        norm_error = float(torch.abs(torch.linalg.vector_norm(candidate) - 1).item())
        infidelity = float(
            1 - torch.abs(torch.vdot(reference, candidate)).square().item()
        )
        metrics = {
            "statevector_max_abs_error": max_abs_error,
            "statevector_norm_error": norm_error,
            "statevector_infidelity": max(0.0, infidelity),
        }
        if max_abs_error > 1e-10 or norm_error > 1e-10 or infidelity > 1e-10:
            raise RuntimeError(f"distributed numerical validation failed: {metrics}")

        properties = torch.cuda.get_device_properties(device)
        record = result.summary()
        record.update(
            {
                "hostname_sha256": hashlib.sha256(
                    platform.node().encode("utf-8")
                ).hexdigest(),
                "device_name": properties.name,
                "device_uuid": str(getattr(properties, "uuid", "")),
                "owned_global_basis_indices": (
                    f"{rank} + k*{result.plan.world_size}, "
                    f"0 <= k < {result.shard_state.shard.local_amplitudes}"
                ),
            }
        )
        rank_records: list[dict[str, Any] | None] = [None, None]
        dist.all_gather_object(rank_records, record)
        if rank == 0:
            payload = _artifact(
                rank_records=[item for item in rank_records if item is not None],
                metrics=metrics,
                network={},
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()

    if rank == 0 and payload is not None:
        network = _network_observation(network_log)
        evidence = payload["evidence"]
        evidence["observations"]["network"] = network
        payload["evidence_sha256"] = _canonical_sha256(evidence)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--network-log",
        type=Path,
        help="rank-zero NCCL debug log used to verify the selected network route",
    )
    args = parser.parse_args()
    payload = probe(network_log=args.network_log)
    if payload is None:
        return 0
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
