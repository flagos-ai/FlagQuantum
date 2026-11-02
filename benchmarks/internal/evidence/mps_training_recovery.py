"""Fixed-workload multi-step extension of the MPS acceptance producer.

Run via ``mps_training_acceptance --recovery-protocol``. The existing capacity
series supervisor owns accelerator reservations, staging, deadlines and logs.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
import torch.distributed as dist

import flagquantum.experimental.distributed as fqxd
import flagquantum.runtime as fqr
from benchmarks.internal.evidence.mps_training_acceptance import build_circuit
from benchmarks.training_comparison import source_identity
from flagquantum.runtime.executors.mps.distributed_state import ShardedMPSState

ROOT = Path(__file__).resolve().parents[3]


def parameters(
    protocol: dict[str, Any], device: torch.device
) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.tensor(value, dtype=torch.float64, device=device, requires_grad=True)
        for value in protocol["initial_parameters"]
    )


def circuit(values: tuple[torch.Tensor, ...], protocol: dict[str, Any]) -> Any:
    return build_circuit(
        values,
        protocol["n_sites"],
        protocol["layers"],
        protocol["family"],
        dtype=torch.complex128,
    )


def local_value(
    values: tuple[torch.Tensor, ...], protocol: dict[str, Any]
) -> tuple[torch.Tensor, Any]:
    state = fqr.run_native(
        circuit(values, protocol),
        mode="mps",
        max_bond=protocol["max_bond"],
        cutoff=protocol["cutoff"],
        truncated_gradient="projected_stop_subspace",
    )
    loss = (
        -sum(
            state.expectation_ps(z=(wire,)).sum() for wire in range(protocol["n_sites"])
        )
        / protocol["n_sites"]
    )
    return loss, state


def local_diagnostic(values: list[float], protocol: dict[str, Any]) -> dict[str, Any]:
    """Compare projected local VJP with finite differences of truncated forward."""
    params = tuple(
        torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in values
    )
    loss, state = local_value(params, protocol)
    gradients = torch.autograd.grad(loss, params)
    epsilon = protocol["finite_difference_epsilon"]
    finite_differences = []
    with torch.no_grad():
        for index in range(len(values)):
            upper = [torch.tensor(v, dtype=torch.float64) for v in values]
            lower = [torch.tensor(v, dtype=torch.float64) for v in values]
            upper[index] += epsilon
            lower[index] -= epsilon
            high, _ = local_value(tuple(upper), protocol)
            low, _ = local_value(tuple(lower), protocol)
            finite_differences.append(float((high - low) / (2 * epsilon)))
    return {
        "parameters": values,
        "objective": float(loss.detach()),
        "projected_gradients": [float(g) for g in gradients],
        "truncated_forward_finite_difference": finite_differences,
        "derivative_semantics_max_abs_difference": max(
            abs(float(g) - fd)
            for g, fd in zip(gradients, finite_differences, strict=True)
        ),
        "max_bond": max(state.bond_dims),
        "finite_difference_epsilon": epsilon,
        "scope": "independent local MPS execution at identical parameters; outside distributed timing; approximate VJP disagreement with finite differences is not sharding error",
    }


def tensor_tree(value: Any) -> Any:
    """Retain compact owner optimizer state, never serialize a full MPS."""
    if isinstance(value, torch.Tensor):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.detach().cpu().tolist(),
        }
    if isinstance(value, dict):
        return {str(k): tensor_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_tree(v) for v in value]
    return value


def tree_max_error(left: Any, right: Any) -> float:
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            raise ValueError("checkpoint state keys differ")
        return max((tree_max_error(left[k], right[k]) for k in left), default=0.0)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError("checkpoint state lengths differ")
        return max(
            (tree_max_error(a, b) for a, b in zip(left, right, strict=True)),
            default=0.0,
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(left - right)
    if left != right:
        raise ValueError("checkpoint state metadata differ")
    return 0.0


def checkpoint(root: Path, rank: int, step: int) -> dict[str, Any]:
    path = root / f"rank-{rank}-step-{step}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    manifest = json.loads((root / "COMMITTED.json").read_text())
    assert manifest["completed_steps"] == step
    assert payload["rank"] == rank and payload["completed_steps"] == step
    return {
        "owned_indices": list(payload["owned_indices"]),
        "parameters": tensor_tree(payload["parameters"]),
        "optimizer_state": tensor_tree(payload["optimizer_state"]),
        "bond_layout": tensor_tree(payload["bond_layout"]),
        "shard_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(
            (root / "COMMITTED.json").read_bytes()
        ).hexdigest(),
        "schema": payload["schema_version"],
        "world_size": payload["world_size"],
    }


def execute(
    protocol: dict[str, Any],
    device: torch.device,
    root: Path,
    steps: int,
    *,
    resume: bool = False,
) -> tuple[Any, list[float], float]:
    params = parameters(protocol, device)
    dist.barrier()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with ExitStack() as stack:

        def forbidden(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError(
                "full-state gather/object gather forbidden inside sharded training"
            )

        stack.enter_context(patch.object(ShardedMPSState, "gather_tensors", forbidden))
        stack.enter_context(patch.object(dist, "all_gather_object", forbidden))
        stack.enter_context(patch.object(dist, "gather_object", forbidden))
        result = fqxd.train_distributed_mps(
            circuit(params, protocol),
            steps=steps,
            optimizer=protocol["optimizer"],
            lr=protocol["learning_rate"],
            device=device,
            hamiltonian_terms=[
                ({wire: "z"}, -1 / protocol["n_sites"])
                for wire in range(protocol["n_sites"])
            ],
            max_bond=protocol["max_bond"],
            cutoff=protocol["cutoff"],
            gradient_policy=protocol["gradient_policy"],
            gradient_tolerance=protocol["gradient_tolerance"],
            canonicalization_policy=protocol["canonicalization_policy"],
            compile_site_kernels=False,
            prefetch_layer_halos=False,
            fuse_local_reverse=False,
            record_parameter_gradients=True,
            checkpoint_dir=root,
            checkpoint_interval=protocol["checkpoint_step"],
            checkpoint_retention_generations=None,
            memory_leak_tolerance_bytes=64 << 20,
            resume=resume,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dist.barrier()
    elapsed = time.perf_counter() - started
    return result, [float(p.detach()) for p in params], elapsed


def run(protocol_path: Path, output: Path) -> None:
    protocol = json.loads(protocol_path.read_text())
    source = source_identity(protocol_path)
    for path in (
        Path(__file__),
        ROOT / "benchmarks/internal/evidence/mps_training_acceptance.py",
        ROOT / "tools/run_mps_capacity_series.py",
    ):
        source["producer_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = os.environ.get("FQ_MPS_TRAINING_BACKEND", "nccl")
    device = (
        torch.device("cuda", local_rank) if backend == "nccl" else torch.device("cpu")
    )
    if backend == "nccl":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(backend, timeout=timedelta(seconds=600))
    root = output.parent / (output.stem + "-checkpoints")
    rank_path = output.parent / (output.stem + f"-rank-{rank:02d}.json")
    rank_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(
            json.dumps(
                {"rank": rank, "phase": "uninterrupted", "steps": protocol["steps"]}
            ),
            flush=True,
        )
        reference, reference_params, reference_wall = execute(
            protocol, device, root / "uninterrupted", protocol["steps"]
        )
        reference_checkpoint = checkpoint(
            root / "uninterrupted", rank, protocol["steps"]
        )
        print(json.dumps({"rank": rank, "phase": "checkpoint_prefix"}), flush=True)
        prefix, _, prefix_wall = execute(
            protocol, device, root / "restarted", protocol["checkpoint_step"]
        )
        prefix_checkpoint = checkpoint(
            root / "restarted", rank, protocol["checkpoint_step"]
        )
        print(json.dumps({"rank": rank, "phase": "restore_and_continue"}), flush=True)
        resumed, resumed_params, resumed_wall = execute(
            protocol, device, root / "restarted", protocol["steps"], resume=True
        )
        resumed_checkpoint = checkpoint(root / "restarted", rank, protocol["steps"])
        reference_steps = [asdict(s) for s in reference.steps]
        restarted_steps = [asdict(s) for s in (*prefix.steps, *resumed.steps)]
        assert resumed.start_step == protocol["checkpoint_step"]
        errors = {
            "objective": max(
                abs(a["loss"] - b["loss"])
                for a, b in zip(reference_steps, restarted_steps, strict=True)
            ),
            "gradient": tree_max_error(
                [tensor_tree(s["parameter_gradients"]) for s in reference_steps],
                [tensor_tree(s["parameter_gradients"]) for s in restarted_steps],
            ),
            "parameters": tree_max_error(reference_params, resumed_params),
            "optimizer_state": tree_max_error(
                reference_checkpoint["optimizer_state"],
                resumed_checkpoint["optimizer_state"],
            ),
            "checkpoint_parameters": tree_max_error(
                reference_checkpoint["parameters"], resumed_checkpoint["parameters"]
            ),
        }
        assert (
            reference_checkpoint["owned_indices"]
            == resumed_checkpoint["owned_indices"]
            == prefix_checkpoint["owned_indices"]
        )
        record = {
            "rank": rank,
            "local_rank": local_rank,
            "node_rank": int(os.environ.get("GROUP_RANK", "0")),
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
            "device_uuid": (
                str(getattr(torch.cuda.get_device_properties(device), "uuid", ""))
                if device.type == "cuda"
                else None
            ),
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            "uninterrupted": reference.summary(),
            "restarted_prefix": prefix.summary(),
            "resumed": resumed.summary(),
            "checkpoint_prefix": prefix_checkpoint,
            "checkpoint_reference": reference_checkpoint,
            "checkpoint_resumed": resumed_checkpoint,
            "reference_parameters_final": reference_params,
            "resumed_parameters_final": resumed_params,
            "recovery_errors": errors,
            "recovery_passed": max(errors.values()) <= protocol["comparison_atol"],
            "barrier_bracketed_call_seconds": {
                "uninterrupted": reference_wall,
                "prefix": prefix_wall,
                "resumed": resumed_wall,
            },
            "synchronization": "barriers bracket calls; runtime synchronizes accelerator forward/reverse/optimizer phases",
            "full_state_gather_forbidden_during_training": True,
        }
        rank_path.write_text(json.dumps(record, indent=2) + "\n")
        gathered: list[Any] = [None] * dist.get_world_size()
        # Compact scalar diagnostics only, after training; no site tensor or state.
        dist.all_gather_object(gathered, record)
        if rank == 0:
            step_parameters = []
            for step in range(protocol["steps"]):
                by_index = {
                    int(i): float(v)
                    for r in gathered
                    for i, v in r["uninterrupted"]["step_metrics"][step][
                        "parameter_values"
                    ]
                }
                step_parameters.append(
                    [by_index[i] for i in range(len(protocol["initial_parameters"]))]
                )
            diagnostics = [
                local_diagnostic(values, protocol) for values in step_parameters
            ]
            for step, diagnostic in enumerate(diagnostics):
                gradients = {
                    int(i): float(v)
                    for r in gathered
                    for i, v in r["uninterrupted"]["step_metrics"][step][
                        "parameter_gradients"
                    ]
                }
                diagnostic["sharding_objective_max_abs_error"] = max(
                    abs(
                        r["uninterrupted"]["step_metrics"][step]["loss"]
                        - diagnostic["objective"]
                    )
                    for r in gathered
                )
                diagnostic["sharding_projected_gradient_max_abs_error"] = max(
                    abs(gradients[i] - value)
                    for i, value in enumerate(diagnostic["projected_gradients"])
                )
            payload = {
                "benchmark": "mps_training_recovery",
                "source": source,
                "protocol": protocol,
                "protocol_canonical_sha256": hashlib.sha256(
                    json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "status": "completed",
                "backend": backend,
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "world_size": dist.get_world_size(),
                "local_world_size": int(os.environ.get("LOCAL_WORLD_SIZE", "1")),
                "node_count": int(os.environ.get("GROUP_WORLD_SIZE", "1")),
                "distribution_semantics": (
                    "sharded_across_ranks"
                    if dist.get_world_size() > 1
                    else "single_rank_distributed_baseline"
                ),
                "claim_evidence_type": "development_smoke",
                "scalability_claim_allowed": False,
                "release_gate_allowed": False,
                "blockers": [
                    "bounded_development_comparison_not_release_certification",
                    "capacity_leg_is_separate_single_step_evidence",
                ],
                "ranks": gathered,
                "matched_local_mps": diagnostics,
                "recovery_passed": all(r["recovery_passed"] for r in gathered),
                "convergence_target": protocol["objective_target"],
                "stopping_reason": "fixed_step_budget_exhausted",
                "timing_scope": "uninterrupted step metrics; checkpoint IO and barrier-bracketed call wall are reported separately; first step retained",
                "amdahl_estimate": None,
                "amdahl_reason": "not measured or fitted; no inferred parallel fraction",
            }
            output.write_text(json.dumps(payload, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "phase": "completed",
                        "world_size": dist.get_world_size(),
                        "recovery_passed": payload["recovery_passed"],
                    }
                ),
                flush=True,
            )
    except BaseException as exc:
        rank_path.write_text(
            json.dumps(
                {
                    "source": source,
                    "protocol": protocol,
                    "rank": rank,
                    "status": (
                        "cuda_oom"
                        if isinstance(exc, torch.OutOfMemoryError)
                        else "failed"
                    ),
                    "error": repr(exc),
                },
                indent=2,
            )
            + "\n"
        )
        raise
    finally:
        dist.destroy_process_group()
